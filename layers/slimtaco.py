import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .common_layers import Prenet, Linear, SimpleAttention


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinear=None,
                 dropout=0.5):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding)
        norm = nn.BatchNorm1d(out_channels)
        dropout = nn.Dropout(p=dropout)
        if nonlinear == 'relu':
            self.net = nn.Sequential(conv1d, norm, nn.ReLU(), dropout)
        elif nonlinear == 'tanh':
            self.net = nn.Sequential(conv1d, norm, nn.Tanh(), dropout)
        else:
            self.net = nn.Sequential(conv1d, norm, dropout)

    def forward(self, x):
        output = self.net(x)
        return output


class Postnet(nn.Module):
    def __init__(self, mel_dim, num_convs=5, conv_dim=512, dropout=0.5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ConvBNBlock(mel_dim, conv_dim, kernel_size=5, nonlinear='tanh',
                        dropout=dropout))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(
                ConvBNBlock(conv_dim, conv_dim, kernel_size=5, nonlinear='tanh',
                            dropout=dropout))
        self.convolutions.append(
            ConvBNBlock(conv_dim, mel_dim, kernel_size=5, nonlinear=None,
                        dropout=dropout))

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_features=512, num_convs=3, dropout=0.5):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(num_convs):
            convolutions.append(
                ConvBNBlock(in_features, in_features, 5, 'relu',
                            dropout=dropout))
        self.convolutions = nn.Sequential(*convolutions)
        self.lstm = nn.LSTM(
            in_features,
            int(in_features / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.rnn_state = None

    def forward(self, x, input_lengths):
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True,
        )
        return outputs

    def inference(self, x):
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs

    def inference_truncated(self, x):
        """
        Preserve encoder state for continuous inference
        """
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, self.rnn_state = self.lstm(x, self.rnn_state)
        return outputs


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, in_features, memory_dim, r, style_dim,
                 prenet_type, prenet_dropout, trans_agent, memory_size):
        super(Decoder, self).__init__()
        self.memory_dim = memory_dim
        self.r = r
        self.memory_size = memory_size if memory_size > 0 else r
        self.encoder_embedding_dim = in_features
        self.query_dim = 512
        self.style_dim = style_dim
        self.decoder_rnn_dim = 512
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(self.memory_dim * memory_size, prenet_type,
                             prenet_dropout,
                             [self.prenet_dim, self.prenet_dim], bias=False)

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + in_features + self.style_dim,
                                         self.query_dim)

        self.attention = SimpleAttention(query_dim=self.query_dim,
                                         embedding_dim=in_features,
                                         style_dim=self.style_dim,
                                         trans_agent=trans_agent)

        self.decoder_rnn = nn.LSTMCell(self.query_dim + in_features + self.style_dim,
                                       self.decoder_rnn_dim, 1)

        self.linear_projection = Linear(self.decoder_rnn_dim + in_features + self.style_dim,
                                        self.memory_dim * r)

        self.attention_rnn_init = nn.Embedding(1, self.query_dim)
        self.memory_init = nn.Embedding(1, self.memory_dim * self.memory_size)
        self.decoder_rnn_inits = nn.Embedding(1, self.decoder_rnn_dim)
        self.memory_truncated = None

    def get_memory_start_frame(self, inputs):
        B = inputs.size(0)
        memory = self.memory_init(inputs.data.new_zeros(B).long())
        return memory

    def _init_states(self, inputs, style, mask, keep_states=False):
        B = inputs.size(0)
        # T = inputs.size(1)

        if not keep_states:
            self.query = self.attention_rnn_init(
                inputs.data.new_zeros(B).long())
            self.attention_rnn_cell_state = Variable(
                inputs.data.new(B, self.query_dim).zero_())

            self.decoder_hidden = self.decoder_rnn_inits(
                inputs.data.new_zeros(B).long())
            self.decoder_cell = Variable(
                inputs.data.new(B, self.decoder_rnn_dim).zero_())

            self.context = Variable(
                inputs.data.new(B, self.encoder_embedding_dim).zero_())

        if style is None:
            self.style = inputs.data.new(B, self.style_dim).zero_()
        else:
            self.style = style
        self.inputs = inputs
        self.mask = mask

    def _unfold_memory(self, memory):
        """Sliding window over memory to get all memory blocks."""
        B = memory.shape[0]
        # memory (B, timesteps, memory_dim)

        # unfold operator is like a sliding window size mem_size, step r
        # timesteps is divisible by r; guaranteed by data loader
        memory = memory.unfold(1, self.memory_size, self.r)
        # memory (B, T_decoder = timesteps // r, memory_dim, self.r)

        memory = memory.contiguous().view(B, -1,
                                          self.memory_dim * self.memory_size)
        # memory (B, T_decoder, memory_dim * self.r)

        # switch to time first
        memory = memory.transpose(0, 1)
        # memory (T_decoder, B, memory_dim * r)
        return memory

    def _update_memory(self, memory, decoder_output):
        if self.memory_size > 0 and \
                decoder_output.shape[-1] < self.memory_size * self.memory_dim:
            new_memory = torch.cat(
                (memory[:, self.r * self.memory_dim:],
                 decoder_output[:, -self.memory_size * self.memory_dim:]),
                dim=-1)
        else:
            new_memory = decoder_output
        return new_memory

    def _parse_outputs(self, outputs, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        outputs = outputs.contiguous().view(outputs.size(0), -1, self.memory_dim)
        outputs = outputs.transpose(1, 2)
        return outputs, alignments

    def decode(self, memory):
        query_input = torch.cat((memory, self.context, self.style), -1)
        self.query, self.attention_rnn_cell_state = self.attention_rnn(
            query_input, (self.query, self.attention_rnn_cell_state))
        self.query = F.dropout(
            self.query, self.p_attention_dropout, self.training)
        self.attention_rnn_cell_state = F.dropout(
            self.attention_rnn_cell_state, self.p_attention_dropout, self.training)

        self.context = self.attention(self.query, self.inputs, self.style)

        memory = torch.cat((self.query, self.context, self.style), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            memory, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell,
                                      self.p_decoder_dropout, self.training)

        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context, self.style),
                                           dim=1)

        decoder_output = self.linear_projection(decoder_hidden_context)

        return decoder_output, self.attention.alpha

    def forward(self, inputs, memory, style, mask):
        memory_start = self.get_memory_start_frame(inputs)
        memory_start = memory_start.view(inputs.size(0),
                                         self.memory_size, self.memory_dim)
        memory = torch.cat((memory_start, memory), dim=1)
        memories = self._unfold_memory(memory)
        memories = self.prenet(memories)

        self._init_states(inputs, style, mask=mask)
        self.attention.init_states(inputs)

        outputs, alignments = [], []
        while len(outputs) < memories.size(0) - 1:
            memory = memories[len(outputs)]
            mel_output, attention_weights = self.decode(memory)
            outputs += [mel_output]
            alignments += [attention_weights]

        outputs, alignments = self._parse_outputs(outputs, alignments)

        return outputs, alignments

    def inference(self, inputs, style):
        memory = self.get_memory_start_frame(inputs)

        self._init_states(inputs, style, mask=None)
        self.attention.init_states(inputs)

        outputs, alignments = [], []
        while True:
            processed_memory = self.prenet(memory)
            output, alignment = self.decode(processed_memory)
            outputs += [output]
            alignments += [alignment]
            memory = self._update_memory(memory, outputs[-1])

            if F.cosine_similarity(self.context, inputs[:, -1, :]) > 0.9:
                break
            elif len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

        outputs, alignments = self._parse_outputs(
            outputs, alignments)

        return outputs, alignments

    def inference_truncated(self, inputs, style):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_memory_start_frame(inputs)
            self._init_states(inputs, style, mask=None, keep_states=False)
        else:
            self._init_states(inputs, style, mask=None, keep_states=True)

        self.attention.init_states(inputs)
        outputs, alignments = [], []
        while True:
            memory = self.prenet(self.memory_truncated)
            mel_output, stop_token, alignment = self.decode(memory)
            outputs += [mel_output]
            alignments += [alignment]
            self.memory_truncated = self._update_memory(self.memory_truncated,
                                                        outputs[-1])

            if F.cosine_similarity(self.context, inputs[:, -1, :]) > 0.9:
                break
            elif len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

        outputs, alignments = self._parse_outputs(
            outputs, alignments)

        return outputs, alignments

    def inference_step(self, inputs, t, memory=None):
        """
        For debug purposes
        """
        if t == 0:
            memory = self.get_memory_start_frame(inputs)
            self._init_states(inputs, mask=None)

        memory = self.prenet(memory)
        mel_output, alignment = self.decode(memory)
        return mel_output, alignment
