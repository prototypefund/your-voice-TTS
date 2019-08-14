import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
from .common_layers import Prenet, Linear


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


class OrderNet(nn.Module):
    def __init__(self, in_dim, num_convs, kernel_size, out_dim,
                 dropout=0.0):
        super(OrderNet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ConvBNBlock(in_dim, out_dim, kernel_size=kernel_size,
                        nonlinear='relu',
                        dropout=dropout))
        for _ in range(1, num_convs):
            self.convolutions.append(
                ConvBNBlock(out_dim, out_dim, kernel_size=kernel_size,
                            nonlinear='relu',
                            dropout=dropout))

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        return x


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
    def __init__(self, in_features, memory_dim, r, style_dim, speaker_dim,
                 prenet_type, prenet_dropout, query_dim, transition_style,
                 ordered_attn, diff_attn, transition_activation):
        super(Decoder, self).__init__()
        self.memory_dim = memory_dim
        self.r = r
        self.context_dim = in_features
        if ordered_attn:
            self.context_dim += 32
        if diff_attn:
            self.context_dim *= 2
        self.query_dim = query_dim
        self.style_dim = style_dim
        self.speaker_dim = speaker_dim
        self.decoder_rnn_dim = 512
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(self.memory_dim * r, prenet_type,
                             prenet_dropout,
                             [self.prenet_dim, self.prenet_dim], bias=False)

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + self.context_dim + self.style_dim + self.speaker_dim,
                                         self.query_dim)

        self.attention = SimpleAttention(query_dim=self.query_dim,
                                         embedding_dim=in_features,
                                         context_dim=self.context_dim,
                                         style_dim=self.style_dim,
                                         speaker_dim=self.speaker_dim,
                                         transition_style=transition_style,
                                         ordered_attn=ordered_attn,
                                         diff_attn=diff_attn,
                                         transition_activation=transition_activation)

        self.decoder_rnn = nn.LSTMCell(self.query_dim + self.context_dim + self.style_dim + self.speaker_dim,
                                       self.decoder_rnn_dim, 1)

        self.linear_projection = Linear(self.decoder_rnn_dim + self.context_dim + self.style_dim + self.speaker_dim,
                                        self.memory_dim * r)

        self.attention_rnn_init = nn.Embedding(1, self.query_dim)
        self.memory_init = nn.Embedding(1, self.memory_dim * self.r)
        self.decoder_rnn_inits = nn.Embedding(1, self.decoder_rnn_dim)
        self.memory_truncated = None

    def get_memory_start_frame(self, inputs):
        B = inputs.size(0)
        memory = self.memory_init(inputs.data.new_zeros(B).long())
        return memory

    def _init_states(self, inputs, style, speaker, mask, keep_states=False):
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
                inputs.data.new(B, self.context_dim).zero_())

        if style is None:
            self.style = inputs.data.new(B, self.style_dim).zero_()
        else:
            self.style = style

        if speaker is None:
            self.speaker = inputs.data.new(B, self.speaker_dim).zero_()
        else:
            self.speaker = speaker

        self.inputs = inputs
        self.mask = mask

    def _reshape_memory(self, memory):
        memory_steps = memory.view(
            memory.size(0), int(memory.size(1) / self.r), -1)
        memory_steps = memory_steps.transpose(0, 1)
        return memory_steps

    def _parse_outputs(self, outputs, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        outputs = outputs.contiguous().view(outputs.size(0), -1, self.memory_dim)
        outputs = outputs.transpose(1, 2)
        return outputs, alignments

    def decode(self, memory):
        query_input = torch.cat((memory, self.context, self.style, self.speaker), -1)
        self.query, self.attention_rnn_cell_state = self.attention_rnn(
            query_input, (self.query, self.attention_rnn_cell_state))
        self.query = F.dropout(
            self.query, self.p_attention_dropout, self.training)
        self.attention_rnn_cell_state = F.dropout(
            self.attention_rnn_cell_state, self.p_attention_dropout, self.training)

        self.context = self.attention(self.query, self.inputs, self.style, self.speaker)

        memory = torch.cat((self.query, self.context, self.style, self.speaker), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            memory, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell,
                                      self.p_decoder_dropout, self.training)

        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context, self.style, self.speaker),
                                           dim=1)

        decoder_output = self.linear_projection(decoder_hidden_context)

        return decoder_output, self.attention.alpha

    def forward(self, inputs, memory, style, speaker, mask, teacher_keep_rate=1.0):
        memory_start = self.get_memory_start_frame(inputs).unsqueeze(0)
        memory_steps = self._reshape_memory(memory)
        memory_steps = torch.cat((memory_start, memory_steps), dim=0)
        memory_steps = self.prenet(memory_steps)

        self._init_states(inputs, style, speaker, mask=mask)
        self.attention.init_states(inputs)

        outputs, alignments = [], []
        while len(outputs) < memory_steps.size(0) - 1:
            if len(outputs) == 0 or teacher_keep_rate > np.random.random():
                memory = memory_steps[len(outputs)]
            else:
                memory = self.prenet(outputs[-1])
            mel_output, attention_weights = self.decode(memory)
            outputs += [mel_output]
            alignments += [attention_weights]

        outputs, alignments = self._parse_outputs(outputs, alignments)

        return outputs, alignments

    def inference(self, inputs, style, speaker):
        memory = self.get_memory_start_frame(inputs)

        self._init_states(inputs, style, speaker, mask=None)
        self.attention.init_states(inputs)

        outputs, alignments = [], []
        alignment_mass_on_last_char = 0.0
        while alignment_mass_on_last_char < 1.0:
            processed_memory = self.prenet(memory)
            output, alignment = self.decode(processed_memory)
            outputs += [output]
            alignments += [alignment]
            alignment_mass_on_last_char += alignment[0, -1]
            memory = output

            if len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

        outputs, alignments = self._parse_outputs(
            outputs, alignments)

        return outputs, alignments

    def inference_truncated(self, inputs, style, speaker):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_memory_start_frame(inputs)
            self._init_states(inputs, style, speaker, mask=None, keep_states=False)
        else:
            self._init_states(inputs, style, speaker, mask=None, keep_states=True)

        self.attention.init_states(inputs)
        outputs, alignments = [], []
        alignment_mass_on_last_char = 0.0
        while alignment_mass_on_last_char < 1.0:
            memory = self.prenet(self.memory_truncated)
            mel_output, stop_token, alignment = self.decode(memory)
            outputs += [mel_output]
            alignments += [alignment]
            alignment_mass_on_last_char += alignment[0, -1]
            self.memory_truncated = mel_output

            if len(outputs) == self.max_decoder_steps:
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


class SimpleAttention(nn.Module):
    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, query_dim, embedding_dim, context_dim, style_dim, speaker_dim,
                 transition_style, ordered_attn, diff_attn, transition_activation="sigmoid"):
        super(SimpleAttention, self).__init__()
        self.transition_style = transition_style
        self.transition_activation = transition_activation
        init_gain = transition_activation if transition_activation != "softsign" else "tanh"
        self.ordered_attn = ordered_attn
        self.diff_attn = diff_attn

        self.ta_u = nn.Linear(query_dim + context_dim + style_dim + speaker_dim, 1,
                           bias=True)
        if transition_style == "dynamicsq":
            self.ta_sq = nn.Linear(
                query_dim + context_dim + style_dim + speaker_dim, 1,
                bias=True)
        if self.ordered_attn:
            self.order_net = OrderNet(embedding_dim, 1, 3, 32)
        self.alpha = None
        self.u = None
        self.sq = None
        if self.diff_attn:
            self.prev_context = None

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        S = inputs.size(2)
        self.alpha = torch.cat(
            (torch.ones((B, 1)),
             torch.zeros((B, T))[:, :-1] + 1e-7), dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones((B, 1))).to(inputs.device)
        self.sq = (1.4 * torch.ones((B, 1))).to(inputs.device)
        if self.diff_attn:
            self.prev_context=torch.zeros((B, S)).to(inputs.device)

    def forward(self, query, inputs, style, speaker):
        fwd_shifted_alpha = F.pad(
            self.alpha[:, :-1].clone().to(inputs.device),
            [1, 0, 0, 0])
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha
                 + self.u * fwd_shifted_alpha
                 + 1e-6) ** self.sq  # (self.sq + 1 - abs(0.5-self.u))
        # renormalize attention weights
        self.alpha = alpha / alpha.sum(dim=1, keepdim=True)

        if self.ordered_attn:
            attended_inputs = self.alpha.unsqueeze(2) * inputs
            order_information = self.order_net(attended_inputs.transpose(1, 2))
            order_information = torch.sum(order_information, dim=2, keepdim=False)
            context = torch.sum(attended_inputs, dim=1, keepdim=False)
            context = torch.cat((context, order_information), dim=-1)
        else:
            context = torch.bmm(self.alpha.unsqueeze(1), inputs)
            context = context.squeeze(1)

        if self.diff_attn:
            context_diff = context - self.prev_context
            self.prev_context = context
            context = torch.cat((context, context_diff), dim=-1)

        # compute transition
        ta_input = torch.cat((context, query.squeeze(1), style, speaker), dim=-1)
        if self.transition_activation == "sigmoid":
            activation_func = lambda x: torch.sigmoid(x)
        elif self.transition_activation == "tanh":
            activation_func = lambda x: 0.5 + torch.tanh(x) * 0.5
        elif self.transition_activation == "softsign":
            activation_func = lambda x: 0.5 + F.softsign(x) * 0.5
        else:
            raise ValueError(f"Transition activation ${self.transition_activation} unknown.")

        self. u = activation_func(self.ta_u(ta_input))
        if self.transition_style == "dynamicsq":
            self.sq = activation_func(self.ta_sq(ta_input)) + 1.0
        else:
            raise ValueError(f"Transition style ${self.transition_style} unknown")

        return context
