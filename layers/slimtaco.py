import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np

from utils.nn import zoneout1, zoneout2
from .common_layers import Linear, LinearBN


class Permutator(nn.Module):
    def __init__(self, target_permutation):
        super(Permutator, self).__init__()
        self.target_permutation = target_permutation

    def forward(self, inputs):
        if len(inputs.shape) == 3:
            out = inputs.permute(*self.target_permutation)
        else:
            out = inputs
        return out


class Prenet(nn.Module):
    def __init__(self,
                 in_features,
                 prenet_type="original",
                 prenet_dropout=0.5,
                 out_features=[256, 256],
                 bias=True):
        super(Prenet, self).__init__()
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        in_features = [in_features] + out_features[:-1]
        if prenet_type == "bn":
            self.layers = nn.ModuleList([
                LinearBN(in_size, out_size, bias=bias)
                for (in_size, out_size) in zip(in_features, out_features)
            ])
        elif prenet_type == "original":
            self.layers = nn.ModuleList([
                nn.Sequential(Linear(in_size, out_size, bias=bias, init_gain="relu"),
                              nn.ReLU(), Permutator((1, 2, 0)),
                              nn.BatchNorm1d(out_size, affine=False),
                              Permutator((2, 0, 1)),
                              nn.Dropout(p=self.prenet_dropout))
                for (in_size, out_size) in zip(in_features, out_features)
            ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinear=None,
                 dropout=0.5):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding)
        norm = nn.BatchNorm1d(out_channels, affine=False)
        dropout = nn.Dropout(p=dropout)
        if nonlinear == 'relu':
            torch.nn.init.kaiming_normal_(conv1d.weight.data, a=0,
                                          nonlinearity=nonlinear)
            torch.nn.init.constant_(conv1d.bias.data, 0)
            self.net = nn.Sequential(conv1d, nn.ReLU(), norm, dropout)
        elif nonlinear == 'tanh':
            self.net = nn.Sequential(conv1d, nn.Tanh(), norm, dropout)
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
        self.var_factor = np.sqrt(0.5)
        convolutions = []
        for _ in range(num_convs):
            convolutions.append(
                ConvBNBlock(in_features, in_features, 5, 'relu',
                            dropout=dropout))
        self.convolutions = nn.ModuleList(convolutions)
        # self.lstm = nn.LSTM(
        #     in_features,
        #     int(in_features / 2),
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True)
        # self.rnn_state = None

    # def forward(self, inputs, input_lengths):
    def forward(self, inputs):
        x = inputs
        for conv in self.convolutions:
            x = conv(x)
        outputs = (x + inputs) * self.var_factor
        outputs = outputs.transpose(1, 2)
        # input_lengths = input_lengths.cpu().numpy()
        # x = nn.utils.rnn.pack_padded_sequence(
        #     x, input_lengths, batch_first=True)
        # self.lstm.flatten_parameters()
        # outputs, _ = self.lstm(x)
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(
        #     outputs,
        #     batch_first=True,
        # )
        return outputs

    def inference(self, x):
        return self.forward(x)
        # for conv in self.convolutions:
        #     x = conv(x)
        # x = x.transpose(1, 2)
        # self.lstm.flatten_parameters()
        # outputs, _ = self.lstm(x)
        # return outputs

    # def inference_truncated(self, x):
    #     """
    #     Preserve encoder state for continuous inference
    #     """
    #     x = self.convolutions(x)
    #     x = x.transpose(1, 2)
    #     self.lstm.flatten_parameters()
    #     outputs, self.rnn_state = self.lstm(x, self.rnn_state)
    #     return outputs


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, in_features, memory_dim, r,
                 prenet_type, prenet_dropout, query_dim, transition_style,
                 lstm_reg="dropout"):
        super(Decoder, self).__init__()
        self.memory_dim = memory_dim
        self.r = r
        self.context_dim = in_features
        self.query_dim = query_dim
        self.decoder_rnn_dim = 512
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1
        self.lstm_reg = lstm_reg

        self.prenet = Prenet(self.memory_dim * r, prenet_type,
                             prenet_dropout,
                             [self.prenet_dim, self.prenet_dim], bias=False)

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + self.context_dim,
                                         self.query_dim)
        # self.attention_rnn = nn.GRUCell(self.prenet_dim + self.context_dim,
        #                                 self.query_dim)
        # self.set_forget_bias(self.attention_rnn)
        # self.init_gru(self.attention_rnn)

        self.attention = SimpleAttention(query_dim=self.query_dim,
                                         context_dim=self.context_dim,
                                         transition_style=transition_style)

        self.decoder_rnn = nn.LSTMCell(self.query_dim + self.context_dim,
                                       self.decoder_rnn_dim, 1)
        # self.decoder_rnn = nn.GRUCell(self.query_dim + self.context_dim,
        #                               self.decoder_rnn_dim, 1)
        # self.set_forget_bias(self.decoder_rnn)
        # self.init_gru(self.decoder_rnn)

        self.linear_projection = Linear(self.decoder_rnn_dim + self.context_dim,
                                        self.memory_dim * r)

        self.attention_rnn_init = nn.Embedding(1, self.query_dim)
        # self.attention_rnn_init.weight.data.normal_(0, 0.7)
        self.context_init = nn.Embedding(1, self.context_dim)
        self.memory_init = nn.Embedding(1, self.memory_dim * self.r)
        self.decoder_rnn_inits = nn.Embedding(1, self.decoder_rnn_dim)
        # self.decoder_rnn_inits.weight.data.normal_(0, 0.7)
        self.memory_truncated = None

    def init_gru(self, gru):
        torch.nn.init.xavier_uniform_(gru.weight_ih.data)
        torch.nn.init.xavier_uniform_(gru.weight_hh.data)
        gru.bias_ih.data.fill_(1e-4)
        gru.bias_hh.data.fill_(1e-4)

    def set_forget_bias(self, lstmcell):
        for names in lstmcell._parameters:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(lstmcell, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def get_memory_start_frame(self, inputs):
        B = inputs.size(0)
        memory = self.memory_init(inputs.data.new_zeros(B).long())
        return memory

    def _init_states(self, inputs, mask, keep_states=False):
        B = inputs.size(0)

        if not keep_states:
            self.query = self.attention_rnn_init(
                inputs.data.new_zeros(B).long())
            self.attention_rnn_cell_state = Variable(
                inputs.data.new(B, self.query_dim).zero_())

            self.decoder_hidden = self.decoder_rnn_inits(
                inputs.data.new_zeros(B).long())
            self.decoder_cell = Variable(
                inputs.data.new(B, self.decoder_rnn_dim).zero_())

            self.context = self.context_init(
                inputs.data.new_zeros(B).long())

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
        query_input = torch.cat((memory, self.context), -1)
        query, attention_rnn_cell_state = self.attention_rnn(
            query_input, (self.query, self.attention_rnn_cell_state))
        # query = self.attention_rnn(query_input, self.query)
        if self.lstm_reg == "dropout":
            self.query = F.dropout(
                query, self.p_attention_dropout, self.training)
            self.attention_rnn_cell_state = F.dropout(
                attention_rnn_cell_state, self.p_attention_dropout,
                self.training)
        elif self.lstm_reg == "zoneout1":
            self.query = zoneout1(query, self.query, self.p_attention_dropout,
                                   self.training)
            self.attention_rnn_cell_state = zoneout1(attention_rnn_cell_state,
                                                     self.attention_rnn_cell_state,
                                                     self.p_attention_dropout,
                                                     self.training)

        self.context = self.attention(self.query, self.inputs)

        memory = torch.cat((self.query, self.context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(
            memory, (self.decoder_hidden, self.decoder_cell))
        # decoder_hidden = self.decoder_rnn(
        #     memory, self.decoder_hidden)
        if self.lstm_reg == "dropout":
            self.decoder_hidden = F.dropout(decoder_hidden,
                                            self.p_decoder_dropout, self.training)
            self.decoder_cell = F.dropout(decoder_cell,
                                          self.p_decoder_dropout, self.training)
        if self.lstm_reg == "zoneout1":
            self.decoder_hidden = zoneout1(decoder_hidden, self.decoder_hidden,
                                           self.p_decoder_dropout, self.training)
            self.decoder_cell = zoneout1(decoder_cell, self.decoder_cell,
                                         self.p_decoder_dropout, self.training)

        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context),
                                           dim=1)

        decoder_output = self.linear_projection(decoder_hidden_context)

        return decoder_output, self.attention.alpha

    def forward(self, inputs, memory, mask, teacher_keep_rate=1.0):
        memory_start = self.get_memory_start_frame(inputs).unsqueeze(0)
        memory_steps = self._reshape_memory(memory)
        memory_steps = torch.cat((memory_start, memory_steps), dim=0)
        memory_steps = self.prenet(memory_steps)

        self._init_states(inputs, mask=mask)
        self.attention.init_states(inputs)

        outputs, alignments = [], []
        while len(outputs) < memory_steps.size(0) - 1:
            if len(outputs) == 0 or teacher_keep_rate == 1.0:
                memory = memory_steps[len(outputs)]
            else:
                given_memory = memory_steps[len(outputs)]
                predicted_memory = self.prenet(outputs[-1])
                memory = teacher_keep_rate * given_memory + \
                         (1.0 - teacher_keep_rate) * predicted_memory
            mel_output, attention_weights = self.decode(memory)
            outputs += [mel_output]
            alignments += [attention_weights]

        outputs, alignments = self._parse_outputs(outputs, alignments)

        return outputs, alignments

    def inference(self, inputs):
        memory = self.get_memory_start_frame(inputs)

        self._init_states(inputs, mask=None)
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

    def inference_truncated(self, inputs):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_memory_start_frame(inputs)
            self._init_states(inputs, mask=None, keep_states=False)
        else:
            self._init_states(inputs, mask=None, keep_states=True)

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
    def __init__(self, query_dim, context_dim,
                 transition_style):
        super(SimpleAttention, self).__init__()
        self.transition_style = transition_style

        self.ta_u = nn.Sequential(Linear(query_dim + context_dim, 1,
                                         bias=True, init_gain="sigmoid"),
                                  nn.Sigmoid())
        if transition_style == "dynamicsq":
            self.ta_sq = nn.Sequential(Linear(query_dim + context_dim, 1,
                                              bias=True, init_gain="sigmoid"),
                                       nn.Sigmoid())
        self.alpha = None
        self.u = None
        self.sq = None

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        S = inputs.size(2)
        self.alpha = torch.cat(
            (torch.ones((B, 1)),
             torch.zeros((B, T))[:, :-1] + 1e-7), dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones((B, 1))).to(inputs.device)
        self.sq = (1.4 * torch.ones((B, 1))).to(inputs.device)

    def forward(self, query, inputs):
        fwd_shifted_alpha = F.pad(
            self.alpha[:, :-1].clone().to(inputs.device),
            [1, 0, 0, 0])
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha
                 + self.u * fwd_shifted_alpha
                 + 1e-6) ** self.sq  # (self.sq + 1 - abs(0.5-self.u))
        # renormalize attention weights
        self.alpha = alpha / alpha.sum(dim=1, keepdim=True)

        context = torch.bmm(self.alpha.unsqueeze(1), inputs)
        context = context.squeeze(1)

        # compute transition
        ta_input = torch.cat((context, query.squeeze(1)), dim=-1)

        self.u = self.ta_u(ta_input)
        if self.transition_style == "dynamicsq":
            self.sq = self.ta_sq(ta_input) + 1.0
        else:
            raise ValueError(f"Transition style ${self.transition_style} unknown")

        return context
