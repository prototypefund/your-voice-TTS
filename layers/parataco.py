import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
from .common_layers import Linear


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
        self.layers = nn.ModuleList([
            Linear(in_size, out_size, bias=bias, init_gain="relu")
            for (in_size, out_size) in zip(in_features, out_features)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.prenet_dropout,
                          training=self.training)
        return x


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


class Speedometer(nn.Module):
    def __init__(self, in_dim):
        super(Speedometer, self).__init__()
        self.speedometer = Linear(in_dim, 1, bias=True, init_gain="sigmoid")

    def step(self, alpha, u, sq):
        fwd_shifted_alpha = F.pad(alpha[:, :-1], [1, 0, 0, 0])
        # compute transition potentials
        alpha = ((1 - u) * alpha
                 + u * fwd_shifted_alpha
                 + 1e-6) ** sq
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, x, r, num_steps):
        B = x.size(0)
        T = x.size(1)
        speeds = torch.sigmoid(self.speedometer(x))
        alphas = []
        alpha = torch.cat(
            (torch.ones((B, 1)),
             torch.zeros((B, T))[:, :-1] + 1e-7), dim=1).to(x.device)
        for _ in range(num_steps):
            u = torch.bmm(alpha.unsqueeze(1), speeds).squeeze(2)
            sq = (torch.ones_like(u) + 0.7) - torch.abs(u - 0.5)
            alpha = self.step(alpha, u, sq)
            alphas.append(alpha)

        alphas = torch.stack(alphas, dim=1)
        contexts = torch.bmm(alphas, x)

        return contexts, alphas

    def inference(self, x, r):
        B = x.size(0)
        T = x.size(1)
        speeds = torch.sigmoid(self.speedometer(x))
        alphas = []
        alpha = torch.cat(
            (torch.ones((B, 1)),
             torch.zeros((B, T))[:, :-1] + 1e-7), dim=1).to(x.device)
        alignment_mass_on_last_char = 0.0
        while alignment_mass_on_last_char < 1.0:
            u = torch.bmm(alpha.unsqueeze(1), speeds).squeeze(2)
            alpha = self.step(alpha, u, 1.4)
            alphas.append(alpha)
            alignment_mass_on_last_char += alpha[0, -1]

        alphas = torch.stack(alphas, dim=1)
        contexts = torch.bmm(alphas, x)

        return contexts, alphas


class Projector(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=[256, 256],
                 bias=True):
        super(Projector, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = nn.ModuleList([
            Linear(in_size, out_size, bias=bias, init_gain="relu")
            for (in_size, out_size) in zip(in_features, out_features)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.relu(linear(x))
        return x


class Encoder(nn.Module):

    def __init__(self, in_features=512, num_convs=3, kernel_size=5, dropout=0.5):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(num_convs):
            convolutions.append(
                ConvBNBlock(in_features, in_features, kernel_size, 'relu',
                            dropout=dropout))
        self.convolutions = nn.Sequential(*convolutions)

    def forward(self, x_in):
        x_in_t = x_in.transpose(1, 2)
        x_out = self.convolutions(x_in_t)
        x_out = x_out.transpose(1, 2)
        return x_in + x_out


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, in_features, memory_dim, r,
                 prenet_type, prenet_dropout):
        super(Decoder, self).__init__()
        self.memory_dim = memory_dim
        self.r = r
        self.context_dim = in_features
        self.prenet_dim = 256

        self.prenet = Prenet(self.memory_dim * r, prenet_type,
                             prenet_dropout,
                             [self.prenet_dim, self.prenet_dim], bias=False)

        self.projector = Projector(self.prenet_dim + in_features,
                                           [self.memory_dim * r, self.memory_dim * r])

        self.memory_init = nn.Embedding(1, self.memory_dim * self.r)

    def get_memory_start_frame(self, inputs):
        B = inputs.size(0)
        memory = self.memory_init(inputs.data.new_zeros(B).long())
        return memory

    def _reshape_memory(self, memory):
        memory_steps = memory.view(
            memory.size(0), int(memory.size(1) / self.r), -1)
        # memory_steps = memory_steps.transpose(0, 1)
        return memory_steps

    def _parse_outputs(self, outputs):
        # outputs = torch.stack(outputs).transpose(0, 1)
        outputs = outputs.contiguous().view(outputs.size(0), -1, self.memory_dim)
        outputs = outputs.transpose(1, 2)
        return outputs

    def forward(self, contexts, memory):
        memory_start = self.get_memory_start_frame(contexts).unsqueeze(1)
        memory_steps = self._reshape_memory(memory)
        memory_steps = torch.cat((memory_start, memory_steps[:, :-1]), dim=1)
        memory_steps = self.prenet(memory_steps)

        outputs = self.projector(torch.cat((contexts, memory_steps), dim=-1))
        outputs = self._parse_outputs(outputs)

        return outputs

    def inference(self, contexts):
        memory = self.get_memory_start_frame(contexts)

        outputs = []
        for context in contexts:
            processed_memory = self.prenet(memory)
            output = self.projector(torch.cat((context, processed_memory), dim=-1))
            outputs += [output]
            memory = output

            if len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

        outputs = self._parse_outputs(outputs)

        return outputs
