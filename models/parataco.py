from math import sqrt
from torch import nn
import torch

from layers.slim_gst_layers import GST
from layers.parataco import Encoder, Speedometer, Decoder, Postnet
from utils.generic_utils import sequence_mask


# TODO: match function arguments with tacotron
class ParaTaco(nn.Module):
    """A smaller version of the tacotron 2 network."""
    def __init__(self,
                 num_chars,
                 r,
                 prenet_type="original",
                 prenet_dropout=0.5,
                 encoder_dropout=0.25,
                 postnet_dropout=0.25):
        super(ParaTaco, self).__init__()
        self.n_mel_channels = 80
        self.n_frames_per_step = r
        self.embedding_size = 256
        self.style_dim = 128
        self.speaker_dim = 64

        self.embedding = nn.Embedding(num_chars, self.embedding_size)

        std = sqrt(2.0 / (num_chars + self.embedding_size))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(self.embedding_size, num_convs=5, kernel_size=3,
                               dropout=encoder_dropout)
        self.speedometer = Speedometer(self.embedding_size)

        self.decoder = Decoder(self.embedding_size, self.n_mel_channels, r,
                               prenet_type, prenet_dropout)
        self.postnet = Postnet(self.n_mel_channels, dropout=postnet_dropout)

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet

    def forward(self, text, text_lengths, mel_specs=None, speaker_ids=None,
                teacher_keep_rate=1.0):
        embedded_inputs = self.embedding(text)
        encoder_outputs = self.encoder(embedded_inputs)
        contexts, alignments = self.speedometer(encoder_outputs,
                                            self.n_frames_per_step,
                                            mel_specs.size(1) // self.n_frames_per_step)
        mel_outputs = self.decoder(contexts, mel_specs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet = self.shape_outputs(
            mel_outputs, mel_outputs_postnet)
        return mel_outputs, mel_outputs_postnet, alignments

    def inference(self, text, mel_specs=None, speaker_ids=None):
        embedded_inputs = self.embedding(text)
        encoder_outputs = self.encoder(embedded_inputs)
        contexts, alignments = self.speedometer.inference(encoder_outputs,
                                                      self.n_frames_per_step)
        mel_outputs = self.decoder(contexts, mel_specs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet = self.shape_outputs(
            mel_outputs, mel_outputs_postnet)
        return mel_outputs, mel_outputs_postnet, alignments