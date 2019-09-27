from math import sqrt
from torch import nn
import torch

from layers.slim_gst_layers import GST
from layers.slimtaco import Encoder, Decoder, Postnet
from utils.generic_utils import sequence_mask
import torch.nn.functional as F


# TODO: match function arguments with tacotron
class SlimTaco(nn.Module):
    """A smaller version of the tacotron 2 network."""
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r,
                 mel_dim=80,
                 prenet_type="original",
                 prenet_dropout=0.5,
                 encoder_dropout=0.25,
                 postnet_dropout=0.25,
                 query_dim=512,
                 attention_type="simple",
                 num_gaussians=10,
                 normalize_attention=False,
                 use_gst=False,
                 init_embedding=True,
                 decoder_lstm_reg="dropout",
                 embedding_size=256,
                 final_activation=None,
                 max_norm=1.0,
                 symmetric=False):
        super(SlimTaco, self).__init__()
        self.n_mel_channels = mel_dim
        self.n_frames_per_step = r
        self.use_gst = use_gst
        self.style_dim = 128
        self.embedding_size = embedding_size if not use_gst else embedding_size + self.style_dim
        self.speaker_dim = 64
        self.final_activation = final_activation
        self.max_norm = max_norm
        self.symmetric = symmetric

        self.embedding = nn.Embedding(num_chars, self.embedding_size, padding_idx=0)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers,
                                                  self.speaker_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(self.embedding_size, dropout=encoder_dropout)
        if self.use_gst:
            self.gst = GST(num_mel=self.n_mel_channels, num_heads=4,
                           num_style_tokens=16,
                           embedding_dim=self.style_dim)
            # self.gst = GlobalStyleTokens(num_mel=self.n_mel_channels,
            #                              num_style_tokens=10,
            #                              token_dim=128,
            #                              prosody_encoding_dim=128,
            #                              scoring_function_name="tanh",
            #                              use_separate_keys=True)
        self.decoder = Decoder(self.embedding_size, self.n_mel_channels, r,
                               prenet_type, prenet_dropout, query_dim, attention_type,
                               num_gaussians, normalize_attention, decoder_lstm_reg)
        self.postnet = Postnet(self.n_mel_channels, dropout=postnet_dropout)

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, speaker_ids=None,
                teacher_keep_rate=1.0):
        # compute mask for padding
        mask = sequence_mask(text_lengths).to(text.device)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        # encoder_outputs = self._add_speaker_embedding(encoder_outputs,
        #                                               speaker_ids)
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            speaker_embedding = self.speaker_embedding(speaker_ids)
        else:
            speaker_embedding = None

        self._add_style_embedding(encoder_outputs, mel_specs)

        mel_outputs, alignments = self.decoder(
            encoder_outputs, mel_specs, mask, teacher_keep_rate)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        mel_outputs, mel_outputs_postnet = self._scale_final_activation(
            mel_outputs, mel_outputs_postnet)
        return mel_outputs, mel_outputs_postnet, alignments

    def inference(self, text, mel_specs=None, speaker_ids=None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        # encoder_outputs = self._add_speaker_embedding(encoder_outputs,
        #                                               speaker_ids)
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            speaker_embedding = self.speaker_embedding(speaker_ids)
        else:
            speaker_embedding = None

        self._add_style_embedding(encoder_outputs, mel_specs)

        mel_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        mel_outputs, mel_outputs_postnet = self._scale_final_activation(
            mel_outputs, mel_outputs_postnet)
        return mel_outputs, mel_outputs_postnet, alignments

    def inference_truncated(self, text, mel_specs=None, speaker_ids=None):
        """Preserve model states for continuous inference."""

        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        self._add_style_embedding(encoder_outputs, mel_specs)

        mel_outputs, alignments = self.decoder.inference_truncated(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        mel_outputs, mel_outputs_postnet = self._scale_final_activation(
            mel_outputs, mel_outputs_postnet)
        return mel_outputs, mel_outputs_postnet, alignments

    def _add_speaker_embedding(self, encoder_outputs, speaker_ids):
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(" [!] Model has speaker embedding layer but speaker_id is not provided")
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            speaker_embeddings = self.speaker_embedding(speaker_ids)

            speaker_embeddings.unsqueeze_(1)
            speaker_embeddings = speaker_embeddings.expand(encoder_outputs.size(0),
                                                           encoder_outputs.size(1),
                                                           -1)
            encoder_outputs = encoder_outputs + speaker_embeddings
        return encoder_outputs

    def _add_style_embedding(self, encoder_outputs, mel_specs):
        if self.use_gst:
            if mel_specs is not None:
                gst_outputs = self.gst(mel_specs).squeeze(1)
                gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1),
                                                 -1)
            else:
                gst_outputs = torch.zeros((encoder_outputs.size(0),
                                           encoder_outputs.size(1),
                                           self.style_dim))
            encoder_outputs = torch.cat((encoder_outputs, gst_outputs), dim=-1)
        return encoder_outputs

    def _scale_final_activation(self, mel_outputs, mel_outputs_postnet):
        if self.final_activation is not None:
            mel_outputs = torch.sigmoid(mel_outputs) * self.max_norm
            mel_outputs_postnet = torch.sigmoid(mel_outputs_postnet) * self.max_norm
            if self.symmetric:
                mel_outputs = mel_outputs * 2 - self.max_norm
                mel_outputs_postnet = mel_outputs_postnet * 2 - self.max_norm
        return mel_outputs, mel_outputs_postnet