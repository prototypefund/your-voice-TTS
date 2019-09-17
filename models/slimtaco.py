from math import sqrt
from torch import nn
import torch

from layers.slim_gst_layers import GST
from layers.slimtaco import Encoder, Decoder, Postnet
from layers.style_encoder import GlobalStyleTokens
from utils.generic_utils import sequence_mask


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
                 num_gaussians=10,
                 normalize_attention=False,
                 use_gst=False,
                 init_embedding=True,
                 decoder_lstm_reg="dropout",
                 embedding_size=256):
        super(SlimTaco, self).__init__()
        self.n_mel_channels = mel_dim
        self.n_frames_per_step = r
        self.use_gst = use_gst
        self.embedding_size = embedding_size
        self.style_dim = 128
        self.speaker_dim = 64

        self.embedding = nn.Embedding(num_chars, self.embedding_size)
        # make padding char zero
        self.embedding.weight.data[0].fill_(0.0)
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
                               prenet_type, prenet_dropout, query_dim,
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

        if self.use_gst and mel_specs is not None:
            gst_outputs = self.gst(mel_specs).squeeze(1)
            # gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            # encoder_outputs = encoder_outputs + gst_outputs
        else:
            gst_outputs = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, mel_specs, mask, teacher_keep_rate)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
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

        if self.use_gst and mel_specs is not None:
            gst_outputs = self.gst(mel_specs).squeeze(1)
            # gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            # encoder_outputs = encoder_outputs + gst_outputs
        else:
            gst_outputs = None

        mel_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments

    def inference_truncated(self, text, mel_specs=None, speaker_ids=None):
        """Preserve model states for continuous inference."""

        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        if self.use_gst and mel_specs is not None:
            gst_outputs = self.gst(mel_specs).squeeze(1)
            # gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            # encoder_outputs = encoder_outputs + gst_outputs
        else:
            gst_outputs = None

        mel_outputs, alignments = self.decoder.inference_truncated(
            encoder_outputs, gst_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
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
