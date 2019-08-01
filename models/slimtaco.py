from math import sqrt
from torch import nn
import torch

from layers.slim_gst_layers import GST
from layers.slimtaco import Encoder, Decoder, Postnet
from utils.generic_utils import sequence_mask


# TODO: match function arguments with tacotron
class SlimTaco(nn.Module):
    """A smaller version of the tacotron 2 network."""
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r,
                 attn_win=False,
                 attn_norm="softmax",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 separate_stopnet=True,
                 use_gst=False,
                 memory_size=5):
        super(SlimTaco, self).__init__()
        self.n_mel_channels = 80
        self.n_frames_per_step = r
        self.use_gst = use_gst
        self.embedding_size = 256
        self.style_dim = 128

        self.embedding = nn.Embedding(num_chars, self.embedding_size)

        std = sqrt(2.0 / (num_chars + self.embedding_size))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers,
                                                  self.embedding_size)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(self.embedding_size, dropout=0.25)
        if self.use_gst:
            self.gst = GST(num_mel=self.n_mel_channels, num_heads=4,
                           num_style_tokens=16,
                           embedding_dim=self.style_dim)
        self.decoder = Decoder(self.embedding_size, self.n_mel_channels, r, self.style_dim,
                               prenet_type, prenet_dropout,
                               trans_agent, memory_size)
        self.postnet = Postnet(self.n_mel_channels, dropout=0.25)

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, speaker_ids=None,
                print_norms=False):
        # compute mask for padding
        mask = sequence_mask(text_lengths).to(text.device)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        if self.use_gst and mel_specs is not None:
            gst_outputs = self.gst(mel_specs).squeeze(1)
            if print_norms:
                print(f"gst norm: {torch.norm(gst_outputs[0])}")
                print(f"encoder norm: {torch.norm(encoder_outputs[0, 1])}")
                print(f"dist first, last: {torch.cosine_similarity(encoder_outputs[0,0], encoder_outputs[0,text_lengths[0]-1], dim=0)}")
            # gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            # encoder_outputs = encoder_outputs + gst_outputs
            # if print_norms:
                # print(f"dist first, last, after adding gst: {torch.cosine_similarity(encoder_outputs[0, 0], encoder_outputs[0, text_lengths[0]-1], dim=0)}")
        else:
            gst_outputs = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, mel_specs, gst_outputs, mask)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments

    def inference(self, text, mel_specs=None, speaker_ids=None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)
        if self.use_gst and mel_specs is not None:
            gst_outputs = self.gst(mel_specs).squeeze(1)
            # gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            # encoder_outputs = encoder_outputs + gst_outputs
        else:
            gst_outputs = None

        mel_outputs, alignments = self.decoder.inference(encoder_outputs, gst_outputs)
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
