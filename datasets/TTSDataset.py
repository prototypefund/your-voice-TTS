import functools
import os
import numpy as np
import collections
import torch
import random
from torch.utils.data import Dataset

from utils.generic_utils import sequence_mask
from utils.text import text_to_sequence, phoneme_to_sequence, pad_with_eos_bos
from utils.data import prepare_data, prepare_tensor, prepare_stop_target, \
    prepare_alignment_target


class MyDataset(Dataset):
    def __init__(self,
                 outputs_per_step,
                 text_cleaner,
                 ap,
                 meta_data,
                 batch_group_size=0,
                 min_seq_len=0,
                 max_seq_len=float("inf"),
                 use_phonemes=True,
                 phoneme_cache_path=None,
                 phoneme_language="en-us",
                 enable_eos_bos=False,
                 verbose=False):
        """
        Args:
            outputs_per_step (int): number of time frames predicted per step.
            text_cleaner (str): text cleaner used for the dataset.
            ap (TTS.utils.AudioProcessor): audio processor object.
            meta_data (list): list of dataset instances.
            batch_group_size (int): (0) range of batch randomization after sorting
                sequences by length.
            min_seq_len (int): (0) minimum sequence length to be processed
                by the loader.
            max_seq_len (int): (float("inf")) maximum sequence length.
            use_phonemes (bool): (true) if true, text converted to phonemes.
            phoneme_cache_path (str): path to cache phoneme features.
            phoneme_language (str): one the languages from
                https://github.com/bootphon/phonemizer#languages
            enable_eos_bos (bool): enable end of sentence and beginning of sentences characters.
            verbose (bool): print diagnostic information.
        """
        self.batch_group_size = batch_group_size
        self.items = meta_data
        self.outputs_per_step = outputs_per_step
        self.sample_rate = ap.sample_rate
        self.cleaners = text_cleaner
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.ap = ap
        self.use_phonemes = use_phonemes
        self.phoneme_cache_path = phoneme_cache_path
        self.phoneme_language = phoneme_language
        self.enable_eos_bos = enable_eos_bos
        self.verbose = verbose
        if use_phonemes and not os.path.isdir(phoneme_cache_path):
            os.makedirs(phoneme_cache_path, exist_ok=True)
        if self.verbose:
            print("\n > DataLoader initialization")
            print(" | > Use phonemes: {}".format(self.use_phonemes))
            if use_phonemes:
                print("   | > phoneme language: {}".format(phoneme_language))
            print(" | > Number of instances : {}".format(len(self.items)))
        self.index = []
        self.make_index()

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    @staticmethod
    def load_np(filename):
        data = np.load(filename).astype('float32')
        return data

    def _generate_and_cache_phoneme_sequence(self, text, cache_path):
        """generate a phoneme sequence from text.

        since the usage is for subsequent caching, we never add bos and
        eos chars here. Instead we add those dynamically later; based on the
        config option."""
        phonemes = phoneme_to_sequence(text, [self.cleaners],
                                       language=self.phoneme_language,
                                       enable_eos_bos=False)
        phonemes = np.asarray(phonemes, dtype=np.int32)
        np.save(cache_path, phonemes)
        return phonemes

    def _load_or_generate_phoneme_sequence(self, wav_file, text):
        file_name = os.path.basename(wav_file).split('.')[0]
        cache_path = os.path.join(self.phoneme_cache_path,
                                  file_name + '_phoneme.npy')
        try:
            phonemes = np.load(cache_path)
        except FileNotFoundError:
            phonemes = self._generate_and_cache_phoneme_sequence(text,
                                                                 cache_path)
        except (ValueError, IOError):
            print(" > ERROR: failed loading phonemes for {}. "
                  "Recomputing.".format(wav_file))
            phonemes = self._generate_and_cache_phoneme_sequence(text,
                                                                 cache_path)
        if self.enable_eos_bos:
            phonemes = pad_with_eos_bos(phonemes)
            phonemes = np.asarray(phonemes, dtype=np.int32)

        return phonemes

    @functools.lru_cache(50000)
    def load_data(self, idx):
        text, wav_file, speaker_name = self.items[idx]
        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)

        if self.use_phonemes:
            text = self._load_or_generate_phoneme_sequence(wav_file, text)
        else:
            text = np.asarray(
                text_to_sequence(text, [self.cleaners]), dtype=np.int32)

        assert text.size > 0, self.items[idx][1]
        assert wav.size > 0, self.items[idx][1]

        mel = self.ap.melspectrogram(wav).astype('float32')
        # linear = self.ap.spectrogram(w).astype('float32')

        sample = {
            'text': text,
            'mel': mel,
            'item_idx': self.items[idx][1],
            'speaker_name': speaker_name
        }
        return sample

    def make_index(self):
        r"""Sort instances based on text length in ascending order"""
        lengths = np.array([len(ins[0]) for ins in self.items])

        idxs = np.argsort(lengths)
        new_index = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len or length > self.max_seq_len:
                ignored.append(idx)
            else:
                new_index.append(idx)

        # shuffle batch groups
        if self.batch_group_size > 0:
            for i in range(len(new_index) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_index = new_index[offset:end_offset]
                random.shuffle(temp_index)
                new_index[offset:end_offset] = temp_index
        self.index = new_index

        if self.verbose:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(" | > Num. instances discarded by max-min (max={}, min={}) seq limits: {}".format(
                self.max_seq_len, self.min_seq_len, len(ignored)))
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.load_data(self.index[idx])

    def collate_fn(self, batch):
        r"""
            Perform preprocessing and create a final data batch:
            1. PAD sequences with the longest sequence in the batch
            2. Convert Audio signal to Spectrograms.
            3. PAD sequences that can be divided by r.
            4. Convert Numpy to Torch tensors.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):

            text_lenghts = np.array([len(d["text"]) for d in batch])
            text_lenghts, ids_sorted_decreasing = torch.sort(
                torch.LongTensor(text_lenghts), dim=0, descending=True)

            mel = [batch[idx]['mel'] for idx in ids_sorted_decreasing]
            item_idxs = [
                batch[idx]['item_idx'] for idx in ids_sorted_decreasing
            ]
            text = [batch[idx]['text'] for idx in ids_sorted_decreasing]
            speaker_name = [batch[idx]['speaker_name']
                            for idx in ids_sorted_decreasing]

            mel_lengths = [m.shape[1] + 1 for m in mel]  # +1 for zero-frame

            # compute 'stop token' targets
            stop_targets = [
                np.array([0.] * (mel_len - 1)) for mel_len in mel_lengths
            ]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets,
                                               self.outputs_per_step)

            align_targets = [
                np.array([1.0] * int(text_len)) for text_len in text_lenghts
            ]

            align_targets = prepare_alignment_target(align_targets)

            # PAD sequences with largest length of the batch
            text = prepare_data(text).astype(np.int32)
            # wav = prepare_data(wav)

            # PAD features with largest length + a zero frame
            # linear = prepare_tensor(linear, self.outputs_per_step)
            mel = prepare_tensor(mel, self.outputs_per_step)
            # assert mel.shape[2] == linear.shape[2]
            timesteps = mel.shape[2]

            # B x T x D
            # linear = linear.transpose(0, 2, 1)
            mel = mel.transpose(0, 2, 1)

            # mask that zeros out alignment in the spec padding zone, and for
            # the border of text padding
            align_mask = np.ones_like(align_targets)
            # finding those alignment that need masking (the longest don't)
            textlen_needs_masking = (text_lenghts < align_targets.shape[1]).nonzero().squeeze(1)
            # setting the mask at the border of text and text padding
            align_mask[textlen_needs_masking, text_lenghts[textlen_needs_masking]] = 0
            align_mask = np.expand_dims(align_mask, 1)
            max_decoder_steps = mel.shape[1] // self.outputs_per_step
            align_mask = np.repeat(align_mask, max_decoder_steps, 1)
            align_mask = torch.BoolTensor(align_mask)
            decoder_steps = torch.ceil(torch.LongTensor(mel_lengths).float() / self.outputs_per_step).long()
            seq_mask = sequence_mask(decoder_steps, max_decoder_steps)
            seq_mask = seq_mask.unsqueeze(2).expand(-1, -1, align_mask.size(2))
            align_mask = align_mask * seq_mask

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            # linear = torch.FloatTensor(linear).contiguous()
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)
            align_targets = torch.FloatTensor(align_targets)
            linear = None

            return text, text_lenghts, speaker_name, linear, mel, mel_lengths, \
                   stop_targets, align_targets, align_mask, item_idxs

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(type(batch[0]))))
