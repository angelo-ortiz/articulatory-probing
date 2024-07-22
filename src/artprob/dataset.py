import math
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import textgrids
import torch
import torchaudio
from jaxtyping import Float, Integer
from panphon import FeatureTable
from scipy.interpolate import interp1d
from scipy.signal import resample
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from artprob.inventory import diph_english_inventory, french_inventory, \
    uk_english_inventory
from artprob.model import SAMPLING_RATE


class Extension(str, Enum):
    EMA = '.ema'
    PARAM = '.param'
    WAV = '.wav'
    TXG = '.TextGrid'
    PHONE = '.phone'
    PHNM = '.phnm'
    TRANS = '.trans'


EMA_TRACES = [
    'ul_x',
    'ul_y',
    'll_x',
    'll_y',
    'li_x',
    'li_y',
    'tt_x',
    'tt_y',
    'tb_x',
    'tb_y',
    'td_x',
    'td_y'
]

ART_PARAMS = ['jh', 'tb', 'td', 'tt', 'lp', 'lh']


def find_file_paths(
        path: Path,
        art_ext: str = Extension.EMA,
        other_exts: tuple[Extension, ...] = (Extension.WAV, Extension.TXG)
) -> list[tuple[Path, ...]]:
    all_paths = []
    # recursively go through all the files with the sought extension `ext`
    for f in path.rglob('*' + art_ext):
        all_ext_files = [f]

        # check that `f` has a correspondence in *all* the other extensions
        for oth_ext in other_exts:
            oth_f = f.with_suffix(oth_ext)
            if not oth_f.exists():
                break
            all_ext_files.append(oth_f)

        # add the file tuple to the list of valid files
        if len(all_ext_files) == len(other_exts) + 1:
            all_paths.append(tuple(all_ext_files))

    return all_paths


class ArtLikeRecording:
    def __init__(
            self,
            path: Path,
            ema_not_art: bool,
            sample_rate: int = 500,
            downsample_method: str = 'fourier'
    ):
        self.path = path
        self.ema_not_art = ema_not_art
        self.data = None
        self.sample_rate = sample_rate
        self.downsample_method = downsample_method
        self.num_samples = -1

    def read_ema(self) -> None:
        tr_channels = [-1] * len(EMA_TRACES)
        with open(self.path, 'rb') as fp:
            for line in fp:
                line = line.decode()
                if line.startswith('EST_Header_End'):
                    break
                if line.startswith('NumFrames'):
                    num_samples = int(line.split()[1])
                elif line.startswith('NumChannels'):
                    # num_channels = int(line.split()[1])
                    pass
                elif line.startswith('Channel_'):
                    channel, trace = line.split()
                    try:
                        idx = EMA_TRACES.index(trace)

                        # the first 2 channels are irrelevant
                        tr_channels[idx] = 2 + int(channel.split('_')[1])
                    except ValueError:
                        pass

            complete = -1 not in tr_channels

            if not complete:
                indices = np.nonzero(np.array(complete) == -1)[0]
                missing = [EMA_TRACES[i] for i in indices]
                raise IOError(
                    f'The following EMA traces are missing in the file {self.path}: '
                    ','.join(missing)
                )

            data = np.fromfile(fp, np.float32).reshape(num_samples, -1)
            self.data = data[:, tr_channels]

    def read_art(self):
        self.data = np.fromfile(self.path, dtype=np.float32).reshape(-1, 6)

    def _down_sample_with_sample_rate(self, target_sample_rate: int) -> None:
        if self.sample_rate == target_sample_rate:
            return

        ds_num_frames = math.ceil(
            self.num_samples * target_sample_rate / self.sample_rate
        )
        match self.downsample_method:
            case 'fourier':
                self.data = resample(self.data, ds_num_frames, axis=0)
            case kind:
                # select the samples with `target_sample_rate` jumps
                interp_fun = interp1d(
                    np.arange(self.num_samples), self.data, kind, axis=0
                )
                ds_stop = (ds_num_frames - 1) * self.sample_rate \
                          / target_sample_rate
                self.data = interp_fun(np.linspace(0, ds_stop, num=ds_num_frames))

    def _down_sample_with_num_samples(self, target_num_samples: int) -> None:
        match self.downsample_method:
            case 'fourier':
                self.data = resample(self.data, target_num_samples, axis=0)
            case kind:
                # select the samples s.t. the first and last samples are the boundaries
                # of the `target_num_samples` expected ones
                interp_fun = interp1d(
                    np.arange(self.num_samples), self.data, kind, axis=0
                )
                self.data = np.apply_along_axis(
                    interp_fun,
                    0,
                    np.linspace(0, self.num_samples - 1, num=target_num_samples)
                )

    def process(
            self,
            sample_rate: Optional[int] = None,
            num_samples: Optional[int] = None
    ) -> Float[np.ndarray, "seq {len(EMA_TRACES)}"]:
        # read the data
        if self.ema_not_art:
            self.read_ema()

            # normalise
            self.data = (self.data - self.data.mean(axis=0, keepdims=True)) \
                        / self.data.std(axis=0, keepdims=True)
        else:
            # add resampling to articulatory parameters
            self.read_art()

        self.num_samples = self.data.shape[0]

        # down-sample
        match sample_rate, num_samples:
            case None, None:
                # put off the down-sampling to after dataset creation (with PyTorch)
                print('Warning: no down-sampling on EMA data!')
            case tar_sample_rate, None:
                self._down_sample_with_sample_rate(tar_sample_rate)
            case None, tar_num_samples:
                self._down_sample_with_num_samples(tar_num_samples)
            case _:
                raise ValueError(
                    f'Received both a target sample rate ({sample_rate} Hz) and number '
                    f'of samples ({num_samples}): choose at most one'
                )

        nan_mask = np.isnan(self.data)
        if nan_mask.sum() > 0:
            print(self.path, np.argwhere(nan_mask))

        return self.data


class EMADataset:
    def __init__(self, file_paths: list[tuple[Path, ...]]):
        self.ema_paths = [pt[0] for pt in file_paths]

    def fetch_batch(
            self,
            indices: Integer[np.ndarray, "batch"],
            num_samples: Integer[np.ndarray, "batch"],
    ) -> Float[torch.Tensor, "batch seq {len(EMA_TRACES)}"]:
        emas = []
        for idx, n_samples in zip(indices, num_samples):
            emas.append(
                ArtLikeRecording(self.ema_paths[idx], ema_not_art=True)
                .process(num_samples=n_samples)
            )

        emas = [torch.from_numpy(_ema) for _ema in emas]
        emas = pad_sequence(emas, batch_first=True)

        return emas


class PhonologicalFeatures:
    def __init__(self, file_path: str):
        phon_features = np.load(file_path, allow_pickle=True)
        self._phonemes = phon_features['phonemes']
        self._phon_features = torch.from_numpy(phon_features['features'])

    @property
    def num_features(self) -> int:
        return self._phon_features.size(1)

    def project_features(
            self,
            non_negative_feats: bool,
            keep_unknown_phon_feats: bool
    ) -> None:
        """Set the features to 1 (presence) or 0 (absence).
        Unknown features are considered as absence.

        Returns
        -------

        """
        zero_mask = self._phon_features.isnan()
        neg_mask = self._phon_features < 0
        match non_negative_feats, keep_unknown_phon_feats:
            case True, True:
                self._phon_features[neg_mask] = 0.
            case True, False:
                self._phon_features[zero_mask | neg_mask] = 0.
            case False, True:
                pass
            case False, False:
                self._phon_features[zero_mask] = -1.

    def phnm2vec(self, phoneme: str) -> torch.Tensor:
        phnm_idx = np.nonzero(self._phonemes == phoneme)[0]

        # TODO: another silence handling techniques?
        #  silence handling so far: a 0-tensor
        if phnm_idx.size == 0:
            feats = torch.zeros_like(self._phon_features[0])
        else:  # take the *unique* phonological features for `phn`
            feats = self._phon_features[phnm_idx[0]]

        return feats


class AudioDataset(Dataset):
    def __init__(
            self,
            file_paths: list[tuple[Path, ...]],
            phon_feats_path: str,
            ema_sample_rate: int,
            ignore_phnm_cache: bool,
            non_negative_feats: bool,
            ema_not_art: bool,
            txg_not_phone: bool,
            language: str,
            remove_silence: bool,
            keep_unknown_phon_feats: bool
    ):
        self._file_paths = file_paths
        self._target_ema_sr = ema_sample_rate
        self._ignore_phnm_cache = ignore_phnm_cache
        self._ema_not_art = ema_not_art
        self._remove_silence = remove_silence

        # set articulatory-data constants
        if ema_not_art:
            self._num_art_dimensions = len(EMA_TRACES)
            self._src_ema_sr = 500
        else:
            self._num_art_dimensions = len(ART_PARAMS)
            self._src_ema_sr = 100

        match language:
            case 'uk':
                mocha_timit_inv = uk_english_inventory(broad_trans=False)
            case 'us':
                mocha_timit_inv = diph_english_inventory()
            case 'fr':
                mocha_timit_inv = french_inventory()
            case _:
                raise ValueError(f'Unknown language: {language}')

        self._phone_to_phnms = mocha_timit_inv.mapping

        if txg_not_phone:
            self.convert_to_phnm_trans = self._textgrid_to_phnm_trans
        else:
            self.convert_to_phnm_trans = self._phone_trans_to_phnm_trans

        # load phonological features
        if phon_feats_path.endswith('/panphon.npz'):
            self._feature_table = FeatureTable()
            self._feature_table.num_features = len(
                self._feature_table
                .word_to_vector_list('p', numeric=True)[0]
            )
        else:
            self._feature_table = PhonologicalFeatures(phon_feats_path)

            if language != 'us':
                self._feature_table.project_features(
                    non_negative_feats,
                    keep_unknown_phon_feats
                )

        self.generate_phnm_files()

    @property
    def file_paths(self) -> list[tuple[Path, ...]]:
        return self._file_paths

    @property
    def num_art_dimensions(self) -> int:
        return self._num_art_dimensions

    @property
    def num_features(self):
        return self._feature_table.num_features

    def _textgrid_to_phnm_trans(self, src_path: Path) -> Optional[Path]:
        dst_path = src_path.with_suffix(Extension.PHNM)
        if not self._ignore_phnm_cache and dst_path.exists():
            # print(
            #     f'Skipped txg->phnm translation of {src_path} because {dst_path} exists'
            # )
            return dst_path

        try:
            grid = textgrids.TextGrid(src_path)
        except Exception as e:
            print(f'Skipped txg->phnm translation of {src_path} because of {e}')
            return None

        fail = False

        # discard transcription files that do not begin or end with a silence.
        for idx in [0, -1]:
            if grid['phones'][idx].text != '':
                print(
                    f'Skipped txg->phnm translation of {src_path} because of '
                    f'non-silence boundary phone "{grid["phones"][idx].text}"!'
                )
                fail = True
                break

        if not fail:
            with open(dst_path, 'w') as fp:
                for phone in grid['phones']:
                    phnms = self._phone_to_phnms[phone.text]

                    # TODO: is this necessary? If yes, then why is it not enforced for
                    #  phone->phoneme conversion below?
                    # discard transcription files containing unrecognised phones
                    if len(phnms) == 1 and phnms[0] == 'spn':
                        print(
                            f'Skipped txg->phnm translation of {src_path} because of found "spn"!'
                        )
                        fail = True
                        break

                    # heuristic:
                    # diphthongs are seen as two phones exactly cut at the midpoint
                    if len(phnms) == 2:
                        x_mid = (phone.xmin + phone.xmax) / 2
                        fp.write(f'{phone.xmin} {x_mid} {phnms[0]}\n')
                        fp.write(f'{x_mid} {phone.xmax} {phnms[1]}\n')
                    else:
                        fp.write(f'{phone.xmin} {phone.xmax} {phnms[0]}\n')

        if fail:
            dst_path.unlink(missing_ok=True)
            dst_path = None

        return dst_path

    def _phone_trans_to_phnm_trans(self, src_path: Path) -> Optional[Path]:
        dst_path = src_path.with_suffix(Extension.PHNM)
        if not self._ignore_phnm_cache and dst_path.exists():
            # print(
            #     f'Skipped phone->phnm translation of {src_path} because {dst_path} exists'
            # )
            return dst_path

        clean_text = ''
        with open(src_path, 'r') as fp:
            for line in fp:
                beg, end, phone = line.split()
                phnm = self._phone_to_phnms[phone]

                clean_text += f'{int(beg) / 100} {int(end) / 100} {phnm}\n'

        with open(dst_path, 'w') as fp:
            fp.write(clean_text)

        return dst_path

    def generate_phnm_files(self) -> None:
        filtered_file_paths = []
        for paths in self._file_paths:  # ema_path, wav_path, txg_phone_path...
            phnm_path = self.convert_to_phnm_trans(paths[2])
            if phnm_path is not None:
                filtered_file_paths.append((*paths[:2], phnm_path))

        self._file_paths = filtered_file_paths

    def remove_boundary_silence(self, ema, wav, phono_feats):
        # ema freq: self._target_ema_sr
        # wav freq: SAMPLING_RATE

        def silence_boundaries(beg_time, end_time, freq):
            left_sil = math.floor(round(beg_time * freq, 2))
            right_sil = math.ceil(round(end_time * freq, 2))
            return left_sil, right_sil

        ema_left, ema_right = silence_boundaries(
            phono_feats[0, 1].item(),
            phono_feats[-1, 0].item(),
            self._target_ema_sr
        )
        wav_left, wav_right = silence_boundaries(
            phono_feats[0, 1].item(),
            phono_feats[-1, 0].item(),
            SAMPLING_RATE
        )

        ema = ema[ema_left:ema_right]
        wav = wav[wav_left:wav_right]
        phono_feats[0, 0] = phono_feats[0, 1]
        phono_feats[:, :2] = phono_feats[:, :2] - phono_feats[0, 0]
        phono_feats[-1, 1] = phono_feats[-1, 0]

        return ema, wav, phono_feats

    def read_featural_transcription(self, phnm_path: Path) -> torch.Tensor:
        phono_feats = []
        with open(phnm_path, 'r') as fp:
            for line in fp:
                beg, end, phnm = line.split()

                if isinstance(self._feature_table, PhonologicalFeatures):
                    feats = self._feature_table.phnm2vec(phnm)
                elif phnm == 'sil':
                    feats = torch.from_numpy(
                        np.zeros_like(
                            self._feature_table.word_to_vector_list('p', numeric=True)
                        )[0]
                    )
                else:
                    feats = self._feature_table.word_to_vector_list(phnm, numeric=True)
                    if len(feats) == 0:
                        print(
                            f'Warning: {phnm} ({phnm_path}) not found in panphon\'s Feature Table!'
                        )
                    else:
                        feats = torch.as_tensor(feats[0])

                phono_feats.append(
                    torch.cat(
                        [torch.as_tensor([float(beg), float(end)]), feats],
                        dim=0
                    )
                )
        return torch.stack(phono_feats, dim=0)

    def __len__(self):
        return len(self._file_paths)

    def __getitem__(self, idx):
        # fetch the corresponding paths, but do nothing with the transcription one
        ema_path, wav_path, phnm_path = self._file_paths[idx][:3]

        # fetch and process the EMA data (fourier resampling)
        ema = ArtLikeRecording(ema_path, self._ema_not_art, self._src_ema_sr) \
            .process(sample_rate=self._target_ema_sr)
        ema = torch.from_numpy(ema)

        # read the waveform
        wav = torchaudio.load(wav_path)[0].mean(dim=0)

        # fetch the phone start time, end time and phonological features
        phono_feats = self.read_featural_transcription(phnm_path)

        if self._remove_silence:
            ema, wav, phono_feats = self.remove_boundary_silence(
                ema, wav, phono_feats
            )

        return ema, wav, phono_feats

    @staticmethod
    def generate_batch(batch):
        emas, wavs, phns = zip(*batch)

        ema_lens = torch.as_tensor([len(e) for e in emas])
        wav_lens = torch.as_tensor([len(w) for w in wavs])
        phn_lens = torch.as_tensor([len(p) for p in phns])

        emas = pad_sequence(emas, batch_first=True)
        wavs = pad_sequence(wavs, batch_first=True)
        phns = pad_sequence(phns, batch_first=True)

        return emas, ema_lens, wavs, wav_lens, phns, phn_lens
