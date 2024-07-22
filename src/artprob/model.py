import math
from functools import cached_property
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from jaxtyping import Float
from numba import njit, prange
from s3prl.nn import S3PRLUpstream
from scipy.interpolate import CubicSpline, KroghInterpolator, PchipInterpolator
from torch import nn
from torch.nn.modules.module import T
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.wav2vec2 import components
from torchaudio.pipelines import HUBERT_BASE
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from artprob import log
from artprob.min_accel.inf_solver import inf_clamped_min_accel_traj, \
    inf_natural_min_accel_traj, inf_natural_min_accel_traj_fixed_timings
from artprob.min_accel.solver import natural_min_accel_traj
from artprob.utils.array_utils import NDArrayFloat
from artprob.utils.optimal_control import poly_interpolate
from artprob.utils.tensor_utils import batch_dot, row_norm

SAMPLING_RATE = 16_000


class Resampler(nn.Module):
    def __init__(self, orig_freq: int, new_freq: int, mode: str):
        super().__init__()
        match mode:
            case 'sinc':
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq,
                    new_freq,
                    resampling_method='sinc_interpolation',
                    lowpass_filter_width=64
                )
            case 'kaiser':
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq,
                    new_freq,
                    resampling_method='kaiser_window',
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    beta=14.769656459379492
                )
            case _:
                self.resampler = self.interpolation_fn(new_freq / orig_freq, mode)

    @staticmethod
    def interpolation_fn(scale_factor: float, mode: str) -> Callable:
        def _interpolate(wav):
            to_squeeze = False
            if wav.dim() == 2:
                wav = wav.unsqueeze(1)
                to_squeeze = True

            new_wav = F.interpolate(wav, scale_factor=scale_factor, mode=mode)
            if to_squeeze:
                return new_wav.squeeze(1)

            return new_wav

        return _interpolate

    def forward(self, wav):
        wav = wav.transpose(-1, -2).contiguous()
        wav = self.resampler(wav)
        wav = wav.transpose(-1, -2).contiguous()
        return wav


class MFCCEncoder(nn.Module):
    DOWN_SAMPLING_RATE: int = 160

    def __init__(
            self,
            n_mfcc: int,
            target_sample_rate: int,
            resample_mode: str,
            permute_last_dims: bool = True
    ):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.permute_last_dims = permute_last_dims
        melkwargs = {"n_mels": max(80, n_mfcc), "n_fft": 321}
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, melkwargs=melkwargs)
        self.resampler = Resampler(
            SAMPLING_RATE // MFCCEncoder.DOWN_SAMPLING_RATE,
            target_sample_rate,
            resample_mode
        )

    @property
    def hidden_size(self) -> int:
        return self.n_mfcc

    def forward(self, wav, wav_lens, phn, phn_lens):
        with torch.no_grad():
            wav = wav.view(wav.size(0), -1)
            enc = self.MFCC(wav)

            if self.permute_last_dims:
                enc = self.resampler(enc.contiguous().permute(0, 2, 1))

        return [enc]


class MFCCDeltasEncoder(nn.Module):
    def __init__(
            self,
            n_mfcc: int,
            target_sample_rate: int,
            resample_mode: str,
            win_length: int = 5
    ):
        super().__init__()
        self.mfcc_encoder = MFCCEncoder(
            n_mfcc, target_sample_rate, resample_mode, permute_last_dims=False
        )
        self.deltas = torchaudio.transforms.ComputeDeltas(win_length=win_length)

    @property
    def hidden_size(self) -> int:
        return 3 * self.mfcc_encoder.hidden_size

    def forward(self, wav, wav_lens, phn, phn_lens):
        with torch.no_grad():
            mfcc = self.mfcc_encoder(wav, wav_lens, phn, phn_lens)
            fst_deriv = self.deltas(mfcc[0])
            snd_deriv = self.deltas(fst_deriv)
            full_rep = torch.cat((mfcc[0], fst_deriv, snd_deriv), dim=1)
            full_rep = self.mfcc_encoder.resampler(
                full_rep.contiguous().permute(0, 2, 1)
            )

        return [full_rep]


class S3PRLWrapper(nn.Module):
    def __init__(
            self,
            model: S3PRLUpstream,
            target_sample_rate: int,
            resample_mode: str,
    ):
        super().__init__()
        self.model = model
        self.model.eval()

        ds_rates = set(model.downsample_rates)
        if len(ds_rates) > 1:
            print(
                'Warning: there is more than one down-sample rate for the given '
                f'pre-trained SSL speech model: {ds_rates}'
            )
        self.resampler = Resampler(
            SAMPLING_RATE // ds_rates.pop(),
            target_sample_rate,
            resample_mode
        )

    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        return self

    def forward(self, wav, wav_lens, phn, phn_lens):
        with torch.no_grad():
            pred, _ = self.model(wav, wav_lens)
            pred = [self.resampler(pr) for pr in pred]

        return pred


def _pad_zeroed_derivatives(
        arr: Float[np.ndarray, "n d"],
        num_der: int
) -> Float[np.ndarray, "n*(1+{num_der}) d"]:
    padded = np.pad(
        arr[:, np.newaxis],
        pad_width=((0, 0), (0, num_der), (0, 0)),
        mode='constant'
    )
    return padded.reshape(-1, arr.shape[-1])


def _pad_zeroed_derivatives_torch(
        tensor: Float[torch.Tensor, "n d"],
        num_der: int
) -> Float[torch.Tensor, "n*(1+{num_der}) d"]:
    padded = F.pad(
        tensor.unsqueeze(1),
        pad=(0, 0, 0, num_der, 0, 0),
        mode='constant'
    )
    return padded.view(-1, tensor.size(-1))


class TimingKnots:
    def __init__(
            self,
            sample_rate: int,
            boundary_derivatives: int,
            boundary_knots: str = 'cst',
            boundary_free_frames: int = 0
    ):
        self.sample_rate = sample_rate
        self.boundary_derivatives = boundary_derivatives
        self.boundary_free_frames = boundary_free_frames

        match boundary_knots:
            case 'mid':
                self._compute = self._midpoint_boundaries
            case 'cst':
                self._compute = self._constant_segment_boundaries
            case 'rnd':
                self._compute = self._rounded_boundaries
            case 'mlt':
                self._compute = self._multi_constant_segment_boundaries
            case _:
                raise ValueError(f'Unknwon boundary knot type: {boundary_knots}')

    def _midpoint_boundaries(self, data: Float[np.ndarray, "k d_p2"]):
        mid_timings = data[:, :2].mean(axis=1)
        timings = np.pad(mid_timings, pad_width=self.boundary_derivatives, mode='edge')

        mid_knots = data[:, 2:]

        # hypothesis: initial and final knots are zeros, this way zero-padding
        # derivatives before the initial knot's position is the same as adding them
        # after it
        knots = np.pad(
            mid_knots,
            pad_width=((self.boundary_derivatives,) * 2, (0,) * 2),
            mode='constant'
        )

        return timings, knots

    def _constant_segment_boundaries(self, data: Float[np.ndarray, "k d_p2"]):
        timings = np.concatenate(
            (
                data[0, :2].repeat(self.boundary_derivatives + 1),
                data[1:-1, :2].mean(axis=1),
                data[-1, :2].repeat(self.boundary_derivatives + 1)
            ), axis=0
        )

        fst_knots = _pad_zeroed_derivatives(data[0:1, 2:], self.boundary_derivatives)
        lst_knots = _pad_zeroed_derivatives(data[-1:, 2:], self.boundary_derivatives)
        knots = np.concatenate(
            (
                fst_knots,
                fst_knots,
                data[1:-1, 2:],
                lst_knots,
                lst_knots
            ), axis=0
        )

        return timings, knots

    def _multi_constant_segment_boundaries(self, data: Float[np.ndarray, "k d_p2"]):
        num_beg_sil_m1 = math.floor(
            round(data[0, 1] * self.sample_rate - self.boundary_free_frames, 2)
        )
        num_beg_sil_m1 = max(num_beg_sil_m1, 0)
        beg_sil_timings = np.linspace(
            0.,
            num_beg_sil_m1 / self.sample_rate,
            num_beg_sil_m1 + 1,
            dtype=np.float32
        )

        num_end_sil_m1 = math.floor(
            round(
                (data[-1, 1] - data[-1, 0]) * self.sample_rate
                - self.boundary_free_frames,
                2
            )
        )
        num_end_sil_m1 = max(num_end_sil_m1, 0)
        end_sil_timings = np.linspace(
            data[-1, 1] - num_end_sil_m1 / self.sample_rate,
            data[-1, 1],
            num_end_sil_m1 + 1,
            dtype=np.float32
        )

        timings = np.concatenate(
            (
                beg_sil_timings.repeat(self.boundary_derivatives + 1),
                data[1:-1, :2].mean(axis=1),
                end_sil_timings.repeat(self.boundary_derivatives + 1)
            ), axis=0
        )

        fst_knots = _pad_zeroed_derivatives(
            data[0:1, 2:].repeat(num_beg_sil_m1 + 1, axis=0),
            self.boundary_derivatives
        )
        lst_knots = _pad_zeroed_derivatives(
            data[-1:, 2:].repeat(num_end_sil_m1 + 1, axis=0),
            self.boundary_derivatives
        )

        knots = np.concatenate(
            (
                fst_knots,
                data[1:-1, 2:],
                lst_knots
            ), axis=0
        )

        return timings, knots

    def _rounded_boundaries(self, data: Float[np.ndarray, "k d_p2"]):
        timings = np.concatenate(
            (
                data[0, 1].repeat(self.boundary_derivatives + 1),
                data[1:-1, :2].mean(axis=1),
                data[-1, 0].repeat(self.boundary_derivatives + 1)
            ), axis=0
        )

        fst_knots = _pad_zeroed_derivatives(data[0:1, 2:], self.boundary_derivatives)
        lst_knots = _pad_zeroed_derivatives(data[-1:, 2:], self.boundary_derivatives)
        knots = np.concatenate(
            (
                fst_knots,
                data[1:-1, 2:],
                lst_knots
            ), axis=0
        )

        return timings, knots

    def __call__(self, data: Float[np.ndarray, "k d_p2"]):
        return self._compute(data)


class PhoneInterpolation(nn.Module):
    def __init__(self, target_sample_rate: int, frame_loc: str = 'end'):
        super().__init__()
        self._target_sample_rate = target_sample_rate

        # define the offset in the sampled timings
        match frame_loc:
            case 'beg':
                self._offset = 1 / target_sample_rate
            case 'mid':
                self._offset = 0.5 / target_sample_rate
            case 'end':
                self._offset = 0.
            case _:
                raise ValueError(f'Unknwon frame location: {frame_loc}')

    def _compute_samples_timings(self, end_time: float) -> NDArrayFloat:
        # use of `math.floor` because of hypothesis "len(wav) > len(ema)"
        # use of `round` because of the inexact floating-point product below
        num_samples = math.floor(round(end_time * self._target_sample_rate, 2))
        samples = np.linspace(
            1 / self._target_sample_rate,
            num_samples / self._target_sample_rate,
            num_samples,
            dtype=np.float32
        )
        return samples - self._offset

    def interpolate(
            self,
            seq: Float[np.ndarray, "phones dim"]
    ) -> Float[np.ndarray, "seq dim-2"]:
        raise NotImplementedError('Subclasses must override this method!')

    def forward(self, wav, wav_lens, phn, phn_lens):
        batch = []
        for seq, l in zip(phn, phn_lens):
            interp_seq = self.interpolate(seq[:l.item()].detach().cpu().numpy())
            batch.append(torch.from_numpy(interp_seq).to(phn.device))
        batch = pad_sequence(batch, batch_first=True)
        return [batch]


class PiecewiseConstantInterpolation(PhoneInterpolation):
    def __init__(self, target_sample_rate: int):
        super().__init__(target_sample_rate)

    def interpolate(
            self,
            seq: Float[np.ndarray, "phones dim"]
    ) -> Float[np.ndarray, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])
        indices = np.searchsorted(seq[:-1, 1], samples, side='left')
        interp_seq = seq[indices, 2:]
        return interp_seq


@njit(parallel=True)
def interp1d(
        samples: Float[np.ndarray, "m"],
        timings: Float[np.ndarray, "n"],
        knots: Float[np.ndarray, "n d"]
) -> Float[np.ndarray, "m d"]:
    interp_seq = np.empty((len(samples), knots.shape[1]), dtype=knots.dtype)

    # interpolate each dimension separately after NaN-value masking
    for d in prange(knots.shape[1]):
        non_nan_mask = ~np.isnan(knots[:, d])
        interp_seq[:, d] = np.interp(
            samples,
            timings[non_nan_mask],
            knots[non_nan_mask, d]
        )

    return interp_seq


class LinearInterpolation(PhoneInterpolation):
    def __init__(self, target_sample_rate: int, boundary_knots: str):
        super().__init__(target_sample_rate)
        self._timing_knots = TimingKnots(
            target_sample_rate,
            boundary_derivatives=0,
            boundary_knots=boundary_knots
        )

    def interpolate(
            self,
            seq: Float[np.ndarray, "phones dim"]
    ) -> Float[np.ndarray, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])
        timings, knots = self._timing_knots(seq)
        interp_seq = interp1d(
            samples,
            timings,
            knots
        )

        return interp_seq


class CubicSplineInterpolation(PhoneInterpolation):
    def __init__(
            self,
            target_sample_rate: int,
            boundary_condition: str,
            boundary_knots: str,
            boundary_free_frames: int
    ):
        super().__init__(target_sample_rate)
        self._bc_type = boundary_condition
        self._timing_knots = TimingKnots(
            target_sample_rate, 0, boundary_knots, boundary_free_frames
        )

    def interpolate(
            self,
            seq: Float[np.ndarray, "phones dim"]
    ) -> Float[np.ndarray, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])
        timings, knots = self._timing_knots(seq)
        cubic_spline = CubicSpline(timings, knots, bc_type=self._bc_type)
        interp_seq = cubic_spline(samples)
        return interp_seq.astype(np.float32)


class MonotoneCubicSplineInterpolation(PhoneInterpolation):
    """In the binary case, it is equivalent to a Cubic Hermite Spline with zero velocity
     at all the knots (including the boundary ones).
    See:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

    """

    def __init__(
            self,
            target_sample_rate: int,
            boundary_knots: str,
            boundary_free_frames: int = 0
    ):
        super().__init__(target_sample_rate)
        self._timing_knots = TimingKnots(
            target_sample_rate, 0, boundary_knots, boundary_free_frames
        )

    def interpolate(
            self,
            seq: Float[np.ndarray, "phones dim"]
    ) -> Float[np.ndarray, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])
        timings, knots = self._timing_knots(seq)
        cubic_spline = PchipInterpolator(timings, knots, extrapolate=True)
        interp_seq = cubic_spline(samples)
        return interp_seq.astype(np.float32)


class TorchTimingKnots:
    def __init__(
            self,
            sample_rate: int,
            boundary_knots: str = 'cst',
            boundary_free_frames: int = 0
    ):
        self.sample_rate = sample_rate
        self.boundary_free_frames = boundary_free_frames

        match boundary_knots:
            case 'mid':
                self._compute = self._midpoint_boundaries
            case 'cst':
                self._compute = self._constant_segment_boundaries
            case 'rnd':
                self._compute = self._rounded_boundaries
            case 'mlt':
                self._compute = self._multi_constant_segment_boundaries
            case _:
                raise ValueError(f'Unknwon boundary knot type: {boundary_knots}')

    @staticmethod
    def _midpoint_boundaries(data: Float[torch.Tensor, "k d_p2"]):
        timings = data[:, :2].mean(dim=1)
        knots = data[:, 2:]

        return timings, knots

    @staticmethod
    def _constant_segment_boundaries(data: Float[torch.Tensor, "k d_p2"]):
        timings = torch.cat(
            (
                data[0, :2],
                data[1:-1, :2].mean(dim=1),
                data[-1, :2]
            ), dim=0
        )
        knots = F.pad(data[:, 2:].unsqueeze(0), (0, 0, 1, 1), mode='replicate')[0]

        return timings, knots

    def _multi_constant_segment_boundaries(self, data: Float[torch.Tensor, "k d_p2"]):
        num_beg_sil_m1 = math.floor(
            round(data[0, 1].item() * self.sample_rate - self.boundary_free_frames, 2)
        )
        num_beg_sil_m1 = max(num_beg_sil_m1, 0)
        beg_sil_timings = torch.linspace(
            0.,
            num_beg_sil_m1 / self.sample_rate,
            num_beg_sil_m1 + 1,
            dtype=torch.float32,
            device=data.device
        )

        num_end_sil_m1 = math.floor(
            round(
                (data[-1, 1] - data[-1, 0]).item() * self.sample_rate
                - self.boundary_free_frames,
                2
            )
        )
        num_end_sil_m1 = max(num_end_sil_m1, 0)
        end_sil_timings = torch.linspace(
            data[-1, 1] - num_end_sil_m1 / self.sample_rate,
            data[-1, 1],
            num_end_sil_m1 + 1,
            dtype=torch.float32,
            device=data.device
        )

        timings = torch.cat(
            (
                beg_sil_timings,
                data[1:-1, :2].mean(dim=1),
                end_sil_timings
            ), dim=0
        )

        knots = torch.cat(
            (
                data[0:1, 2:].repeat_interleave(num_beg_sil_m1 + 1, dim=0),
                data[1:-1, 2:],
                data[-1:, 2:].repeat_interleave(num_end_sil_m1 + 1, dim=0)
            ), dim=0
        )

        return timings, knots

    @staticmethod
    def _rounded_boundaries(data: Float[torch.Tensor, "k d_p2"]):
        timings = torch.cat(
            (
                data[0, 1:2],
                data[1:-1, :2].mean(dim=1),
                data[-1, 0:1]
            ), dim=0
        )
        knots = data[:, 2:]

        return timings, knots

    def __call__(self, data: Float[torch.Tensor, "k d_p2"]):
        return self._compute(data)


class TensorPhoneInterpolation(PhoneInterpolation):
    """Interpolation method for tensor inputs

    Parameters
    ----------
    target_sample_rate : int
    underspecified_targets : bool
        Whether the targets can be underspecified. Set to False if no target
        phonological feature is allowed to be unknown.
    boundary_knots : str
    boundary_free_frames : int, default=0
    target_sync : bool, default=True
        Whether the target timings are the same across dimensions. Set to False if the
        dimension-wise target timings are allowed to be asynchronous or if the
        interpolation method does not handle multi-dimension target unspecification.
    """

    def __init__(
            self,
            target_sample_rate: int,
            underspecified_targets: bool,
            boundary_knots: str,
            boundary_free_frames: int = 0,
            target_sync: bool = True,
    ):
        super().__init__(target_sample_rate)
        self._timing_knots = TorchTimingKnots(
            target_sample_rate,
            boundary_knots,
            boundary_free_frames
        )

        self.interpolate = self._select_interpolate_mode(
            underspecified_targets,
            target_sync
        )

    def _compute_samples_timings(
            self,
            end_time: Float[torch.Tensor, ""]
    ) -> torch.Tensor:
        # use of `math.floor` because of hypothesis "len(wav) > len(ema)"
        # use of `round` because of the inexact floating-point product below
        num_samples = math.floor(round(end_time.item() * self._target_sample_rate, 2))
        samples = torch.linspace(
            1 / self._target_sample_rate,
            num_samples / self._target_sample_rate,
            num_samples,
            dtype=torch.float32,
            device=end_time.device
        )
        return samples - self._offset

    def _select_interpolate_mode(
            self,
            underspecified_targets: bool,
            target_sync: bool
    ) -> Callable:
        match underspecified_targets, target_sync:
            case True, True:
                return self._interpolate_underspecified_sync
            case True, False:
                return self._interpolate_underspecified_nosync
            case False, True:
                return self._interpolate_specified_sync
            case False, False:
                return self._interpolate_specified_nosync

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not handle synchronisation of '
            'fully-specified targets yet!'
        )

    def _interpolate_specified_nosync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        interp_dim = []
        for i in range(2, seq.size(1)):
            interp_dim.append(
                self._interpolate_specified_sync(seq[:, [0, 1, i]])
            )

        interp_seq = torch.cat(interp_dim, dim=1)

        return interp_seq

    def _interpolate_underspecified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not handle synchronisation of '
            'underspecified targets yet!'
        )

    def _interpolate_underspecified_nosync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        interp_dim = []
        for i in range(2, seq.size(1)):
            non_nan_mask = ~torch.isnan(seq[:, i])
            interp_dim.append(
                self._interpolate_specified_sync(seq[non_nan_mask][:, [0, 1, i]])
            )

        interp_seq = torch.cat(interp_dim, dim=1)

        return interp_seq

    def forward(self, wav, wav_lens, phn, phn_lens):
        batch = []
        for seq, l in zip(phn, phn_lens):
            batch.append(self.interpolate(seq[:l.item()]))

        batch = pad_sequence(batch, batch_first=True)

        return [batch]


##############################
# Natural Cubic Spline
##############################

class NaturalCubicSplineInterpolation(TensorPhoneInterpolation):
    """Natural cubic spline in PyTorch.
    Supports underspecified targets. Targets are always synchronised across dimensions.
    """

    def __init__(
            self,
            target_sample_rate: int,
            boundary_knots: str,
            boundary_free_frames: int = 0
    ):
        super().__init__(
            target_sample_rate,
            underspecified_targets=True,
            boundary_knots=boundary_knots,
            boundary_free_frames=boundary_free_frames,
            target_sync=True
        )

    def _interpolate_underspecified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        timings, knots = self._timing_knots(seq)

        coeffs = natural_cubic_spline_coeffs(timings, knots)
        spline = NaturalCubicSpline(coeffs)

        interp_seq = spline.evaluate(samples)

        return interp_seq


class OptimisedCubicSplineInterpolation(TensorPhoneInterpolation):
    """Natural cubic spline whose timinigs minimise the norm of a given derivative."""

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float,
            weak_coeff: float,
            target_sync: bool,
            clamp: bool = False
    ):
        super().__init__(
            target_sample_rate,
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=False if underspecified_targets else target_sync
        )
        if underspecified_targets and target_sync:
            print(
                f'Warning: {self.__class__.__name__} does not support synchronisation '
                'of underspecified targets: the underspecified targets will not be '
                'synchronised.'
            )

        self._no_boundary_silence = no_boundary_silence
        self._max_iter = max_iter
        self._lr = lr
        self._clamp = clamp
        self._weak_coeff = weak_coeff
        if weak_coeff > 0:
            self._knot_lr = knot_lr
            self._mse = nn.MSELoss(reduction='mean')

    @classmethod
    def compute_loss(
            cls,
            coeffs: torch.Tensor,
            timing_deltas: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError('Subclasses must override this method!')

    def optimise_trajectory(
            self,
            timings: torch.Tensor,
            knots: torch.Tensor,
            bounds: Optional[torch.Tensor]
    ):
        t_0, t_f = timings[[0, -1]]
        timings = timings.detach().clone().requires_grad_(True)
        timing_optim = torch.optim.Adam([timings], lr=self._lr)

        if self._weak_coeff > 0:
            weak_knots = knots.detach().clone().requires_grad_(True)
            knot_optim = torch.optim.Adam([weak_knots], lr=self._knot_lr)
        else:
            weak_knots = knots

        for it in range(self._max_iter):
            timing_optim.zero_grad()
            if self._weak_coeff > 0:
                knot_optim.zero_grad()

            coeffs = natural_cubic_spline_coeffs(timings, weak_knots)
            loss = self.compute_loss(coeffs, torch.diff(timings, n=1, dim=0)) \
                   + (self._weak_coeff * self._mse(weak_knots, knots)
                      if self._weak_coeff > 0 else 0)

            loss.backward()
            timing_optim.step()
            if self._weak_coeff > 0:
                knot_optim.step()

            # the initial and last timings are fixed
            timings.data[0] = t_0
            timings.data[-1] = t_f

            if bounds is not None:
                timings.data[1:-1] = timings.data[1:-1].clamp(
                    min=bounds[:, 0],
                    max=bounds[:, 1]
                )

        if bounds is not None:
            coeffs = natural_cubic_spline_coeffs(timings, weak_knots)

        return [_coeff.detach() for _coeff in coeffs]

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        timings, knots = self._timing_knots(seq)

        if self._clamp:
            bounds = seq[1:-1, :2] if self._no_boundary_silence else seq[:, :2]
        else:
            bounds = None

        coeffs = self.optimise_trajectory(
            timings,
            knots,
            bounds
        )

        spline = NaturalCubicSpline(coeffs)

        interp_seq = spline.evaluate(samples)

        if not self._no_boundary_silence:
            t0, tf = coeffs[0][[1, -2]]
            zero_mask = (samples <= t0) | (samples >= tf)
            interp_seq[zero_mask] = 0

        return interp_seq


class MinAccelCubicSplineInterpolation(OptimisedCubicSplineInterpolation):
    """Natural cubic spline with timing optimisation that minimises the acceleration."""

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            target_sync: bool = True,
            clamp: bool = False
    ):
        super().__init__(
            target_sample_rate,
            no_boundary_silence=no_boundary_silence,
            underspecified_targets=underspecified_targets,
            max_iter=max_iter,
            lr=lr,
            knot_lr=knot_lr,
            weak_coeff=weak_coeff,
            target_sync=target_sync,
            clamp=clamp
        )

    @classmethod
    def compute_loss(
            cls,
            coeffs: torch.Tensor,
            timing_deltas: torch.Tensor
    ) -> torch.Tensor:
        loss = 3 * row_norm(coeffs[-1], squared=True) @ timing_deltas**3 \
               + 3 * batch_dot(coeffs[-1], coeffs[-2]) @ timing_deltas**2 \
               + row_norm(coeffs[-2], squared=True) @ timing_deltas
        return loss


class MinJerkCubicSplineInterpolation(OptimisedCubicSplineInterpolation):
    """Natural cubic spline with timing optimisation that minimises the jerk."""

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            target_sync: bool = True,
            clamp: bool = False
    ):
        super().__init__(
            target_sample_rate,
            no_boundary_silence=no_boundary_silence,
            underspecified_targets=underspecified_targets,
            max_iter=max_iter,
            lr=lr,
            knot_lr=knot_lr,
            weak_coeff=weak_coeff,
            target_sync=target_sync,
            clamp=clamp
        )

    @classmethod
    def compute_loss(
            cls,
            coeffs: torch.Tensor,
            timing_deltas: torch.Tensor
    ) -> torch.Tensor:
        loss = row_norm(coeffs[-1], squared=True) @ timing_deltas
        return loss


##############################
# Cubic Hermite Spline
##############################

class CubicHermiteSplineInterpolation(TensorPhoneInterpolation):
    """Cubic Hermite Spline with zero-valued knot velocities."""

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool
    ):
        super().__init__(
            target_sample_rate,
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=True
        )

    @staticmethod
    def evaluate(samples, timings, knots, timing_deltas, knot_deltas):
        indices = torch.bucketize(samples, timings, right=True) - 1
        indices = torch.clamp(indices, 0, timing_deltas.size(0) - 1)
        delta_t = (samples - timings[indices]).unsqueeze(1)
        delta_knot = knot_deltas[indices]
        interp_seq = knots[indices] \
                     + (3 * timing_deltas[indices] - 2 * delta_t) \
                     * delta_knot * delta_t**2 \
                     / timing_deltas[indices]**3

        return interp_seq

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        timings, knots = self._timing_knots(seq)
        timing_deltas = torch.diff(timings, n=1, dim=0).unsqueeze(1)
        knot_deltas = torch.diff(knots, n=1, dim=0)

        interp_seq = self.evaluate(samples, timings, knots, timing_deltas, knot_deltas)

        return interp_seq

    def _interpolate_underspecified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        """This interpolation method does not change the timings across dimensions, so
        the (a-) synchronous character of the underspecified interpolation does not
         matter.

        Parameters
        ----------
        seq

        Returns
        -------

        """
        return self._interpolate_underspecified_nosync(seq)


class OptimisedCubicHermiteSplineInterpolation(TensorPhoneInterpolation):
    """Cubic Hermite Spline with zero-valued knot velocities whose timings minimise the
    norm of a given derivative."""

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float,
            weak_coeff: float,
            target_sync: bool = True
    ):
        super().__init__(
            target_sample_rate,
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=False if underspecified_targets else target_sync
        )
        if underspecified_targets and target_sync:
            print(
                f'Warning: {self.__class__.__name__} does not support synchronisation '
                'of underspecified targets: the underspecified targets will not be '
                'synchronised.'
            )

        self._max_iter = max_iter
        self._lr = lr
        self._weak_coeff = weak_coeff
        if weak_coeff > 0:
            self._knot_lr = knot_lr
            self._mse = nn.MSELoss(reduction='mean')

    @classmethod
    def compute_loss(
            cls,
            knot_deltas_sq: torch.Tensor,
            timing_deltas: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError('Subclasses must override this method!')

    def optimise_trajectory(
            self,
            timing_deltas: torch.Tensor,
            knots: torch.Tensor,
            bounds: Optional[torch.Tensor] = None
    ):
        timing_deltas = timing_deltas.clone().requires_grad_(True)
        timing_optim = torch.optim.Adam([timing_deltas], lr=self._lr)

        if self._weak_coeff > 0:
            weak_knots = knots.detach().clone().requires_grad_(True)
            knot_optim = torch.optim.Adam([weak_knots], lr=self._knot_lr)
        else:
            weak_knots = knots

        for _ in range(self._max_iter):
            timing_optim.zero_grad()
            if self._weak_coeff > 0:
                knot_optim.zero_grad()

            knot_deltas = torch.diff(weak_knots, n=1, dim=0)
            knot_deltas_sq = row_norm(knot_deltas, squared=True)
            loss = self.compute_loss(knot_deltas_sq, timing_deltas) \
                   + (self._weak_coeff * self._mse(weak_knots, knots)
                      if self._weak_coeff > 0 else 0)

            loss.backward()
            timing_optim.step()
            if self._weak_coeff > 0:
                knot_optim.step()

        return timing_deltas.detach(), knot_deltas.detach()

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        timings, knots = self._timing_knots(seq)
        timing_deltas = torch.diff(timings, n=1, dim=0)

        timing_deltas, knot_deltas = self.optimise_trajectory(
            timing_deltas,
            knots,
            bounds=None
        )

        timings = torch.cumsum(
            torch.cat([torch.zeros(1, device=seq.device), timing_deltas], dim=0),
            dim=0
        )
        timing_deltas = timing_deltas.unsqueeze(1)

        interp_seq = CubicHermiteSplineInterpolation.evaluate(
            samples, timings, knots, timing_deltas, knot_deltas
        )

        return interp_seq


class MinAccelCubicHermiteSplineInterpolation(OptimisedCubicHermiteSplineInterpolation):
    """Cubic Hermite Spline with zero-valued knot velocities whose timings minimise the
    acceleration.

    """

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            target_sync: bool = True
    ):
        super().__init__(
            target_sample_rate,
            no_boundary_silence=no_boundary_silence,
            underspecified_targets=underspecified_targets,
            max_iter=max_iter,
            lr=lr,
            knot_lr=knot_lr,
            weak_coeff=weak_coeff,
            target_sync=target_sync
        )

    @classmethod
    def compute_loss(
            cls,
            knot_deltas_sq: torch.Tensor,
            timing_deltas: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(knot_deltas_sq / timing_deltas**3)


class MinJerkCubicHermiteSplineInterpolation(OptimisedCubicHermiteSplineInterpolation):
    """Cubic Hermite Spline with zero-valued knot velocities whose timings minimise the
    jerk.

    """

    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            target_sync: bool = True
    ):
        super().__init__(
            target_sample_rate,
            no_boundary_silence=no_boundary_silence,
            underspecified_targets=underspecified_targets,
            max_iter=max_iter,
            lr=lr,
            knot_lr=knot_lr,
            weak_coeff=weak_coeff,
            target_sync=target_sync
        )

    @classmethod
    def compute_loss(
            cls,
            knot_deltas_sq: torch.Tensor,
            timing_deltas: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(knot_deltas_sq / timing_deltas**5)


##############################
# Minimum acceleration
##############################

class FixedTimingsNaturalMinAccelInterpolation(TensorPhoneInterpolation):
    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            loglevel: log.Level = log.Level.WARNING
    ):
        super().__init__(
            target_sample_rate,
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=True
        )
        self._no_boundary_silence = no_boundary_silence
        self._loglevel = loglevel

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        _timings, knots = self._timing_knots(seq)

        if self._no_boundary_silence:
            t0 = _timings[0]
            T = _timings[-1] - t0
            knots = knots[1:-1]
            timings = _timings[1:-1] - t0
        else:
            t0 = _timings[1]
            T = _timings[-2] - t0
            knots = knots[2:-2]
            timings = _timings[2:-2] - t0

        with torch.no_grad():
            all_knots, _ = inf_natural_min_accel_traj_fixed_timings(
                knots,
                timings,
                T=T,
                initial_pos=None,
                loglevel=self._loglevel
            )

        timings = F.pad(timings, pad=(1, 0))
        samples -= t0

        interp_seq = torch.zeros(
            samples.size(0), all_knots.size(-1), device=all_knots.device
        )

        mask = (0 < samples) & (samples < T)

        interp_seq[mask] = poly_interpolate(timings, all_knots, samples[mask])

        return interp_seq

    def _interpolate_underspecified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        """This interpolation method does not change the timings across dimensions, so
        the (a-) synchronous character of the underspecified interpolation does not
         matter.

        Parameters
        ----------
        seq

        Returns
        -------

        """
        return self._interpolate_underspecified_nosync(seq)


class LearntInfiniteLAClampedMinAccelInterpolation(TensorPhoneInterpolation):
    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            patience: int = 3,
            min_delta: float = 0,
            target_sync: bool = True,
            clamp: bool = False
    ):
        super().__init__(
            target_sample_rate,

            # TODO: add support for underspecification
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=target_sync  # TODO: add support for target desynchronisation
        )
        self._no_boundary_silence = no_boundary_silence
        self._max_iter = max_iter
        self._lr = lr
        self._knot_lr = knot_lr
        self._weak_coeff = weak_coeff
        self._patience = patience
        self._min_delta = min_delta
        self._clamp = clamp

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        _timings, knots = self._timing_knots(seq)

        if self._no_boundary_silence:
            t0 = _timings[0]
            T = _timings[-1] - t0
            knots = knots[1:-1]
        else:
            t0 = _timings[1]
            T = _timings[-2] - t0
            knots = knots[2:-2]

        all_knots, timings, _, _ = inf_clamped_min_accel_traj(
            knots,
            T,
            initial_coords=None,
            max_iter=self._max_iter,
            lr=self._lr,
            knot_lr=self._knot_lr,
            weak_coeff=self._weak_coeff,
            patience=self._patience,
            min_delta=self._min_delta,
            history=False
        )

        timings = F.pad(timings, pad=(1, 0))
        samples -= t0

        interp_seq = torch.zeros(
            samples.size(0), all_knots.size(-1), device=all_knots.device
        )

        interpolation_mask = (0 < samples) & (samples < T)

        interp_seq[interpolation_mask] = poly_interpolate(
            timings.detach(),
            all_knots.detach(),
            samples[interpolation_mask]
        )

        return interp_seq


class LearntInfiniteLANaturalMinAccelInterpolation(TensorPhoneInterpolation):
    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            patience: int = 3,
            min_delta: float = 0,
            target_sync: bool = True,
            clamp: bool = False
    ):
        super().__init__(
            target_sample_rate,

            # TODO: add support for underspecification
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=target_sync  # TODO: add support for target desynchronisation
        )
        self._no_boundary_silence = no_boundary_silence
        self._max_iter = max_iter
        self._lr = lr
        self._knot_lr = knot_lr
        self._weak_coeff = weak_coeff
        self._patience = patience
        self._min_delta = min_delta
        self._clamp = clamp

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        _timings, knots = self._timing_knots(seq)

        if self._no_boundary_silence:
            t0 = _timings[0]
            T = _timings[-1] - t0
            knots = knots[1:-1]
        else:
            t0 = _timings[1]
            T = _timings[-2] - t0
            knots = knots[2:-2]

        all_knots, timings, _, _ = inf_natural_min_accel_traj(
            knots,
            T,
            initial_pos=None,
            max_iter=self._max_iter,
            lr=self._lr,
            knot_lr=self._knot_lr,
            weak_coeff=self._weak_coeff,
            patience=self._patience,
            min_delta=self._min_delta,
            history=False
        )

        timings = F.pad(timings, pad=(1, 0))
        samples -= t0

        interp_seq = torch.zeros(
            samples.size(0), all_knots.size(-1), device=all_knots.device
        )

        interpolation_mask = (0 < samples) & (samples < T)

        interp_seq[interpolation_mask] = poly_interpolate(
            timings.detach(),
            all_knots.detach(),
            samples[interpolation_mask]
        )

        return interp_seq


class LearntFiniteLANaturalMinAccelInterpolation(TensorPhoneInterpolation):
    def __init__(
            self,
            target_sample_rate: int,
            no_boundary_silence: bool,
            underspecified_targets: bool,
            look_ahead: int,
            max_iter: int,
            lr: float,
            knot_lr: float = 1e-3,
            weak_coeff: float = 0,
            patience: int = 3,
            min_delta: float = 0,
            target_sync: bool = True,
            clamp: bool = False  # unused
    ):
        super().__init__(
            target_sample_rate,

            # TODO: add support for underspecification
            underspecified_targets=underspecified_targets,
            boundary_knots='mid' if no_boundary_silence else 'cst',
            target_sync=target_sync  # TODO: add support for target desynchronisation
        )
        self._no_boundary_silence = no_boundary_silence
        self._look_ahead = look_ahead
        self._max_iter = max_iter
        self._lr = lr
        self._knot_lr = knot_lr
        self._weak_coeff = weak_coeff
        self._patience = patience
        self._min_delta = min_delta
        self._clamp = False

    def _interpolate_specified_sync(
            self,
            seq: Float[torch.Tensor, "phones dim"]
    ) -> Float[torch.Tensor, "seq dim-2"]:
        samples = self._compute_samples_timings(seq[-1, 1])

        _timings, knots = self._timing_knots(seq)

        if self._no_boundary_silence:
            t0 = _timings[0]
            T = _timings[-1] - t0
            knots = knots[1:-1]
        else:
            t0 = _timings[1]
            T = _timings[-2] - t0
            knots = knots[2:-2]

        all_knots, timings = natural_min_accel_traj(
            knots,
            T,
            look_ahead=self._look_ahead,
            initial_pos=None,
            max_iter=self._max_iter,
            lr=self._lr,
            knot_lr=self._knot_lr,
            weak_coeff=self._weak_coeff,
            patience=self._patience,
            min_delta=self._min_delta,
            history=False
        )

        samples -= t0

        interp_seq = torch.zeros(
            samples.size(0), all_knots.size(-1), device=all_knots.device
        )

        interpolation_mask = (0 < samples) & (samples < T)

        interp_seq[interpolation_mask] = poly_interpolate(
            timings.detach(),
            all_knots.detach(),
            samples[interpolation_mask]
        )

        return interp_seq
