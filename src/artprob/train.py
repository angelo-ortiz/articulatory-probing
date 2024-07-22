import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from artprob.criterion import CCACriterion, LinearCriterion, MultiLinearCriterion
from artprob.dataset import AudioDataset, Extension, find_file_paths
from artprob.model import CubicHermiteSplineInterpolation, CubicSplineInterpolation, \
    FixedTimingsNaturalMinAccelInterpolation, \
    LearntFiniteLANaturalMinAccelInterpolation, \
    LearntInfiniteLAClampedMinAccelInterpolation, \
    LearntInfiniteLANaturalMinAccelInterpolation, LinearInterpolation, \
    MFCCDeltasEncoder, MFCCEncoder, MinAccelCubicHermiteSplineInterpolation, \
    MinAccelCubicSplineInterpolation, MinJerkCubicHermiteSplineInterpolation, \
    MinJerkCubicSplineInterpolation, MonotoneCubicSplineInterpolation, \
    NaturalCubicSplineInterpolation, PiecewiseConstantInterpolation, S3PRLWrapper
from artprob.utils.array_utils import EarlyStopping, array2str_list
from artprob.utils.misc import set_seed
from artprob.utils.tensor_utils import save_checkpoint

S3PRL_MODELS = [
    'wavlm-base',
    'wavlm-large',
    'wav2vec2-base',
    'wav2vec2-large',
    'hubert-base',
    'hubert-large',
]

OTH_MODELS = [
    'mfcc',
    'mfcc-deltas'
]

INTERP_METHODS = [
    'piecewise-cst',
    'linear',
    'cubic-spline',
    'mono-cubic',
    'nat-cubic-spline',
    'ma-cubic-spline',
    'mj-cubic-spline',
    'cubic',
    'ma-cubic',
    'mj-cubic',
    'nat-min-accel-fix',
    'cla-min-accel',
    'nat-min-accel',
    'nat-min-accel-fla',
]


def extend_path(path: Path, ext: str):
    return path.parent / (path.name + ext)


def s3prl_load_info(model: str, local_path: Optional[str]) -> tuple[str, ...]:
    if local_path is not None:
        return model.split('-')[0] + '_local', local_path
    match model:
        case 'wavlm-base':
            return ('wavlm_base',)
        case 'wavlm-large':
            return ('wavlm_large',)
        case 'wav2vec2-base':
            return ('wav2vec2_base_960',)
        case 'wav2vec2-large':
            return ('wav2vec2_large_ll60k',)
        case 'hubert-base':
            return ('hubert_base',)
        case 'hubert-large':
            return ('hubert_large_ll60k',)


def fetch_model_n_info(args, num_features: int):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_name = args.model

    match args.model:
        case mod if mod in S3PRL_MODELS:
            from s3prl.nn import S3PRLUpstream

            model = S3PRLUpstream(*s3prl_load_info(mod, args.model_path))
            hidden_sizes = model.hidden_sizes
            model = S3PRLWrapper(model, args.ema_sample_rate, args.resample_mode)
            model_name = f'{model_name}-{args.resample_mode}'
        case 'mfcc':
            model = MFCCEncoder(args.n_mfcc, args.ema_sample_rate, args.resample_mode)
            hidden_sizes = [model.hidden_size]
            model_name = f'{model_name}-{args.resample_mode}'
        case 'mfcc-deltas':
            model = MFCCDeltasEncoder(
                args.n_mfcc, args.ema_sample_rate, args.resample_mode, win_length=5
            )
            hidden_sizes = [model.hidden_size]
            model_name += f'-{args.resample_mode}'
        case 'piecewise-cst':
            model = PiecewiseConstantInterpolation(args.ema_sample_rate)
            hidden_sizes = [num_features]
            device = 'cpu'
        case 'linear':
            model = LinearInterpolation(args.ema_sample_rate, args.boundary_knots)
            hidden_sizes = [num_features]
            device = 'cpu'
        case 'cubic-spline':
            model = CubicSplineInterpolation(
                args.ema_sample_rate,
                args.boundary_condition,
                args.boundary_knots,
                args.boundary_free_frames
            )
            hidden_sizes = [num_features]
            device = 'cpu'
            model_name += f'-{args.boundary_condition}-{args.boundary_knots}'
        case 'mono-cubic':
            model = MonotoneCubicSplineInterpolation(
                args.ema_sample_rate,
                args.boundary_knots,
                args.boundary_free_frames
            )
            hidden_sizes = [num_features]
            device = 'cpu'
            model_name += f'-{args.boundary_knots}'
        case 'nat-cubic-spline':
            model = NaturalCubicSplineInterpolation(
                args.ema_sample_rate,
                args.boundary_knots,
                args.boundary_free_frames
            )
            hidden_sizes = [num_features]
        case 'ma-cubic-spline':
            model = MinAccelCubicSplineInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                not args.desynchronise_targets,
                args.clamp
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.weak_coeff}'
        case 'mj-cubic-spline':
            model = MinJerkCubicSplineInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                not args.desynchronise_targets,
                args.clamp
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.weak_coeff}'
        case 'cubic':
            model = CubicHermiteSplineInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats
            )
            hidden_sizes = [num_features]
        case 'ma-cubic':
            model = MinAccelCubicHermiteSplineInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                not args.desynchronise_targets
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.weak_coeff}'
        case 'mj-cubic':
            model = MinJerkCubicHermiteSplineInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                not args.desynchronise_targets
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.weak_coeff}'
        case 'cla-min-accel':
            model = LearntInfiniteLAClampedMinAccelInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                args.patience,
                args.min_delta,
                not args.desynchronise_targets,
                args.clamp
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.max_iter}'
        case 'nat-min-accel-fix':
            model = FixedTimingsNaturalMinAccelInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats
            )
            hidden_sizes = [num_features]
        case 'nat-min-accel':
            model = LearntInfiniteLANaturalMinAccelInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                args.patience,
                args.min_delta,
                not args.desynchronise_targets,
                args.clamp
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.max_iter}'
        case 'nat-min-accel-fla':
            model = LearntFiniteLANaturalMinAccelInterpolation(
                args.ema_sample_rate,
                args.remove_silence,
                args.keep_unknown_phon_feats,
                args.look_ahead,
                args.max_iter,
                args.lr,
                args.knot_lr,
                args.weak_coeff,
                args.patience,
                args.min_delta,
                not args.desynchronise_targets,
                args.clamp
            )
            hidden_sizes = [num_features]
            model_name += f'-{args.look_ahead}'
        case _:
            raise ValueError(
                f'Unrecognised speech model or interpolation method: {args.model}'
            )

    return model, model_name, hidden_sizes, device


def save_results(path: Path, speaker: str, scores: Float[np.ndarray, 'dim lay']):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f'Created directory {path}!')

    layer_scores = scores.mean(axis=0)
    best_layer = layer_scores.argmax()
    path = path / f'{speaker}_scores.json'

    score_dict = {
        'best': int(best_layer),  # cast for serialisation
    }
    for lay, lay_scr in enumerate(scores.T.astype(float)):
        score_dict[lay] = (lay_scr.mean(), lay_scr.tolist())

    with open(path, 'w') as fp:
        json.dump(score_dict, fp, indent=2)

    print(f'Saved the scores at {path}!')


def parse_args():
    parser = argparse.ArgumentParser(description='EMA evaluation: Training')

    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='Batch size for the data loader.'
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random seed.'
    )

    _data = parser.add_argument_group(
        'Data',
        description='Arguments specifying the (characteristics of) the data.'
    )
    _data.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the train set.'
    )
    _data.add_argument(
        '--ema_sample_rate',
        default=50,
        type=int,
        help='Sample rate to which the EMA traces (500 Hz) will be down-sampled.'
    )
    _data.add_argument(
        '--art_params',
        action='store_true',
        help='Whether the articulatory data comes as articulatory parameters '
             '(obtained from EMA traces) or as raw EMA traces.'
    )
    _data.add_argument(
        '--remove_silence',
        action='store_true',
        help='Whether to remove boundary silence from the articulatory and phonological'
             ' data.'
    )

    _model = parser.add_argument_group(
        'Model',
        description='Arguments defining the model to train/test.'
    )
    _model.add_argument(
        '--model',
        type=str,
        choices=S3PRL_MODELS + OTH_MODELS + INTERP_METHODS,
        required=True,
        help='Pre-trained SSL speech model or interpolation method to use.'
    )
    _model.add_argument(
        '--model_path',
        default=None,
        type=str,
        help='Path to the saved pre-trained SSL speech model.'
    )
    _model.add_argument(
        '--model_size',
        type=int,
        help='Size of the last layer in the fine-tuned HuBERT-base model.'
    )
    _model.add_argument(
        '--resample_mode',
        default='sinc',
        type=str,
        choices=['sinc', 'kaiser', 'nearest', 'linear'],
        help='Resampling mode to use on top of SSL and MFCC representations.'
    )
    _model.add_argument(
        '--n_mfcc',
        default=20,
        type=int,
        help='Number of MFC coefficients to retain. (For MFCC (-deltas) only.)'
    )

    _features = parser.add_argument_group(
        'Phonological features',
        description='Arguments used to pre-process the table of phonological features.'
    )
    _features.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to the phonological features.'
    )
    _features.add_argument(
        '--language',
        default='uk',
        type=str,
        choices=['uk', 'us', 'fr'],
        help='The language (or English dialect) of the phonological features.'
    )
    _features.add_argument(
        '--non_negative_feats',
        action='store_true',
        help='Whether the phonological features should be restricted to the '
             'non-negative orthant or not.'
    )
    _features.add_argument(
        '--keep_unknown_phon_feats',
        action='store_true',
        help='Whether to ignore the unknown value, given as NaN, for certain '
             'phonological features.'
    )
    _features.add_argument(
        '--direct_phone_trans',
        action='store_true',
        help='Whether the phonetic transcription is in "direct format" (beg end phone) '
             'or in Praat\'s TextGrid format.'
    )
    _features.add_argument(
        '--ignore_phnm_cache',
        action='store_true',
        help='Whether to ignore the cached phoneme transcription (because of a change '
             'in the phonological features.)'
    )

    interpolation = parser.add_argument_group(
        'Interpolation methods',
        description="Arguments used by the interpolation methods considered."
    )
    interpolation.add_argument(
        '--boundary_condition',
        default='natural',
        type=str,
        choices=['not-a-knot', 'periodic', 'clamped', 'natural'],
        help='Boundary condition type. (For cubic spline interpolator only.)'
    )
    interpolation.add_argument(
        '--boundary_knots',
        default='cst',
        type=str,
        choices=['mid', 'cst', 'rnd', 'mlt'],
        help='Boundary knot type: midpoint zero, constant zero or round-like zero. '
             '(For linear and cubic interpolators.)'
    )
    interpolation.add_argument(
        '--boundary_free_frames',
        default=0,
        type=int,
        help='Number of silence frames (neighouring the effective utterance) not set '
             'to be silence. (For boundary knot type `mlt` only.)'
    )
    interpolation.add_argument(
        '--look_ahead',
        default=3,
        type=int,
        help='Look-ahead for the minimum-acceleration solver. (For min-accel '
             'interpolator only.)'
    )
    interpolation.add_argument(
        '--max_iter',
        default=1000,
        type=int,
        help='Maximum number of iterations for the derivative-optimising solvers. '
             '(For min-accel and min-jerk cubic interpolators only.)'
    )
    interpolation.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help='Learning rate for the gradient descent on the timings.'
             '(For timing-optimising interpolators only.)'
    )
    interpolation.add_argument(
        '--knot_lr',
        default=1e-3,
        type=float,
        help='Learning rate for the gradient descent on the knots.'
             '(For knot-optimising interpolators only.)'
    )
    interpolation.add_argument(
        '--weak_coeff',
        default=0,
        type=float,
        help='Weight coefficient for the knot error: must be non-negative.'
             '(For knot-optimising interpolators only.)'
    )
    interpolation.add_argument(
        '--desynchronise_targets',
        action='store_true',
        help='Allow asynchronous targets across articulatory features.'
             '(For time-optimising interpolators only.)'
    )
    interpolation.add_argument(
        '--clamp',
        action='store_true',
        help='Whether to clamp the optimised timings to be within the time intervals '
             'from the phonemic transcription. '
             '(For derivative-optimising interpolators only.)'
    )

    subparsers = parser.add_subparsers(dest='criterion')

    cca = subparsers.add_parser(
        'cca',
        help='Run canonical-correlation analysis between latent representations and '
             'EMA (-based) traces.'
    )
    cca.add_argument(
        '--results_path',
        type=str,
        required=True,
        help='Path where to save the CCA results.'
    )
    cca.add_argument(
        '--scope',
        default='both',
        type=str,
        choices=['train', 'test', 'both'],
        help='Whether to run the analysis on the training, test or both data splits.'
    )

    probing = subparsers.add_parser(
        'probing',
        help='Run a probing analysis of latent representations against EMA (-based) '
             'traces.'
    )
    probing.add_argument(
        '--type',
        type=str,
        choices=['linear', 'multilinear'],
        required=True,
        help='Probing type to use.'
    )
    probing.add_argument(
        '--no_bias',
        action='store_true',
        help='Whether to add a bias to the criterion\'s  linear layers. '
             '(Linear criterion only)'
    )
    probing.add_argument(
        '--epoch',
        default=10,
        type=int,
        help='Number of epochs to run for the linear criterion.'
    )
    probing.add_argument(
        '--checkpoint_path',
        default=None,
        type=str,
        help='Path to the directory from where to fetch and where to save the '
             'linear criterion\'s checkpoint.'
    )
    probing.add_argument(
        '--save_step',
        default=2,
        type=int,
        help='Number of epochs between two consecutive model and criterion saves.'
    )

    early_stopping = probing.add_argument_group(
        'Early stopping',
        description='Arguments for early stopping the training procedure. '
                    '(For probing criterion only)'
    )
    early_stopping.add_argument(
        '--patience',
        default=5,
        type=int,
        help='Number of epochs to wait without score improvement before stopping '
             'the training procedure.'
    )
    early_stopping.add_argument(
        '--min_delta',
        default=0,
        type=float,
        help='Minimum change in the score to qualify as an improvement.'
    )
    early_stopping.add_argument(
        '--checked_value',
        default='loss',
        type=str,
        choices=['loss', 'corr'],
        help='Validation value to check.'
    )
    args = parser.parse_args()

    return args


def train_epoch(data_loader, model, criterion, optimisers, stop_optim, device):
    model.train()
    criterion.train()

    cum_losses = 0.
    total_items = 0

    for ema, ema_lens, waveform, wav_lens, phn_feats, phn_lens in data_loader:
        ema = ema.to(device)
        ema_lens = ema_lens.to(device)
        waveform = waveform.to(device)
        wav_lens = wav_lens.to(device)
        phn_feats = phn_feats.to(device)
        phn_lens = phn_lens.to(device)

        for to_stop, optim in zip(stop_optim, optimisers):
            if not to_stop:
                optim.zero_grad()

        lat_reprs = model(waveform, wav_lens, phn_feats, phn_lens)

        losses = criterion(lat_reprs, ema, ema_lens)
        cum_losses += losses.detach().cpu().numpy()
        total_items += ema.nelement()

        loss = losses.sum() / ema.nelement()
        loss.backward()

        for to_stop, optim in zip(stop_optim, optimisers):
            if not to_stop:
                optim.step()

    cum_losses /= total_items

    return cum_losses


def val_epoch(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()
    criterion.enable_ema_store()

    cum_losses = 0.
    total_items = 0

    for ema, ema_lens, waveform, wav_lens, phn_feats, phn_lens in data_loader:
        ema = ema.to(device)
        ema_lens = ema_lens.to(device)
        waveform = waveform.to(device)
        wav_lens = wav_lens.to(device)
        phn_feats = phn_feats.to(device)
        phn_lens = phn_lens.to(device)

        lat_reprs = model(waveform, wav_lens, phn_feats, phn_lens)

        with torch.no_grad():
            losses = criterion(lat_reprs, ema, ema_lens)

        cum_losses += losses.detach().cpu().numpy()
        total_items += ema.nelement()

    cum_losses /= total_items
    corrs = criterion.compute_correlations()
    criterion.disable_ema_store()

    return cum_losses, corrs


def probing(
        train_dataset,
        val_dataset,
        batch_size,
        n_epoch,
        model,
        criterion,
        optimisers,
        early_stopping,
        es_checked_value,
        checkpoint_path,
        save_step,
        device
):
    stop_optim = np.zeros(len(optimisers), dtype=bool)
    best_corr = -1.
    best_state_dict = None

    for ep in range(n_epoch):
        print(f'Starting epoch {ep}')

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.generate_batch,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.generate_batch,
            shuffle=False,
            num_workers=4
        )

        tr_losses = train_epoch(
            train_loader, model, criterion, optimisers, stop_optim, device
        )
        print(
            f'Training losses:          | {" | ".join(array2str_list(tr_losses))} |'
        )

        val_losses, corrs = val_epoch(val_loader, model, criterion, device)
        print(f'Validation losses:        | {" | ".join(array2str_list(val_losses))} |')
        for dim, dim_corrs in enumerate(corrs):
            print(
                f'Dim {dim:02d} val. correlations: | {" | ".join(array2str_list(dim_corrs))} |'
            )

        layer_corrs = corrs.mean(dim=0)
        print(
            f'Validation correlations:  | {" | ".join(array2str_list(layer_corrs))} |\n'
        )

        if early_stopping is not None:
            stop_optim = early_stopping(
                val_losses if es_checked_value == 'loss' else layer_corrs
            )
            if np.all(stop_optim):
                print(
                    'Stopped because all the optimisers hit the patience '
                    f'(set to {early_stopping.patience})'
                )
                break

        if max(layer_corrs) > best_corr:
            best_corr = max(layer_corrs)
            best_state_dict = deepcopy(criterion.state_dict())

        if ep % save_step == 0 or ep == n_epoch - 1:
            save_checkpoint(
                model.state_dict(),
                criterion.state_dict(),
                tuple(optim.state_dict() for optim in optimisers),
                best_state_dict,
                extend_path(checkpoint_path, f'_{ep}.pt')
            )

    print(f'The best correlation was {best_corr:.3f}')


def cca(
        dataset,
        batch_size,
        model,
        criterion,
        device
):
    model.eval()
    criterion.eval()
    criterion.enable_ema_store()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generate_batch,
        shuffle=False,
        num_workers=4
    )

    for ema, ema_lens, waveform, wav_lens, phn_feats, phn_lens in data_loader:
        ema = ema.to(device)
        ema_lens = ema_lens.to(device)
        waveform = waveform.to(device)
        wav_lens = wav_lens.to(device)
        phn_feats = phn_feats.to(device)
        phn_lens = phn_lens.to(device)

        with torch.no_grad():
            lat_reprs = model(waveform, wav_lens, phn_feats, phn_lens)
            criterion(lat_reprs, ema, ema_lens)

    scores = criterion.compute_correlations()
    criterion.disable_ema_store()

    for dim, dim_scores in enumerate(scores):
        print(f'Dim {dim} CCA scores): | {" | ".join(array2str_list(dim_scores))} |\n')

    return scores.cpu().numpy()


def main():
    t0 = time.perf_counter()

    args = parse_args()

    set_seed(args.seed)

    if args.art_params:
        art_ext = Extension.PARAM
    else:
        art_ext = Extension.EMA

    if args.direct_phone_trans:
        other_exts = (Extension.WAV, Extension.PHONE)
    else:
        other_exts = (Extension.WAV, Extension.TXG)

    match args.criterion:
        case 'probing':
            if args.checkpoint_path is not None:
                checkpoint_path = Path(args.checkpoint_path)
                if not checkpoint_path.is_dir():
                    checkpoint_path.mkdir()
                checkpoint_path = checkpoint_path / 'checkpoint'
                with open(extend_path(checkpoint_path, '_args.json'), 'w') as file:
                    json.dump(vars(args), file, indent=2)
                args.checkpoint_path = checkpoint_path

            train_dir = Path(args.dataset) / 'train'
            if not train_dir.exists():
                train_dir = train_dir.parent

            file_paths = find_file_paths(train_dir, art_ext, other_exts)

            random.shuffle(file_paths)

            tr_size = int(0.95 * len(file_paths))

            train_paths, val_paths = file_paths[:tr_size], file_paths[tr_size:]

            train_dataset = AudioDataset(
                train_paths,
                args.features,
                args.ema_sample_rate,
                args.ignore_phnm_cache,
                args.non_negative_feats,
                not args.art_params,
                not args.direct_phone_trans,
                args.language,
                args.remove_silence,
                args.keep_unknown_phon_feats
            )
            val_dataset = AudioDataset(
                val_paths,
                args.features,
                args.ema_sample_rate,
                args.ignore_phnm_cache,
                args.non_negative_feats,
                not args.art_params,
                not args.direct_phone_trans,
                args.language,
                args.remove_silence,
                args.keep_unknown_phon_feats
            )

            model, _, hidden_sizes, device = fetch_model_n_info(
                args,
                train_dataset.num_features
            )

            model.to(device)

            match args.type:
                case 'linear':
                    criterion = LinearCriterion(
                        hidden_sizes,
                        train_dataset.num_art_dimensions,
                        bias=not args.no_bias
                    )
                    criterion.to(device)
                    optimisers = [torch.optim.Adam(params) for params in
                                  criterion.parameters()]

                    es_score_fn = np.negative if args.checked_value == 'loss' else np.copy
                    early_stopping = EarlyStopping(
                        patience=args.patience,
                        score_fn=es_score_fn,
                        min_delta=args.min_delta
                    )
                case 'multilinear':
                    criterion = MultiLinearCriterion(
                        len(hidden_sizes),
                        hidden_sizes[0],
                        train_dataset.num_art_dimensions,
                        bias=not args.no_bias
                    )
                    criterion.to(device)
                    optimisers = [torch.optim.Adam(criterion.parameters())]

                    early_stopping = None
                case _:
                    raise ValueError(f'Unknown probing: {args.type}')

            probing(
                train_dataset,
                val_dataset,
                args.batch_size,
                args.epoch,
                model,
                criterion,
                optimisers,
                early_stopping,
                args.checked_value,
                args.checkpoint_path,
                args.save_step,
                device
            )
        case 'cca':
            args.dataset = Path(args.dataset)

            match args.scope:
                case 'train':
                    train_dir = args.dataset / 'train'
                    file_paths = find_file_paths(train_dir, art_ext, other_exts)
                case 'test':
                    test_dir = args.dataset / 'test'
                    file_paths = find_file_paths(test_dir, art_ext, other_exts)
                case 'both':
                    train_dir = args.dataset / 'train'
                    test_dir = args.dataset / 'test'
                    file_paths = find_file_paths(train_dir, art_ext, other_exts) \
                                 + find_file_paths(test_dir, art_ext, other_exts)
                case _:
                    raise ValueError(f'Unknown CCA scope: {args.scope}')

            dataset = AudioDataset(
                file_paths,
                args.features,
                args.ema_sample_rate,
                args.ignore_phnm_cache,
                args.non_negative_feats,
                not args.art_params,
                not args.direct_phone_trans,
                args.language,
                args.remove_silence,
                args.keep_unknown_phon_feats
            )

            model, model_name, hidden_sizes, device = fetch_model_n_info(
                args,
                dataset.num_features
            )

            model.to(device)

            criterion = CCACriterion(
                len(hidden_sizes),
                dataset.num_art_dimensions,
                args.seed
            )

            scores = cca(
                dataset,
                args.batch_size,
                model,
                criterion,
                device
            )

            save_results(
                Path(args.results_path),
                args.dataset.name,
                scores
            )
        case _:
            raise ValueError(f'Unknown criterion: {args.criterion}')
    print(f'Elapsed time: {time.perf_counter() - t0:.3e} seconds.')


if __name__ == '__main__':
    main()
