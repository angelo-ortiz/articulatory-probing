import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from artprob.criterion import LinearCriterion, MultiLinearCriterion
from artprob.dataset import AudioDataset, Extension, find_file_paths
from artprob.train import fetch_model_n_info, save_results, val_epoch
from artprob.utils.array_utils import array2str_list


def parse_args():
    parser = argparse.ArgumentParser(description='EMA evaluation: Test')

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='The path to the dataset on which to evaluate.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The model\'s name.'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        required=True,
        help='Path where to save the correlation results.'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='The path to the directory from where to fetch the model and criterion '
             'checkpoint.'
    )

    args = parser.parse_args()

    return args


def test_epoch(args, data_loader, model, criterion, device):
    test_losses, corrs = val_epoch(data_loader, model, criterion, device)

    print(f'Test losses:              | {" | ".join(array2str_list(test_losses))} |')
    for dim, dim_corrs in enumerate(corrs):
        print(
            f'Dim {dim:02d} test correlations: | {" | ".join(array2str_list(dim_corrs))} |'
        )

    layer_corrs = corrs.mean(dim=0)
    print(f'Test correlations:        | {" | ".join(array2str_list(layer_corrs))} |')

    return args.dataset.name, corrs.cpu().numpy()


def main():
    args = parse_args()

    ckpt_root = Path(args.checkpoint_path)
    setattr(args, 'model_variant', ckpt_root.name)

    # ensure the checkpoint file with the training arguments exists
    if not ckpt_root.is_dir():
        raise ValueError(
            f'The given checkpoint path "{ckpt_root}" does not exist or is not a '
            'directory!'
        )

    param_file = ckpt_root / 'checkpoint_args.json'
    if not param_file.exists():
        raise ValueError(
            f'The given checkpoint path "{ckpt_root}" does not contain a parameter '
            f'file "checkpoint_args.json"!'
        )

    # load the training arguments
    with open(param_file, 'r') as fp:
        params = json.load(fp)

        for k, v in params.items():
            if k in ['dataset', 'model', 'checkpoint_path']:
                continue

            setattr(args, k, v)

    args.dataset = Path(args.dataset)
    test_dir = args.dataset / 'test'

    if args.art_params:
        art_ext = Extension.PARAM
    else:
        art_ext = Extension.EMA

    if args.direct_phone_trans:
        other_exts = (Extension.WAV, Extension.PHONE)
    else:
        other_exts = (Extension.WAV, Extension.TXG)

    test_paths = find_file_paths(test_dir, art_ext, other_exts)

    # create the audio dataset and dataloader
    test_dataset = AudioDataset(
        test_paths,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_dataset.generate_batch,
        shuffle=False,
        num_workers=4
    )

    # fetch the model
    model, model_name, hidden_sizes, device = fetch_model_n_info(
        args,
        test_dataset.num_features
    )

    # fetch the criterion
    match args.type:
        case 'linear':
            criterion = LinearCriterion(
                hidden_sizes,
                test_dataset.num_art_dimensions,
                bias=not args.no_bias
            )
        case 'multi':
            criterion = MultiLinearCriterion(
                len(hidden_sizes),
                hidden_sizes[0],
                test_dataset.num_art_dimensions,
                bias=not args.no_bias
            )
        case _:
            raise ValueError(f'Unknown probing: {args.type}')

    # find the latest checkpoint
    checkpoints = list(ckpt_root.glob('*.pt'))
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
    setattr(args, 'ckpt_path', checkpoints[-1])

    # fetch the state diction ary from the latest checkpoint
    state_dict = torch.load(checkpoints[-1], 'cpu')

    # load the saved state dictionary into the model
    model.load_state_dict(state_dict['latest'])
    model.to(device)

    # load the saved state dictionary into the criterion
    criterion.load_state_dict(state_dict['best'])
    criterion.to(device)

    results_args = test_epoch(args, test_loader, model, criterion, device)

    save_results(Path(args.results_path), *results_args)


if __name__ == '__main__':
    main()
