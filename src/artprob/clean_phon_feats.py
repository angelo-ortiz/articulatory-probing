import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IPA_MODES = ['feat', 'phnm', 'mix']
ART_PHON_MODES = ['scalar', 'one-hot', 'group-scalar', 'scalar-mix', 'one-hot-mix']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch phonological features from a table'
    )
    parser.add_argument(
        '--table_path',
        type=str,
        required=True,
        help='The path to the table with the phonological features.'
    )
    parser.add_argument(
        '--save_path',
        default=None,
        type=str,
        help='The path to the directory where the phoneme array and feature matrix '
             'will be saved.'
    )
    subparsers = parser.add_subparsers(dest='features')

    ipa = subparsers.add_parser('ipa', help='IPA-based features')
    ipa.add_argument(
        '--filter_non_distinctive',
        action='store_true',
        help='Filter out the non-distinctive features.'
    )
    ipa.add_argument(
        '--value_mode',
        default='feat',
        type=str,
        choices=IPA_MODES,
        help='Whether the IPA features are kept as is, discarded (one-hot phonemes '
             'instead) or mix (both IPA and one-hot phonemes).'
    )

    art_phono = subparsers.add_parser(
        'art_phono',
        help='Articulatory-phonology (AP) features'
    )
    art_phono.add_argument(
        '--keep_velum',
        action='store_true',
        help='Keep the velum feature.'
    )
    art_phono.add_argument(
        '--value_mode',
        default='scalar',
        type=str,
        choices=ART_PHON_MODES,
        help='Whether the categorial AP features are attributed a scalar value, one-hot'
             ' encoded or attributed a real value after articulator grouping.'
    )

    args = parser.parse_args()

    input_file = Path(args.table_path)

    if args.save_path is None:
        output_dir = input_file.parent
    else:
        output_dir = Path(args.save_path)

    args.table_path = input_file
    args.save_path = output_dir / input_file.with_suffix('.npz').name

    return args


def _ipa_df2phnm(phonemes):
    phonemes = np.pad(phonemes, (0, 1), constant_values='sil')
    header = phonemes
    features = np.eye(len(phonemes), dtype=np.float32)
    num_all_feats = len(phonemes)
    return phonemes, header, features, num_all_feats


def _ipa_df2feat(data, filter_non_distinctive):
    header = data.columns[1:].to_numpy()
    features = data.iloc[:, 1:].to_numpy()
    features = np.piecewise(
        features,
        [features == '+', features == '-', features == '0'],
        [1, -1, float('nan')]
    )
    features = np.asarray(features, dtype=np.float32)
    num_all_feats = features.shape[1]
    if filter_non_distinctive:
        # absence of zero values does not affect distinctiveness
        mask = np.apply_along_axis(
            lambda arr: 1 in arr and -1 in arr, 0, features
        )
        header = header[mask]
        features = features[:, mask]
    return header, features, num_all_feats


def process_ipa(xlsx_path: Path, filter_non_distinctive: bool, value_mode: bool):
    data = pd.read_excel(xlsx_path, header=0, dtype=str)

    phonemes = data['phoneme'].to_numpy()

    match value_mode:
        case 'phnm':
            phonemes, header, features, num_all_feats = _ipa_df2phnm(phonemes)
        case 'feat':
            header, features, num_all_feats = _ipa_df2feat(data, filter_non_distinctive)
        case 'mix':
            l_header, l_features, l_num_all_feats = _ipa_df2feat(
                data,
                filter_non_distinctive
            )
            phonemes, r_header, r_features, r_num_all_feats = _ipa_df2phnm(phonemes)
            header = np.concatenate((l_header, r_header), axis=0)
            features = np.concatenate(
                (
                    np.pad(l_features, ((0, 1), (0, 0))),
                    # , constant_values=float('nan')),
                    r_features
                ), axis=1
            )
            num_all_feats = l_num_all_feats + r_num_all_feats
        case _:
            raise ValueError(f'Unknown value mode: {value_mode}')

    return phonemes, header, features, num_all_feats


# _art_phon_order = {
#     'lip-loc': {'den': -1, 'lab': 0, 'pro': 1},
#     'lip-open': {'cl': 0, 'cr': 1, 'n': 2, 'w': 3},
#     'tt-loc': {'ret': -1, 'p-a': 0, 'alv': 1, 'den': 2},
#     'tt-open': {'cl': 0, 'cr': 1, 'n': 2, 'm-n': 3, 'm': 4, 'w': 5},
#     'tb-loc': {'pha': -1, 'uvu': 0, 'vel': 1, 'pal': 2},
#     'tb-open': {'cl': 0, 'cr': 1, 'n': 2, 'm-n': 3, 'm': 4, 'w': 5},
#     'velum': {'cl': 0, 'op': 1},
#     'glottis': {'cl': 0, 'cr': 1, 'w': 2}
# }

_art_phon_order = {
    'lip-loc': {'den': 0, 'lab': 1, 'pro': 2},
    'lip-open': {'cl': 0, 'cr': 1, 'n': 3, 'w': 6},
    'tt-loc': {'ret': 3, 'p-a': 2, 'alv': 1, 'den': 0},
    'tt-open': {'cl': 0, 'cr': 1, 'n': 2, 'm-n': 3, 'm': 5, 'w': 8},
    'tb-loc': {'pha': 3, 'uvu': 2, 'vel': 1, 'pal': 0},
    'tb-open': {'cl': 0, 'cr': 1, 'n': 2, 'm-n': 3, 'm': 5, 'w': 8},
    'velum': {'cl': 0, 'op': 1},
    'glottis': {'cl': 0, 'cr': 1, 'w': 3}
}

# non-interpolatable space: categorial, no total order
_strat_art_phon_order = {
    'lip': {
        'pro_cl': 0,
        'pro_n': 1,
        'pro_w': 2,
        'lab_cl': 3,
        'lab_cr': 4,
        'lab_w': 5,
        'den_cl': 6,
        'den_cr': 7
    },
    'tongue': {
        'den_cl_pal_cl': 0,
        'den_cl_pal_n': 1,
        'den_cr_uvu_m': 2,
        'alv_cl_vel_m': 3,
        'alv_cl_uvu_n': 4,
        'alv_cl_uvu_m': 5,
        'alv_cr_vel_m': 6,
        'alv_cr_uvu_m': 7,
        'alv_n_vel_m': 8,
        'alv_n_uvu_m': 9,
        'alv_m-n_pal_n': 10,
        'alv_m-n_pal_m-n': 11,
        'alv_m_pal_m': 12,
        'alv_m_vel_m': 13,
        'alv_m_uvu_m': 14,
        'alv_m_uvu_w': 15,
        'alv_w_vel_m': 16,
        'alv_w_vel_w': 17,
        'alv_w_uvu_m-n': 18,
        'alv_w_pha_m-n': 19,
        'p-a_cr_pal_m-n': 20,
        'p-a_cr_pal_m': 21,
        'p-a_m_uvu_m-n': 22,
        'p-a_w_vel_cl': 23,
        'p-a_w_vel_cr': 24,
        'p-a_w_vel_n': 25,
        'p-a_w_uvu_n': 26,
        'p-a_w_uvu_m-n': 27,
        'p-a_w_pha_m-n': 28,
        'ret_n_uvu_m': 29
    },
    'glottis': {
        'cl_cl': 0,
        'cl_cr': 1,
        'cl_w': 2,
        'op_cr': 3
    }
}


def _str2float(feat_name, value):
    return _art_phon_order[feat_name].get(value, float('nan'))


def _stratify_feats(feats):
    str_feats = [
        feats[0],
        _strat_art_phon_order['lip']['_'.join(feats[1:3])],
    ]

    has_nan = [',' in f for f in feats[3:7]]
    if any(has_nan):
        str_feats.append(float('nan'))
    else:
        str_feats.append(_strat_art_phon_order['tongue']['_'.join(feats[3:7])])

    has_nan = [',' in f for f in feats[7:]]
    if any(has_nan):
        str_feats.append(float('nan'))
    elif len(feats) == 8:  # no velum
        str_feats.append(_art_phon_order['glottis'][feats[7]])
    else:
        str_feats.append(_strat_art_phon_order['glottis']['_'.join(feats[7:9])])

    return pd.Series(str_feats, index=['phone', 'l', 't', 'g'])


def _str2onehot(feat_name, value):
    try:
        idx = list(_art_phon_order[feat_name].keys()).index(value)
    except ValueError:
        onehot = [float('nan')] * len(_art_phon_order[feat_name])
    else:
        onehot = [0] * len(_art_phon_order[feat_name])
        onehot[idx] = 1

    return onehot


def _one_hot_feats(feats):
    # generate new index
    index = ['phone']
    for art, art_dict in _art_phon_order.items():
        for f in art_dict:
            index.append(f'{art}_{f}')

    # generate one-hot features
    str_feats = [feats.iloc[0]]
    for art, value in zip(_art_phon_order, feats.iloc[1:]):
        str_feats.extend(_str2onehot(art, value))

    return pd.Series(str_feats, index=index)


def process_art_phon(
        xlsx_path: Path,
        keep_velum: bool,
        value_mode: str
):
    data = pd.read_excel(xlsx_path, header=0, dtype=str)
    data.columns = map(str.lower, data.columns)

    phonemes = data['phone'].to_numpy()

    num_all_feats = len(data.columns) - 1

    if not keep_velum:
        data = data.drop('velum', axis=1)
        _art_phon_order.pop('velum')

    match value_mode:
        case 'scalar':
            for feat in _art_phon_order.keys():
                data[feat] = data[feat].apply(lambda v: _str2float(feat, v))
        case 'scalar-mix':
            for feat in _art_phon_order.keys():
                data[feat] = data[feat].apply(lambda v: _str2float(feat, v))

            data2 = pd.DataFrame(
                data=np.eye(len(phonemes), dtype=np.float32),
                index=data.index,
                columns=phonemes
            )
            data = data.join(data2)
        case 'one-hot':
            data = data.apply(_one_hot_feats, axis=1)
        case 'group-scalar':
            data = data.apply(_stratify_feats, axis=1)
        case 'one-hot-mix':
            data1 = data.apply(_one_hot_feats, axis=1)
            data2 = pd.DataFrame(
                data=np.eye(len(phonemes), dtype=np.float32),
                index=data1.index,
                columns=phonemes
            )
            data = data1.join(data2)
        case _:
            raise ValueError(f'Unknown value mode: {value_mode}')

    header = data.columns[1:].to_numpy()
    features = data.iloc[:, 1:].to_numpy(dtype=np.float32)

    return phonemes, header, features, num_all_feats


def main():
    args = parse_args()

    match args.features:
        case 'ipa':
            phonemes, header, features, num_all_feats = process_ipa(
                args.table_path,
                args.filter_non_distinctive,
                args.value_mode
            )
        case 'art_phono':
            phonemes, header, features, num_all_feats = process_art_phon(
                args.table_path,
                args.keep_velum,
                args.value_mode
            )
        case _:
            raise ValueError(f'Unknown feature set: {args.features}')

    num_filt_feats = features.shape[1]
    np.savez(
        args.save_path, phonemes=phonemes, header=header, features=features
    )
    print(
        f'Saved phonemes ({len(phonemes)}), features ({num_filt_feats} from '
        f'{num_all_feats}) and values in {args.save_path}'
    )


if __name__ == '__main__':
    main()
