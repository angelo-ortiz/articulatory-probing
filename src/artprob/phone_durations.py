import argparse
from math import ceil
from operator import itemgetter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import textgrids

from gsp.ema_eval.dataset import AudioDataset, find_file_paths


def plot_hist(duration_dict, print_not_plot=True):
    all_durations = [d for dur in duration_dict.values() for d in dur]
    upper_bound = ceil(max(all_durations))
    print(sorted(all_durations)[-10:])

    # Plot histogram
    if print_not_plot:
        counts, edges = np.histogram(
            all_durations,
            bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1., upper_bound],  # 'doane',
            density=False
        )
        for i, c in enumerate(counts[:-1]):
            print(f'| [{edges[i]:.2f}, {edges[i + 1]:.2f}) | {c} |')
        print(f'| [{edges[-2]:.2f}, {edges[-1]:.2f}] | {counts[-1]} |')
    else:
        plt.hist(all_durations)
        plt.show()


def sort_mean_std(duration_dict):
    duration_list = [(k, np.mean(vs), np.std(vs)) for k, vs in
                     duration_dict.items() if vs]

    # ascending sort according to the mean duration
    return sorted(duration_list, key=itemgetter(1))


def get_phone_duration(file_paths) -> Dict[str, List[float]]:
    phone_durations = {}
    long_count = 0
    for _, _, txg_path, _ in file_paths:
        try:
            grid = textgrids.TextGrid(txg_path)
        except Exception as e:
            print(f'Skipped txg->phon translation of {txg_path} because of {e}')
            continue

        for phone in grid['phones']:
            dur_list = phone_durations.setdefault(phone.text, [])
            dur = phone.xmax - phone.xmin
            if dur >= 0.5 and phone.text not in ['', 'spn']:
                long_count += 1
                print(f'"{phone.text}"', dur, txg_path)
            dur_list.append(dur)

    print(f'There were {long_count} (particularly) long phones (see above)')

    return phone_durations


def get_phoneme_durations(file_paths, features):
    dataset = AudioDataset(file_paths, features, 500, True, True, False)

    phoneme_durations = {}

    for paths in dataset.file_paths:
        # read file with phone transcription
        with open(paths[2], 'r') as fp:
            for line in fp:
                beg, end, phn = line.split()

                dur_list = phoneme_durations.setdefault(phn, [])
                dur_list.append(float(end) - float(beg))

    return sort_mean_std(phoneme_durations)


def parse_args():
    parser = argparse.ArgumentParser(description='Analysis of phone durations')

    parser.add_argument(
        '--dataset', type=str, required=True, help='The path to the dataset.'
    )
    parser.add_argument(
        '--features', type=str, required=True,
        help='The path to the phonological features.'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    train = Path(args.dataset)
    file_paths = find_file_paths(train)

    phone_durations = get_phone_duration(file_paths)

    plot_hist(phone_durations, print_not_plot=True)

    for tup in sort_mean_std(phone_durations):
        print(tup)

    phoneme_durations = get_phoneme_durations(file_paths, args.features)

    for tup in phoneme_durations:
        print(tup)


if __name__ == '__main__':
    main()
