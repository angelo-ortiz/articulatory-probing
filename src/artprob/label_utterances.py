import argparse
from pathlib import Path

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Transcription separation')

    parser.add_argument(
        '--path', type=str, required=True, help='The path to the transcriptions file.'
    )
    parser.add_argument(
        '--mocha',
        type=str,
        required=True,
        help='The path of the clean MOCHA-TIMIT dataset where to write the transcriptions.',
    )

    args = parser.parse_args()
    args.path = Path(args.path)
    args.mocha = Path(args.mocha)

    return args


def main():
    args = parse_args()

    speakers = [sp for sp in args.mocha.iterdir() if sp.is_dir()]
    sp_bname = [sp.name for sp in speakers]
    print('Found the following speakers for MOCHA-TIMIT: ' + ', '.join(sp_bname))
    done = set()

    with open(args.path, 'r') as fp:
        for line in tqdm.tqdm(fp):
            if not line.rstrip():
                continue

            # split the utterance number and content
            number, utt = line.split('.', maxsplit=1)

            # remove the spaces between the utterance number and content
            if number not in done:
                utt = utt.lstrip()

            for sp in speakers:
                # the name of the corresponding label file
                fname = sp / f'{sp.name}_{number}.lab'

                if fname.with_suffix('.wav').exists():
                    # save the utterance for all the speakers considered that have a
                    # matching waveform file
                    with open(fname, 'a') as out:
                        out.write(utt.rstrip('\n'))

                done.add(number)

    print('Done!')


if __name__ == '__main__':
    main()
