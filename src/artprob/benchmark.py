import copy
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from artprob.utils.misc import flatten_lists


def _score_arg_max(
        layers: list[dict[str, float]],
        best_score: float
) -> tuple[list[tuple[str, int]], float]:
    matches = []
    for lay, ckpt_dict in enumerate(layers):
        for ckpt, score in ckpt_dict.items():
            if ckpt != 'best' and score == best_score:
                matches.append((ckpt, lay))
    return matches, best_score


class Results:
    _SP_DATA: list[str] = ['ckpt', 'layer', 'score']

    def __init__(self, db_path):
        self._db_path = db_path

        # db: {speaker -> {model -> ([{ckpt -> score}], best_score)}}
        self._db = self._read_db()

    def _read_db(self) -> dict[str, dict[str, tuple[list[dict[str, float]], float]]]:
        try:
            with open(self._db_path, 'r') as fp:
                db = json.load(fp)
        except FileNotFoundError:
            print(
                f'Info: the DB file {self._db_path} does not exist. It will be created.'
            )
            db = {}

        return db

    def save_db(self) -> None:
        with open(self._db_path, 'w') as fp:
            json.dump(self._db, fp, indent=2)

    def add_entry(
            self,
            model: str,
            path: Path,
            speaker: str,
            scores: Float[torch.Tensor, "lay"]
    ) -> None:
        ckpt = path.stem.split('_')[1]
        speaker_dict = self._db.setdefault(speaker, {})
        model_lists, model_best = speaker_dict.get(model, ([], -1.))
        fst_model_entry = len(model_lists) == 0
        for lay, scr in enumerate(scores):
            if fst_model_entry:
                layer_dict = {'best': -1.}
                model_lists.append(layer_dict)
            else:
                layer_dict = model_lists[lay]

            layer_dict[ckpt] = scr.item()
            if scr > layer_dict['best']:
                layer_dict['best'] = scr.item()
            if layer_dict['best'] > model_best:
                model_best = layer_dict['best']

        speaker_dict[model] = model_lists, model_best

    def _get_columns(self, speakers: list[str]):
        columns = [[f'{sp}: {dt}' for dt in self._SP_DATA] for sp in speakers]
        return flatten_lists(columns)

    def _process_model_n_speakers(
            self,
            model_sel: Optional[list[str]] = None,
            speaker_sel: Optional[list[str]] = None
    ) -> tuple[set[str] | None, set[str]]:
        model_sel = set(model_sel) if model_sel is not None else None
        speakers = self._db.keys()
        if speaker_sel is not None:
            speakers &= set(speaker_sel)
        return model_sel, speakers

    def benchmark(
            self,
            model_sel: Optional[list[str]] = None,
            speaker_sel: Optional[list[str]] = None,
    ) -> dict[str, dict[str, float]]:
        model_sel, speakers = self._process_model_n_speakers(model_sel, speaker_sel)
        reduced_db = {}
        for sp, sp_dict in self._db.items():
            if sp in speakers:
                models = sp_dict.keys()
                if model_sel is not None:
                    models &= model_sel

                # keep only the best score
                reduced_db[sp] = {mod: sp_dict[mod][1] for mod in models}

        return reduced_db

    def detailed_benchmark(
            self,
            model_sel: Optional[list[str]] = None,
            speaker_sel: Optional[list[str]] = None,
    ) -> dict[str, dict[str, str | int | float]]:
        model_sel, speakers = self._process_model_n_speakers(model_sel, speaker_sel)
        reduced_db = {}
        for sp, sp_dict in self._db.items():
            if sp in speakers:
                models = sp_dict.keys()
                if model_sel is not None:
                    models &= model_sel

                # keep the best score as well as its checkpoints and layers
                data_dicts = [{} for _ in range(len(self._SP_DATA))]
                for mod, mod_tuple in sp_dict.items():
                    matches, scr = _score_arg_max(*mod_tuple)
                    assert len(matches) > 0, \
                        f'Found no matches for {mod}\'s top score on {sp}'

                    if len(matches) > 1:
                        print(
                            f'Found {len(matches)} top scores for {mod} tested on {sp}:'
                            f' taking the first one'
                        )
                    data_dicts[0][mod] = matches[0][0]
                    data_dicts[1][mod] = matches[0][1]
                    data_dicts[2][mod] = scr

                for dt, dico in zip(self._SP_DATA, data_dicts):
                    reduced_db[f'{sp}-{dt}'] = dico

        return reduced_db


def print_benchmark(
        benchmark: dict[str, dict[str, Any]]
) -> None:
    models = set()
    for sp, sp_dict in benchmark.items():
        models |= sp_dict.keys()

    to_add = {}
    for mod in models:
        scores = [sp_dict.get(mod, np.nan) for sp, sp_dict in benchmark.items()]
        to_add[mod] = np.mean(scores)

    benchmark = copy.deepcopy(benchmark)
    benchmark.update({'avg': to_add})

    df = pd.DataFrame.from_dict(benchmark)
    df.sort_values(by='avg', ascending=False, inplace=True)

    print(df)
