from typing import Callable

import numpy as np
from jaxtyping import Bool, Float
from numpy import typing as npt

NDArrayFloat = npt.NDArray[np.float_]


def row_dot(a, b):
    return np.einsum('ij,ij->i', a, b)


def row_squared_norm(a):
    return row_dot(a, a)


def vector_cartesian_product(*arrays):
    la = len(arrays)
    dim = arrays[0].shape[-1]
    dtype = np.result_type(*arrays)
    la_list = [len(a) for a in arrays]
    arr = np.empty(la_list + [la, dim], dtype=dtype)
    for i, a in enumerate(np.ix_(*[np.arange(l) for l in la_list])):
        arr[..., i, :] = arrays[i][a]
    return arr.reshape(-1, la, dim)


def array2str_list(array: NDArrayFloat) -> list[str]:
    return [f'{el:.3f}' for el in array]


class EarlyStopping:
    """
    Class for implementing early stopping during model training.

    Parameters
    ----------
    patience : int
        How long to wait since the last time the score improved.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(
            self,
            patience: int,
            score_fn: Callable[
                [Float[np.ndarray, "n"]],
                Float[np.ndarray, "n"]
            ] = np.negative,
            min_delta: float = 0
    ):
        self._patience = patience
        self._score_fn = score_fn
        self._min_delta = min_delta

        self._best_scores = None
        self._counters = None
        self._to_stop = None

    @property
    def patience(self):
        return self._patience

    def __call__(self, values: Float[np.ndarray, "lay"]) -> Bool[np.ndarray, "lay"]:
        scores = self._score_fn(values)

        if self._best_scores is None:
            self._best_scores = scores
            self._counters = np.zeros_like(scores, dtype=int)
            self._to_stop = np.zeros_like(scores, dtype=bool)
        else:
            no_improvement = scores < self._best_scores + self._min_delta
            to_reset = ~(self._to_stop | no_improvement)
            self._counters[~to_reset] += no_improvement[~to_reset]
            self._counters[to_reset] = 0
            self._best_scores[to_reset] = scores[to_reset]
            self._to_stop = self._counters >= self._patience

        return self._to_stop
