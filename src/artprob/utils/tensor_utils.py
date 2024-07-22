from typing import Callable

import torch
from jaxtyping import Bool, Float


def row_norm(tensor, squared=False):
    norms = torch.einsum('...d,...d->...', tensor, tensor)

    if not squared:
        norms = torch.sqrt(norms)
    return norms


def batch_dot(ref, other):
    return torch.einsum('bx,bx->b', ref, other)


def euclidean_distances(
        ref, other, ref_norm_squared=None, oth_norm_squared=None, squared=False
):
    if ref_norm_squared is None:
        ref_norm_squared = row_norm(ref, squared=True)
    ref_sq = ref_norm_squared.unsqueeze(1)

    if oth_norm_squared is None:
        oth_norm_squared = row_norm(other, squared=True)
    other_sq = oth_norm_squared.unsqueeze(0)

    distances = -2. * torch.einsum('x...d,y...d->xy...', ref, other)
    distances += ref_sq
    distances += other_sq
    distances = torch.clip(distances, min=0.)

    if not squared:
        distances = torch.sqrt(distances)

    return distances


def _pearson_correlation(ref, other):
    return torch.corrcoef(torch.stack((ref, other)))[0, 1].item()


def pearson_correlation(
        ref: Float[torch.Tensor, "n d"],
        other: Float[torch.Tensor, "m d"]
) -> Float[torch.Tensor, "n m"]:
    ref_mean = ref - ref.mean(dim=1, keepdim=True)
    other_mean = other - other.mean(dim=1, keepdim=True)

    ref_square_sum = row_norm(ref_mean, squared=True).unsqueeze(1)
    oth_square_sum = row_norm(other_mean, squared=True).unsqueeze(0)

    return (ref_mean @ other_mean.T) / torch.mul(ref_square_sum, oth_square_sum).sqrt()


def save_checkpoint(
        model_state, criterion_state, optimizer_state, best_state, checkpoint_path
):

    state_dict = {
        "latest": model_state,
        "criterion": criterion_state,
        "optimiser": optimizer_state,
        "best": best_state
    }

    torch.save(state_dict, checkpoint_path)


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
                [Float[torch.Tensor, "n"]],
                Float[torch.Tensor, "n"]
            ] = torch.neg,
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

    def __call__(self, values: Float[torch.Tensor, "lay"]) -> Bool[torch.Tensor, "lay"]:
        scores = self._score_fn(values)

        if self._best_scores is None:
            self._best_scores = scores
            self._counters = torch.zeros_like(scores, dtype=torch.int)
            self._to_stop = torch.zeros_like(scores, dtype=torch.bool)
        else:
            no_improvement = scores < self._best_scores + self._min_delta
            to_reset = ~(self._to_stop | no_improvement)
            self._counters[~to_reset] += no_improvement[~to_reset]
            self._counters[to_reset] = 0
            self._best_scores[to_reset] = scores[to_reset]
            self._to_stop = self._counters >= self._patience

        return self._to_stop
