from typing import Optional

import torch
from jaxtyping import Float

from artprob import log
from artprob.min_accel.inf_solver import inf_natural_min_accel_traj


def natural_min_accel_traj(
        positions: Float[torch.Tensor, "k d_x"],
        T: float,
        look_ahead: int,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None,
        max_iter: int = 1000,
        lr: float = 1e-4,
        knot_lr: float = 1e-3,
        weak_coeff: float = 0,
        patience: int = 3,
        min_delta: float = 0,
        history: bool = False
):
    # constants
    num_knots, d_x = positions.size()

    timings = torch.zeros(num_knots + 1, device=positions.device)
    all_derivatives = torch.zeros(num_knots + 1, 4, d_x, device=positions.device)
    all_derivatives[1:, 0] = positions
    if initial_pos is not None:
        all_derivatives[0, 0] = initial_pos

    if num_knots >= look_ahead:
        for k in range(num_knots - look_ahead + 1):
            T_k = (look_ahead + 1) * (T - timings[k].item()) / (num_knots - k + 1)

            _derivatives, _timings, _, _ = inf_natural_min_accel_traj(
                positions[k:k + look_ahead].detach(),
                T_k,
                initial_pos=all_derivatives[k, 0].detach(),
                timing_bounds=None,
                max_iter=max_iter,
                lr=lr,
                knot_lr=knot_lr,
                weak_coeff=weak_coeff,
                patience=patience,
                min_delta=min_delta,
                history=history,
                loglevel=log.Level.WARNING,
            )

            timings[k + 1] = timings[k] + _timings[0]
            all_derivatives[k + 1, 1:3] = _derivatives[1, 1:3]
            all_derivatives[k, 3:] = _derivatives[0, 3:]

            timings[-look_ahead + 1:] = timings[-look_ahead - 1] + _timings[1:]
            all_derivatives[-look_ahead + 1:, 1:3] = _derivatives[2:, 1:3]
            all_derivatives[-look_ahead:, 3:] = _derivatives[1:, 3:]
    elif num_knots > 0:
        all_derivatives, timings, _, _ = inf_natural_min_accel_traj(
            positions.detach(),
            T,
            initial_pos=initial_pos.detach(),
            timing_bounds=None,
            max_iter=max_iter,
            lr=lr,
            knot_lr=knot_lr,
            weak_coeff=weak_coeff,
            patience=patience,
            min_delta=min_delta,
            history=history,
            loglevel=log.Level.WARNING,
        )

    return all_derivatives, timings
