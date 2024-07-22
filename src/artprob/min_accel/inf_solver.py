import time
from typing import Optional

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F

from artprob import log
from artprob.min_accel.linsys import clamped_full_linear, natural_full_linear, \
    natural_tridiagonal_linear
from artprob.utils.optimal_control import compute_coords, compute_derivatives
from artprob.utils.tensor_utils import EarlyStopping, batch_dot, row_norm


def _compute_cubic_cost(all_accel, all_jerk, timings, T):
    deltas = torch.diff(
        timings,
        n=1,
        dim=0,
        prepend=torch.zeros_like(timings[:1]),
        append=T * torch.ones_like(timings[:1])
    )

    cost = row_norm(all_accel, squared=True) @ deltas \
           + batch_dot(all_accel, all_jerk) @ deltas**2 \
           + row_norm(all_jerk, squared=True) @ deltas**3 / 3
    return cost


def compute_cost(
        positions: Float[torch.Tensor, "k d_x"],
        initial_coords: Float[torch.Tensor, "2 d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        a0_dlambdas: Float[torch.Tensor, "k+2 d_x"]
) -> Float[torch.Tensor, ""]:
    accel = compute_derivatives(
        positions,
        initial_coords,
        timings,
        a0_dlambdas,
        n=1
    )[:, 0]

    all_accel = torch.cat((a0_dlambdas[:1], accel), dim=0)
    all_jerk = torch.cumsum(a0_dlambdas[1:], dim=0)

    cost = _compute_cubic_cost(all_accel, all_jerk, timings, T)

    return cost


# TODO: `a0_dlambdas` from `v0_dlambdas`?
def compute_cost_natural(
        positions: Float[torch.Tensor, "k d_x"],
        initial_pos: Float[torch.Tensor, "d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        v0_dlambdas: Float[torch.Tensor, "k+2 d_x"]
) -> Float[torch.Tensor, ""]:
    initial_coords = torch.stack((initial_pos, v0_dlambdas[0]), dim=0)
    a0_dlambdas = F.pad(v0_dlambdas[1:], pad=(0, 0, 1, 0))
    cost = compute_cost(positions, initial_coords, timings, T, a0_dlambdas)
    return cost


def compute_coords_natural(
        positions: Float[torch.Tensor, "k d_x"],
        initial_pos: Float[torch.Tensor, "d_x"],
        timings: Float[torch.Tensor, "k"],
        v0_dlambdas: Float[torch.Tensor, "k+2 d_x"]
) -> Float[torch.Tensor, "k+1 4 d_x"]:
    initial_coords = torch.stack((initial_pos, v0_dlambdas[0]), dim=0)
    fst_der_deltas = F.pad(v0_dlambdas[1:], pad=(0, 0, 1, 0))
    all_coords = compute_coords(positions, initial_coords, timings, fst_der_deltas)
    return all_coords


def inf_clamped_min_accel_traj(
        positions: Float[torch.Tensor, "k d_x"],
        T: float,
        initial_coords: Optional[Float[torch.Tensor, "2 d_x"]] = None,
        max_iter: int = 1000,
        lr: float = 1e-4,
        knot_lr: float = 1e-3,
        weak_coeff: float = 0,
        patience: int = 3,
        min_delta: float = 0,
        history: bool = False,
        loglevel: log.Level = log.Level.WARNING
):
    t_start = time.perf_counter()

    logger = log.get_logger('inf-clamped-min-accel', loglevel)

    early_stopping = EarlyStopping(
        patience=patience,
        score_fn=torch.neg,
        min_delta=min_delta
    )

    knots, d_x = positions.size()

    if initial_coords is None:
        initial_coords = torch.zeros(2, d_x, device=positions.device)

    # hypothesis: uniform duration between targets
    timings = (torch.linspace(0, T, knots + 2, device=positions.device)[1:-1]
               .requires_grad_(True))
    logger.debug(f'Initial timings: {timings}')

    timing_optim = torch.optim.Adam([timings], lr=lr)

    if weak_coeff > 0:
        underspec_positions = positions.detach().clone().requires_grad_(True)
        pos_optim = torch.optim.Adam([underspec_positions], lr=knot_lr)
        mse = nn.MSELoss(reduction='mean')
    else:
        underspec_positions = positions
    logger.debug(f'Initial positions: {underspec_positions}')

    if history:
        timing_history = []
        cost_history = []

    stopped = False

    for i in range(max_iter):
        logger.debug(f'Iteration {i}:')

        timing_optim.zero_grad()
        if weak_coeff > 0:
            pos_optim.zero_grad()

        a0_dlambdas = clamped_full_linear(
            underspec_positions, timings, T, initial_coords
        )
        logger.debug(f'\ta0: {a0_dlambdas[0]}')
        logger.debug(f'\tdelta lambdas: {a0_dlambdas[1:]}')

        timing_cost = compute_cost(
            underspec_positions,
            initial_coords,
            timings,
            T,
            a0_dlambdas
        )
        logger.debug(f'\tTiming cost: {timing_cost}')

        if weak_coeff > 0:
            pos_cost = mse(underspec_positions, positions)
        else:
            pos_cost = 0
        logger.debug(f'\tPosition cost: {pos_cost}')

        cost = timing_cost + weak_coeff * pos_cost
        logger.debug(f'\tTotal cost: {cost}')

        if history:
            timing_history.append(timings.detach().cpu().numpy())
            cost_history.append(cost.detach().cpu().item())

        stop_optim = early_stopping(cost)
        if torch.all(stop_optim):
            stopped = True
            logger.debug('\tEarly stopped!')
            break

        cost.backward()
        timing_optim.step()
        if weak_coeff > 0:
            pos_optim.step()

        logger.debug(f'\tTimings: {timings}')
        logger.debug(f'\tPositions: {underspec_positions}')

    if not stopped:
        a0_dlambdas = clamped_full_linear(
            underspec_positions, timings, T, initial_coords
        )
        logger.debug(f'a0: {a0_dlambdas[0]}')
        logger.debug(f'delta lambdas: {a0_dlambdas[1:]}')

        cost = compute_cost(
            underspec_positions,
            initial_coords,
            timings,
            T,
            a0_dlambdas
        )

    all_coords = compute_coords(
        underspec_positions,
        initial_coords,
        timings,
        a0_dlambdas
    )

    t_optim = time.perf_counter() - t_start
    logger.debug(f'Execution time: {t_optim}')

    if history:
        return all_coords, timings, t_optim, timing_history, cost_history
    else:
        return all_coords, timings, cost.item(), t_optim


# TODO: check correctness
def inf_natural_min_accel_traj_old(
        positions: Float[torch.Tensor, "k d_x"],
        T: float,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None,
        timing_bounds: Optional[Float[torch.Tensor, "k 2"]] = None,
        max_iter: int = 1000,
        lr: float = 1e-4,
        knot_lr: float = 1e-3,
        weak_coeff: float = 0,
        patience: int = 3,
        min_delta: float = 0,
        history: bool = False,
        loglevel: log.Level = log.Level.WARNING
):
    t_start = time.perf_counter()

    logger = log.get_logger('inf-natural-min-accel', loglevel)

    early_stopping = EarlyStopping(
        patience=patience,
        score_fn=torch.neg,
        min_delta=min_delta
    )

    knots, d_x = positions.size()

    if initial_pos is None:
        initial_pos = torch.zeros(d_x, device=positions.device)

    # hypothesis: uniform duration between targets
    timings = (torch.linspace(0, T, knots + 2, device=positions.device)[1:-1]
               .requires_grad_(True))
    logger.debug(f'Initial timings: {timings}')

    timing_optim = torch.optim.Adam([timings], lr=lr)

    if weak_coeff > 0:
        underspec_positions = positions.detach().clone().requires_grad_(True)
        pos_optim = torch.optim.Adam([underspec_positions], lr=knot_lr)
        mse = nn.MSELoss(reduction='mean')
    else:
        underspec_positions = positions
    logger.debug(f'Initial positions: {underspec_positions}')

    if history:
        timing_history = []
        cost_history = []

    stopped = False

    for i in range(max_iter):
        logger.debug(f'Iteration {i}:')

        timing_optim.zero_grad()
        if weak_coeff > 0:
            pos_optim.zero_grad()

        v0_dlambdas = natural_full_linear(underspec_positions, timings, T, initial_pos)
        logger.debug(f'\tv0: {v0_dlambdas[0]}')
        logger.debug(f'\tdelta lambdas: {v0_dlambdas[1:]}')

        timing_cost = compute_cost_natural(
            underspec_positions,
            initial_pos,
            timings,
            T,
            v0_dlambdas
        )
        logger.debug(f'\tTiming cost: {timing_cost}')

        if weak_coeff > 0:
            pos_cost = mse(underspec_positions, positions)
        else:
            pos_cost = 0
        logger.debug(f'\tPosition cost: {pos_cost}')

        cost = timing_cost + weak_coeff * pos_cost
        logger.debug(f'\tTotal cost: {cost}')

        if history:
            timing_history.append(timings.detach().cpu().numpy())
            cost_history.append(cost.detach().cpu().item())

        stop_optim = early_stopping(cost)
        if torch.all(stop_optim):
            stopped = True
            logger.debug('\tEarly stopped!')
            break

        cost.backward()
        timing_optim.step()
        if weak_coeff > 0:
            pos_optim.step()

        logger.debug(f'\tTimings: {timings}')
        logger.debug(f'\tPositions: {underspec_positions}')

    if not stopped:
        v0_dlambdas = natural_full_linear(underspec_positions, timings, T, initial_pos)
        logger.debug(f'v0: {v0_dlambdas[0]}')
        logger.debug(f'delta lambdas: {v0_dlambdas[1:]}')

        cost = compute_cost_natural(
            underspec_positions,
            initial_pos,
            timings,
            T,
            v0_dlambdas
        )

    all_coords = compute_coords_natural(
        underspec_positions,
        initial_pos,
        timings,
        v0_dlambdas
    )

    t_optim = time.perf_counter() - t_start
    logger.debug(f'Execution time: {t_optim}')

    if history:
        return all_coords, timings, t_optim, timing_history, cost_history
    else:
        return all_coords, timings, cost.item(), t_optim


def inf_natural_min_accel_traj_fixed_timings_old(
        positions: Float[torch.Tensor, "k d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None,
        loglevel: log.Level = log.Level.WARNING
):
    t_start = time.perf_counter()

    logger = log.get_logger('inf-natural-min-accel-fixed', loglevel)

    d_x = positions.size(1)

    if initial_pos is None:
        initial_pos = torch.zeros(d_x, device=positions.device)

    v0_dlambdas = natural_full_linear(positions, timings, T, initial_pos)
    logger.debug(f'v0: {v0_dlambdas[0]}')
    logger.debug(f'delta lambdas: {v0_dlambdas[1:]}')
    t_mid = time.perf_counter()
    logger.debug(f'Solve time: {t_mid - t_start:.3e}')

    all_coords = compute_coords_natural(
        positions,
        initial_pos,
        timings,
        v0_dlambdas
    )
    logger.debug(f'Evaluation time: {time.perf_counter() - t_mid:.3e}')

    _derivative_names = ('position', 'velocity', 'acceleration', 'jerk')
    for name, knot_coord in zip(_derivative_names, torch.unbind(all_coords, dim=1)):
        logger.debug(f'{name}: {knot_coord}')

    t_optim = time.perf_counter() - t_start
    logger.debug(f'Execution time: {t_optim}')

    return all_coords, t_optim


def inf_natural_min_accel_traj(
        positions: Float[torch.Tensor, "k d_x"],
        T: float,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None,
        timing_bounds: Optional[Float[torch.Tensor, "k 2"]] = None,
        max_iter: int = 1000,
        lr: float = 1e-4,
        knot_lr: float = 1e-3,
        weak_coeff: float = 0,
        patience: int = 3,
        min_delta: float = 0,
        history: bool = False,
        loglevel: log.Level = log.Level.WARNING
):
    t_start = time.perf_counter()

    logger = log.get_logger('inf-natural-min-accel', loglevel)

    early_stopping = EarlyStopping(
        patience=patience,
        score_fn=torch.neg,
        min_delta=min_delta
    )

    knots, d_x = positions.size()

    if initial_pos is None:
        initial_pos = torch.zeros(d_x, device=positions.device)

    # hypothesis: uniform duration between targets
    timings = (torch.linspace(0, T, knots + 2, device=positions.device)[1:-1]
               .requires_grad_(True))
    logger.debug(f'Initial timings: {timings}')

    timing_optim = torch.optim.Adam([timings], lr=lr)

    if weak_coeff > 0:
        underspec_positions = positions.detach().clone().requires_grad_(True)
        pos_optim = torch.optim.Adam([underspec_positions], lr=knot_lr)
        mse = nn.MSELoss(reduction='mean')
    else:
        underspec_positions = positions
    logger.debug(f'Initial positions: {underspec_positions}')

    if history:
        timing_history = []
        cost_history = []

    stopped = False

    for i in range(max_iter):
        logger.debug(f'Iteration {i}:')

        timing_optim.zero_grad()
        if weak_coeff > 0:
            pos_optim.zero_grad()

        all_coords = natural_tridiagonal_linear(
            underspec_positions,
            timings,
            T,
            initial_pos
        )

        timing_cost = _compute_cubic_cost(
            all_coords[:, 2],
            all_coords[:, 3],
            timings,
            T
        )

        logger.debug(f'\tTiming cost: {timing_cost}')

        if weak_coeff > 0:
            pos_cost = mse(underspec_positions, positions)
        else:
            pos_cost = 0
        logger.debug(f'\tPosition cost: {pos_cost}')

        cost = timing_cost + weak_coeff * pos_cost
        logger.debug(f'\tTotal cost: {cost}')

        if history:
            timing_history.append(timings.detach().cpu().numpy())
            cost_history.append(cost.detach().cpu().item())

        stop_optim = early_stopping(cost)
        if torch.all(stop_optim):
            stopped = True
            logger.debug('\tEarly stopped!')
            break

        cost.backward()
        timing_optim.step()
        if weak_coeff > 0:
            pos_optim.step()

        logger.debug(f'\tTimings: {timings}')
        logger.debug(f'\tPositions: {underspec_positions}')

    if not stopped:
        all_coords = natural_tridiagonal_linear(
            underspec_positions,
            timings,
            T,
            initial_pos
        )

        cost = _compute_cubic_cost(
            all_coords[:, 2],
            all_coords[:, 3],
            timings,
            T
        )

    t_optim = time.perf_counter() - t_start
    logger.debug(f'Execution time: {t_optim}')

    if history:
        return all_coords, timings, t_optim, timing_history, cost_history
    else:
        return all_coords, timings, cost.item(), t_optim


def inf_natural_min_accel_traj_fixed_timings(
        positions: Float[torch.Tensor, "k d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None,
        loglevel: log.Level = log.Level.WARNING
):
    t_start = time.perf_counter()

    logger = log.get_logger('inf-natural-min-accel-fixed', loglevel)

    d_x = positions.size(1)

    if initial_pos is None:
        initial_pos = torch.zeros(d_x, device=positions.device)

    all_coords = natural_tridiagonal_linear(positions, timings, T, initial_pos)

    _derivative_names = ('position', 'velocity', 'acceleration', 'jerk')
    for name, knot_coord in zip(_derivative_names, torch.unbind(all_coords, dim=1)):
        logger.debug(f'{name}: {knot_coord}')

    t_optim = time.perf_counter() - t_start
    logger.debug(f'Execution time: {t_optim}')

    return all_coords, t_optim
