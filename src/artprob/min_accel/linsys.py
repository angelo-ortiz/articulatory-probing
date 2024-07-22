from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float


def _delta_lambda_matrix(
        timings: Float[torch.Tensor, "k"],
        T: float
) -> Float[torch.Tensor, "k+1 k+1"]:
    diffs = F.pad(timings, (0, 1), value=T).unsqueeze(1) - F.pad(timings, (1, 0))
    diffs = diffs**3
    diffs = torch.tril(diffs)
    return diffs


def clamped_full_linear(
        positions: Float[torch.Tensor, "k d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        initial_coords: Optional[Float[torch.Tensor, "2 d_x"]] = None
) -> Float[torch.Tensor, "k+2 d_x"]:
    A_pos = torch.cat(
        (
            3 * F.pad(timings, (0, 1), value=T).unsqueeze(1)**2,
            _delta_lambda_matrix(timings, T)
        ), dim=1
    )
    A_vel = torch.cat(
        (
            torch.as_tensor([2 * T, T**2], device=timings.device),
            (T - timings)**2
        ), dim=0
    )

    mat_A = torch.cat(
        (
            A_pos,
            A_vel.unsqueeze(0)
        ), dim=0
    )

    if initial_coords is None:
        initial_coords = torch.zeros(2, positions.size(1), device=positions.device)

    B_pos = positions - initial_coords[0] - initial_coords[1] * timings.unsqueeze(1)

    mat_B = torch.cat(
        (
            6 * B_pos,
            -6 * initial_coords[:1] - initial_coords[1] * T,
            -2 * initial_coords[1:]
        ), dim=0
    )

    a0_dlambdas = torch.linalg.solve(mat_A, mat_B)

    return a0_dlambdas


def natural_full_linear(
        positions: Float[torch.Tensor, "k d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None
) -> Float[torch.Tensor, "k+2 d_x"]:
    A_pos = torch.cat(
        (
            6 * F.pad(timings, (0, 1), value=T).unsqueeze(1),
            _delta_lambda_matrix(timings, T)
        ), dim=1
    )
    A_accel = torch.cat(
        (
            torch.as_tensor([0, T], device=timings.device),
            T - timings
        ), dim=0
    )

    mat_A = torch.cat(
        (
            A_pos,
            A_accel.unsqueeze(0)
        ), dim=0
    )

    if initial_pos is None:
        initial_pos = torch.zeros(positions.size(1), device=positions.device)

    mat_B = torch.cat(
        (
            6 * (positions - initial_pos),
            -6 * initial_pos.unsqueeze(0),
            torch.zeros(1, positions.size(1), device=positions.device)
        ), dim=0
    )

    v0_dlambdas = torch.linalg.solve(mat_A, mat_B)

    return v0_dlambdas


def _tridiagonal_solve(A_lower, A_diag, A_upper, B):
    length = B.size(0)
    new_A_upper = [A_upper[0] / A_diag[0]]
    new_B = [B[0] / A_diag[0]]
    for i in range(1, length - 1):
        denom = A_diag[i] - A_lower[i - 1] * new_A_upper[-1]
        new_A_upper.append(A_upper[i] / denom)
        new_B.append((B[i] - A_lower[i - 1]) / denom)
    denom = A_diag[length - 1] - A_lower[length - 2] * new_A_upper[-1]
    new_B.append((B[length - 1] - A_lower[length - 2]) / denom)

    solution = [new_B[length - 1]]
    for i in range(length - 2, -1, -1):
        solution.append(new_B[i] - new_A_upper[i] * solution[-1])

    solution.reverse()

    return torch.stack(solution, dim=0)


def natural_tridiagonal_linear(
        positions: Float[torch.Tensor, "k d_x"],
        timings: Float[torch.Tensor, "k"],
        T: float,
        initial_pos: Optional[Float[torch.Tensor, "d_x"]] = None
) -> Float[torch.Tensor, "k+1 4 d_x"]:
    delta_timing = torch.diff(
        timings,
        dim=0,
        prepend=torch.zeros_like(timings[:1]),
        append=T * torch.ones_like(timings[:1])
    )
    delta_timing_reciprocal = torch.reciprocal(delta_timing)
    delta_timing_reciprocal_squared = torch.square(delta_timing_reciprocal)

    if initial_pos is None:
        initial_pos = torch.zeros_like(positions[:1])
    else:
        initial_pos = initial_pos.unsqueeze(0)

    three_delta_pos = 3 * torch.diff(
        positions,
        dim=0,
        prepend=initial_pos,
        append=torch.zeros_like(positions[:1])
    )
    delta_pos_scaled = three_delta_pos * delta_timing_reciprocal_squared.unsqueeze(1)

    diagonal_coeffs = F.pad(delta_timing_reciprocal, pad=(0, 1)) \
                      + F.pad(delta_timing_reciprocal, pad=(1, 0))

    independent_coeffs = F.pad(delta_pos_scaled, pad=(0, 0, 0, 1)) \
                         + F.pad(delta_pos_scaled, pad=(0, 0, 1, 0))

    velocity = _tridiagonal_solve(
        delta_timing_reciprocal,
        2 * diagonal_coeffs,
        delta_timing_reciprocal,
        independent_coeffs
    )

    delta_timing_reciprocal = delta_timing_reciprocal.unsqueeze(1)
    delta_timing_reciprocal_squared = delta_timing_reciprocal_squared.unsqueeze(1)

    half_acceleration = delta_pos_scaled \
                        - (2 * velocity[:-1] - velocity[1:]) * delta_timing_reciprocal

    half_jerk = delta_pos_scaled * delta_timing_reciprocal \
                - 3 * velocity[:-1] * delta_timing_reciprocal_squared \
                - 3 * half_acceleration * delta_timing_reciprocal

    all_coords = torch.stack(
        (
            torch.cat((initial_pos, positions), dim=0),
            velocity[:-1],
            2 * half_acceleration,
            2 * half_jerk
        ), dim=1
    )

    return all_coords
