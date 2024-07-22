from math import factorial as fact

import torch
from jaxtyping import Float


def _p_vector(
        ts: Float[torch.Tensor, "k"],
        n: int,
        device: torch.device
) -> Float[torch.Tensor, "k*{n}"]:
    size = n * len(ts)
    vector_p = torch.empty(size, device=device)

    for i in range(n):
        vector_p[torch.arange(i, size, n, device=device)] = ts**(n - i) / fact(n - i)

    return vector_p


def _j0_exp(ts: Float[torch.Tensor, "k"], n: int) -> Float[torch.Tensor, "k*{n} {n}"]:
    j0 = torch.diag(torch.ones(n - 1, device=ts.device), diagonal=1)
    exp_matrices = torch.matrix_exp(ts.view(-1, 1, 1) * j0)
    return exp_matrices.view(-1, exp_matrices.size(-1))


def compute_derivatives(
        positions: Float[torch.Tensor, "k d_x"],
        initial_coords: Float[torch.Tensor, "m d_x"],
        timings: Float[torch.Tensor, "k"],
        fst_der_deltas: Float[torch.Tensor, "k+m d_x"],
        n: int
) -> Float[torch.Tensor, "k {n} d_x"]:
    num_knots, d_x = positions.size()
    matrix_p = torch.zeros(n * num_knots, num_knots + 1, device=timings.device)
    prev_timing = 0
    for i in range(num_knots):
        matrix_p[n * i:, i] = _p_vector(
            timings[i:] - prev_timing, n, timings.device
        )
        prev_timing = timings[i]

    matrix_e = _j0_exp(timings, n)

    num_fst_der = initial_coords.size(0) - 1

    fst_derivatives = torch.cat((initial_coords, fst_der_deltas[:num_fst_der]), dim=0)

    all_derivatives = matrix_p @ fst_der_deltas[num_fst_der:] \
                      + matrix_e @ fst_derivatives[-n:]

    return all_derivatives.view(num_knots, n, d_x)


def compute_coords(
        positions: Float[torch.Tensor, "k d_x"],
        initial_coords: Float[torch.Tensor, "m d_x"],
        timings: Float[torch.Tensor, "k"],
        fst_der_deltas: Float[torch.Tensor, "k+m d_x"]
) -> Float[torch.Tensor, "k+1 2*m d_x"]:
    num_fst_der = initial_coords.size(0) - 1

    knot_derivatives = compute_derivatives(
        positions,
        initial_coords,
        timings,
        fst_der_deltas,
        n=2 * num_fst_der
    )

    knot_position_n_derivatives = torch.cat(
        (
            torch.cat((initial_coords, fst_der_deltas[:num_fst_der]), dim=0)
            .unsqueeze(0),
            torch.cat((positions.unsqueeze(1), knot_derivatives), dim=1)
        ), dim=0
    )

    all_coords = torch.cat(
        (
            knot_position_n_derivatives,
            torch.cumsum(fst_der_deltas[num_fst_der:], dim=0).unsqueeze(1)
        ), dim=1
    )

    return all_coords


def _evaluate(
        left_coords: Float[torch.Tensor, "k m d_x"],
        delta_t: Float[torch.Tensor, "k"]
) -> Float[torch.Tensor, "k d_x"]:
    num_der = left_coords.size(1)

    m_to_1 = []
    for i in range(num_der):
        m_to_1.append(delta_t**i / fact(i))

    m_to_1 = torch.stack(m_to_1, dim=1)

    evals = torch.einsum('kmd,km->kd', left_coords, m_to_1)

    return evals


def poly_interpolate(
        timings: Float[torch.Tensor, "k"],
        full_coords: Float[torch.Tensor, "k m d_x"],
        eval_timings: Float[torch.Tensor, "n"]
) -> Float[torch.Tensor, "n d_x"]:
    """Note: requires that the given coordinates and timings cover the initial (rest)
    knot as well as all the intermediate knots.

    Parameters
    ----------
    eval_timings
    full_coords
    timings

    Returns
    -------

    """
    assert full_coords.size(0) == timings.size(0), \
        (f'The number of coordinates ({full_coords.size(0)}) is different from the '
         f'number of timings ({timings.size(0)}): they should be the same!')

    sample_indices = torch.bucketize(eval_timings, timings, right=True) - 1
    dt = eval_timings - timings[sample_indices]
    sample_coords = full_coords[sample_indices]
    interpolated = _evaluate(sample_coords, dt)
    return interpolated
