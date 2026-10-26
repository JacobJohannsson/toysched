"""Resource models: the footprint step, then the contention fixed point.

The footprint step turns each running task's time breakdown into an effective one,
given how full the footprint resources are. The contention computation then solves
for the rate factors g_r of the capacity resources on that effective breakdown.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class FootprintResult:
    unslowed: np.ndarray  # (m,) f_{j,0} + f^hit_j, plus misses with no overflow target
    bound: np.ndarray  # (m, N) f'_{j,r}
    usage: np.ndarray  # (m, N) a'_{j,r}
    load: np.ndarray  # (K,) V_c
    hit_fraction: np.ndarray  # (K,) h_c


def apply_footprints(
    breakdown: np.ndarray,
    rho: np.ndarray,
    footprint_shared: np.ndarray,
    footprint_slot: np.ndarray,
    slot_counts: np.ndarray,
    limits: np.ndarray,
    overflow: Sequence[int | None],
    miss_penalty: np.ndarray,
) -> FootprintResult:
    """Effective breakdown and usage of running tasks after footprint misses.

    Args:
        breakdown: (m, 1 + K + N) time breakdowns: compute, footprint, capacity.
        rho: (m, N) ρ_{j,r}.
        footprint_shared: (m, K) w^sh_{j,c}, held once per task.
        footprint_slot: (m, K) w^sl_{j,c}, held once per slot.
        slot_counts: (m,) k_j.
        limits: (K,) B_c.
        overflow: o(c) for each footprint resource, an index into the capacity
            resources, or None.
        miss_penalty: (K,) η_c.
    """
    K = len(limits)
    f0 = breakdown[:, 0]
    fc = breakdown[:, 1 : 1 + K]
    bound = breakdown[:, 1 + K :].copy()

    load = (footprint_shared + slot_counts[:, None] * footprint_slot).sum(axis=0)
    with np.errstate(divide="ignore"):
        hit = np.where(load > limits, limits / load, 1.0)

    unslowed = f0 + fc @ hit
    for c, target in enumerate(overflow):
        miss_time = miss_penalty[c] * fc[:, c] * (1 - hit[c])
        if target is None:
            unslowed = unslowed + miss_time
        else:
            bound[:, target] += miss_time
    return FootprintResult(unslowed, bound, bound * rho, load, hit)


@dataclass
class ContentionResult:
    speed: np.ndarray  # v per slot, one entry per running task
    rate_factor: np.ndarray  # g_r
    load: np.ndarray  # U_r
    iterations: int
    converged: bool


def solve_contention(
    unslowed: np.ndarray,
    bound: np.ndarray,
    usage: np.ndarray,
    base_speed: np.ndarray,
    slot_counts: np.ndarray,
    capacities: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> ContentionResult:
    """Per-slot speeds and capacity loads for a set of running tasks.

    Args:
        unslowed: (m,) time per unit of work that contention never slows.
        bound: (m, N) time per unit of work bound by each capacity resource.
        usage: (m, N) average usage on one slot at full speed.
        base_speed: (m,) s_j[k_j], the per-slot speed before contention.
        slot_counts: (m,) k_j.
        capacities: (N,) C_r.
    """
    m, n = usage.shape
    g = np.ones(n)
    capacities = np.asarray(capacities, dtype=float)

    def evaluate(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = base_speed / (unslowed + bound @ (1.0 / g))
        load = (slot_counts * v) @ usage
        return v, load

    v, load = evaluate(g)
    if n == 0 or m == 0:
        return ContentionResult(v, g, load, 0, True)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(load > 0, capacities / load, np.inf)
        g_new = np.minimum(1.0, g * ratio)
        delta = np.max(np.abs(g_new - g))
        g = g_new
        v, load = evaluate(g)
        if delta <= tol:
            converged = True
            break
    return ContentionResult(v, g, load, it, converged)
