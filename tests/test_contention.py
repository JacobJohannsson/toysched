import numpy as np
import pytest

from toysched.contention import apply_footprints, solve_contention


def _solve(breakdown, usage, base_speed, slot_counts, capacities):
    """Breakdown is (compute, capacity resources...), with no footprint resources."""
    breakdown = np.array(breakdown, dtype=float)
    return solve_contention(
        breakdown[:, 0],
        breakdown[:, 1:],
        np.array(usage, dtype=float),
        np.array(base_speed, dtype=float),
        np.array(slot_counts, dtype=float),
        np.array(capacities, dtype=float),
    )


def test_no_contention_runs_at_base_speed():
    res = _solve([[0.6, 0.3, 0.1]], [[0.09, 0.06]], [0.9], [2], [1, 1])
    np.testing.assert_allclose(res.rate_factor, [1, 1])
    np.testing.assert_allclose(res.speed, [0.9])
    np.testing.assert_allclose(res.load, [2 * 0.9 * 0.09, 2 * 0.9 * 0.06])


def test_fully_bound_tasks_solve_in_closed_form():
    # f_0 = 0, so v = s g and U = g Σ k a s; then g = C / Σ k a s.
    res = _solve([[0, 1], [0, 1]], [[0.5], [0.3]], [1, 0.8], [2, 3], [1])
    g = 1 / (2 * 0.5 * 1 + 3 * 0.3 * 0.8)
    assert res.rate_factor[0] == pytest.approx(g)
    assert res.load[0] == pytest.approx(1)


def test_equilibrium_conditions():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = rng.integers(1, 9)
        f = rng.dirichlet([5, 4, 0.5], size=m)
        usage = f[:, 1:] * rng.uniform(0.1, 1, size=(m, 2))
        s = rng.uniform(0.3, 1, size=m)
        k = rng.integers(1, 4, size=m).astype(float)
        res = _solve(f, usage, s, k, [1, 1])
        assert res.converged
        g = res.rate_factor
        # Speeds follow Amdahl's law at the solved factors
        np.testing.assert_allclose(res.speed, s / (f[:, 0] + f[:, 1:] @ (1 / g)))
        # No resource over capacity, and full wherever it is slowed
        assert np.all(res.load <= 1 + 1e-6)
        for r in range(2):
            if g[r] < 1 - 1e-6:
                assert res.load[r] == pytest.approx(1, abs=1e-6)


def test_amdahl_example_from_spec():
    # With g_mem = 0.5, f_mem = 0.8 runs at 1 / 1.8 and f_mem = 0.2 at 1 / 1.2.
    for f_mem, expected in [(0.8, 0.56), (0.2, 0.83)]:
        assert 1 / ((1 - f_mem) + f_mem / 0.5) == pytest.approx(expected, abs=0.01)


def test_pure_compute_task_is_never_slowed():
    res = _solve([[1, 0], [0, 1]], [[0], [0.9]], [1, 1], [1, 4], [1])
    assert res.speed[0] == pytest.approx(1)
    assert res.rate_factor[0] < 1


# ---------------------------------------------------------------- footprint step


def _footprints(
    breakdown, rho, shared, slot, k, limits=(1.0,), overflow=(0,), eta=(3.0,)
):
    return apply_footprints(
        np.array(breakdown, dtype=float),
        np.array(rho, dtype=float),
        np.array(shared, dtype=float),
        np.array(slot, dtype=float),
        np.array(k, dtype=float),
        np.array(limits, dtype=float),
        overflow,
        np.array(eta, dtype=float),
    )


# Breakdown order: compute, cache, memory bandwidth, storage I/O
F = [[0.5, 0.2, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
RHO = [[0.3, 0.6], [0.4, 0.5]]


def test_footprint_that_fits_changes_nothing():
    fp = _footprints(F, RHO, [[0.2], [0.3]], [[0.01], [0.02]], [2, 3])
    assert fp.load[0] == pytest.approx(0.2 + 0.02 + 0.3 + 0.06)
    assert fp.hit_fraction[0] == 1
    f = np.array(F)
    np.testing.assert_allclose(fp.unslowed, f[:, 0] + f[:, 1])
    np.testing.assert_allclose(fp.bound, f[:, 2:])
    np.testing.assert_allclose(fp.usage, f[:, 2:] * np.array(RHO))


def test_shared_footprint_counts_once_per_task_and_slot_part_per_slot():
    one_task_many_slots = _footprints(F[:1], RHO[:1], [[0.5]], [[0.05]], [8])
    assert one_task_many_slots.load[0] == pytest.approx(0.5 + 8 * 0.05)
    many_tasks = _footprints(F * 4, RHO * 4, [[0.5]] * 8, [[0.05]] * 8, [1] * 8)
    assert many_tasks.load[0] == pytest.approx(8 * 0.55)
    assert many_tasks.hit_fraction[0] == pytest.approx(1 / (8 * 0.55))


def test_overflow_moves_misses_onto_the_target_resource():
    # Load 2.0 on a cache of 1: half the cache accesses miss
    fp = _footprints(F, RHO, [[1.0], [1.0]], [[0.0], [0.0]], [1, 1])
    h = 0.5
    assert fp.hit_fraction[0] == pytest.approx(h)
    f = np.array(F)
    np.testing.assert_allclose(fp.unslowed, f[:, 0] + f[:, 1] * h)
    expected_mem = f[:, 2] + 3.0 * f[:, 1] * (1 - h)
    np.testing.assert_allclose(fp.bound[:, 0], expected_mem)
    np.testing.assert_allclose(fp.bound[:, 1], f[:, 3])  # I/O untouched
    np.testing.assert_allclose(fp.usage[:, 0], expected_mem * np.array(RHO)[:, 0])


def test_overflow_to_none_only_costs_latency():
    fp = _footprints(F, RHO, [[1.0], [1.0]], [[0.0], [0.0]], [1, 1], overflow=(None,))
    f = np.array(F)
    np.testing.assert_allclose(
        fp.unslowed, f[:, 0] + f[:, 1] * 0.5 + 3.0 * f[:, 1] * 0.5
    )
    np.testing.assert_allclose(fp.bound, f[:, 2:])


def test_misses_saturate_memory_and_slow_tasks():
    """End to end: the same tasks run slower when their footprints overflow."""

    def speeds(shared):
        fp = _footprints(F, RHO, shared, [[0.0], [0.0]], [4, 4])
        res = solve_contention(
            fp.unslowed,
            fp.bound,
            fp.usage,
            np.ones(2),
            np.array([4.0, 4.0]),
            np.ones(2),
        )
        return res

    fits, spills = speeds([[0.3], [0.3]]), speeds([[1.5], [1.5]])
    assert np.all(spills.speed < fits.speed)
    # Fitting leaves memory at about 0.56 load; the misses push it past capacity
    assert fits.rate_factor[0] == 1 and fits.load[0] < 1
    assert spills.rate_factor[0] < 1
    assert spills.load[0] == pytest.approx(1, abs=1e-6)
