import math

import numpy as np
import pytest

from toysched.config import Config
from toysched.workload import (
    coherency_for_peak,
    generate_workload,
    max_parallelism,
    sample_job,
    usl_speedup,
)


def test_usl_starts_at_one_and_decreases():
    for alpha, kappa in [(0.0, 0.0), (0.02, 0.0), (0.03, 0.02), (0.2, 0.1)]:
        s = usl_speedup(alpha, kappa, 8)
        assert s[0] == 1.0
        assert np.all(np.diff(s) <= 0)
        assert np.all(s > 0)


def test_no_serial_fraction_and_no_coherency_is_perfect_scaling():
    s = usl_speedup(0.0, 0.0, 8)
    assert np.all(s == 1)
    assert max_parallelism(s) == 8


def test_amdahl_example():
    # α = 0.03 on 8 slots: per-slot speed 1 / 1.21, total speedup 6.6x
    s = usl_speedup(0.03, 0.0, 8)
    assert s[7] == pytest.approx(1 / 1.21)
    assert 8 * s[7] == pytest.approx(6.61, abs=0.01)
    assert max_parallelism(s) == 8  # Amdahl alone never peaks


def test_interior_peak_example():
    # α = 0.03, κ = 0.02: continuous peak at √(0.97 / 0.02) ≈ 6.96, so k* = 7, and
    # the per-slot speed there is about 0.5
    s = usl_speedup(0.03, 0.02, 8)
    assert max_parallelism(s) == 7
    assert s[6] == pytest.approx(0.495, abs=0.001)


def test_k_star_matches_closed_form_peak():
    rng = np.random.default_rng(0)
    S = 8
    for _ in range(2000):
        alpha = rng.uniform(0, 0.3)
        peak = rng.uniform(0.3, 30)
        kappa = coherency_for_peak(alpha, peak)
        assert math.sqrt((1 - alpha) / kappa) == pytest.approx(peak)
        s = usl_speedup(alpha, kappa, S)
        total = lambda k: k * s[k - 1]  # noqa: E731
        lo = min(max(math.floor(peak), 1), S)
        hi = min(max(math.ceil(peak), 1), S)
        expected = lo if total(lo) >= total(hi) else hi
        assert max_parallelism(s) == expected


def test_sampled_speedup_parameters():
    cfg = Config(num_jobs=3000)
    tasks = [
        t for j in generate_workload(cfg, np.random.default_rng(0)) for t in j.tasks
    ]
    alpha = np.array([t.serial_fraction for t in tasks])
    peak = np.array([t.peak for t in tasks])
    k_star = np.array([t.max_parallelism for t in tasks])
    assert np.median(alpha) == pytest.approx(0.02, rel=0.1)
    assert np.median(peak) == pytest.approx(4 * cfg.num_slots, rel=0.1)
    # P(peak < S) = Φ(-ln 4 / 1.0) ≈ 0.08
    assert np.mean(peak < cfg.num_slots) == pytest.approx(0.08, abs=0.02)
    assert np.all(k_star[peak >= cfg.num_slots + 1] == cfg.num_slots)


def test_peak_none_disables_coherency():
    cfg = Config(num_jobs=50, peak_median_frac=None)
    for j in generate_workload(cfg, np.random.default_rng(0)):
        for t in j.tasks:
            assert t.coherency == 0.0 and t.max_parallelism == cfg.num_slots


def _check_dag(job):
    L = job.height
    tasks = job.tasks
    T = len(tasks)
    assert L <= T <= 2 ** (L - 1) or T == L

    levels = {}
    for t in tasks:
        levels.setdefault(t.level, []).append(t.id)
    assert sorted(levels) == list(range(1, L + 1))
    for l, ids in levels.items():
        assert len(ids) <= 3 ** (l - 1)
    assert levels[1] == [0]

    assert len(set(job.edges)) == len(job.edges)
    for p, c in job.edges:
        assert tasks[p].level < tasks[c].level
    for t in tasks:
        if t.level > 1:
            assert any(tasks[p].level == t.level - 1 for p in t.parents)
        else:
            assert t.parents == ()
        if t.level < L:
            assert t.children
        else:
            assert t.children == ()


def test_generated_dags_are_well_formed():
    cfg = Config()
    rng = np.random.default_rng(0)
    for i in range(500):
        _check_dag(sample_job(cfg, i, 0.0, rng))


@pytest.mark.parametrize("L", [1, 2, 3, 4, 5, 6])
def test_every_height_is_well_formed(L):
    cfg = Config(height_mean=L, height_std=0.0)
    rng = np.random.default_rng(L)
    for i in range(50):
        job = sample_job(cfg, i, 0.0, rng)
        assert job.height == L
        _check_dag(job)


def test_task_attributes():
    cfg = Config()
    jobs = generate_workload(cfg, np.random.default_rng(1))
    for job in jobs:
        for t in job.tasks:
            assert t.breakdown.sum() == pytest.approx(1.0)
            assert np.all(t.breakdown >= 0)
            assert np.all((t.rho > 0) & (t.rho <= 1))
            K = cfg.num_footprint_resources
            np.testing.assert_allclose(t.usage, t.breakdown[1 + K :] * t.rho)
            assert np.all(t.footprint_shared > 0) and np.all(t.footprint_slot > 0)
            assert t.work > 0
            assert 1 <= t.max_parallelism <= cfg.num_slots
            assert len(t.speedup) == cfg.num_slots


def test_breakdown_calibration():
    """Spec: E[f_cache] ≈ 0.20, E[f_mem] ≈ 0.20 and E[f_io] ≈ 0.11 by default."""
    cfg = Config()
    rng = np.random.default_rng(2)
    f = np.array(
        [
            t.breakdown
            for j in generate_workload(cfg.with_overrides(num_jobs=2000), rng)
            for t in j.tasks
        ]
    )
    assert f[:, 1].mean() == pytest.approx(0.20, abs=0.02)
    assert f[:, 2].mean() == pytest.approx(0.20, abs=0.02)
    assert f[:, 3].mean() == pytest.approx(0.11, abs=0.02)


def test_sampled_footprints():
    cfg = Config(num_jobs=2000)
    tasks = [
        t for j in generate_workload(cfg, np.random.default_rng(3)) for t in j.tasks
    ]
    shared = np.array([t.footprint_shared[0] for t in tasks])
    slot = np.array([t.footprint_slot[0] for t in tasks])
    assert np.median(shared) == pytest.approx(0.25, rel=0.1)
    assert np.median(slot) == pytest.approx(0.02, rel=0.1)
    assert np.std(np.log(shared)) == pytest.approx(1.0, rel=0.1)


def test_workload_modes():
    batch = generate_workload(Config(), np.random.default_rng(0))
    assert all(j.time_inserted == 0 for j in batch)
    stream = generate_workload(
        Config(workload_mode="streaming"), np.random.default_rng(0)
    )
    times = [j.time_inserted for j in stream]
    assert times == sorted(times) and times[0] > 0
    # Mean gap 1 / λ_a = 10
    assert np.mean(np.diff(times)) == pytest.approx(10, rel=0.3)


def test_workload_is_reproducible():
    a = generate_workload(Config(), np.random.default_rng(7))
    b = generate_workload(Config(), np.random.default_rng(7))
    for ja, jb in zip(a, b):
        assert ja.edges == jb.edges
        assert [t.work for t in ja.tasks] == [t.work for t in jb.tasks]
