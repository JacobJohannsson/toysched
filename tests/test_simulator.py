import heapq

import numpy as np
import pytest

from toysched.config import Config
from toysched.metrics import episode_metrics
from toysched.policies import BASELINES
from toysched.simulator import Simulator
from toysched.workload import generate_workload

SANITY = Config(
    height_min=1,
    height_max=1,
    serial_fraction_median=0.0,
    peak_median_frac=None,
    capacity_resources=(),
    capacity_limits=(),
    rho_median=(),
    footprint_resources=(),
    footprint_limits=(),
    footprint_overflow=(),
    miss_penalty=(),
    footprint_shared_median=(),
    footprint_slot_median=(),
)


def _run(cfg, name, seed=0):
    workload = generate_workload(cfg, np.random.default_rng(seed))
    sim = Simulator(cfg, workload, BASELINES[name]()).run()
    return sim, episode_metrics(sim)


@pytest.mark.parametrize("mode", ["batch", "streaming"])
@pytest.mark.parametrize("name", list(BASELINES))
def test_episode_invariants(mode, name):
    cfg = Config(num_jobs=50, workload_mode=mode)
    sim, m = _run(cfg, name)
    assert m["num_finished"] == 50
    assert m["contention_failures"] == 0
    # ∫ Q dt is the total latency
    assert m["q_integral"] == pytest.approx(m["total_latency"])
    # No slot is faster than 1, so the work bounds the busy slot-time from below
    total_work = sum(j.total_work for j in sim.jobs)
    assert sim.busy_slot_time >= total_work - 1e-6
    assert 0 < m["slot_utilization"] <= 1 + 1e-9
    assert sum(m["slot_time_fraction"].values()) == pytest.approx(1)
    for job in sim.jobs:
        for t in job.tasks:
            assert t.finished and t.remaining == 0
            # A task starts only after all its children finish
            for c in t.spec.children:
                assert job.tasks[c].finish_time <= t.eligible_time + 1e-9
        assert job.completion_time == job.tasks[0].finish_time
        assert job.completion_time >= job.time_inserted


def test_runs_are_deterministic():
    cfg = Config(num_jobs=30)
    _, a = _run(cfg, "srpt", seed=3)
    _, b = _run(cfg, "srpt", seed=3)
    assert a == b


def test_decisions_on_fixed_grid():
    cfg = Config(num_jobs=30, decision_interval=2.5)
    sim, _ = _run(cfg, "fifo")
    times = [d["time"] for d in sim.decisions]
    np.testing.assert_allclose(times, 2.5 * np.arange(len(times)))


def test_intervals_cover_the_episode():
    cfg = Config(num_jobs=30, workload_mode="streaming")
    sim, m = _run(cfg, "fifo")
    assert sum(iv["q_integral"] for iv in sim.intervals) == pytest.approx(
        m["q_integral"]
    )
    for a, b in zip(sim.intervals, sim.intervals[1:]):
        assert a["end"] == pytest.approx(b["start"])


def test_single_task_runs_for_its_work_time_without_contention():
    cfg = SANITY.with_overrides(
        num_jobs=1, serial_fraction_median=0.05, peak_median_frac=0.5
    )
    sim, m = _run(cfg, "fifo")
    task = sim.jobs[0].tasks[0]
    k = task.max_parallelism
    expected = task.spec.work / (k * task.spec.speedup[k - 1])
    assert m["makespan"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "overrides",
    [
        # Capacity resources only
        dict(
            footprint_resources=(),
            footprint_limits=(),
            footprint_overflow=(),
            miss_penalty=(),
            footprint_shared_median=(),
            footprint_slot_median=(),
            dirichlet_light=(5.0, 4.0, 0.5),
            dirichlet_heavy=(2.0, 2.0, 6.0),
        ),
        # A footprint resource whose misses only cost latency
        dict(
            capacity_resources=(),
            capacity_limits=(),
            rho_median=(),
            footprint_overflow=(None,),
            dirichlet_light=(5.0, 4.0),
            dirichlet_heavy=(2.0, 2.0),
        ),
    ],
    ids=["capacity-only", "footprint-only"],
)
def test_resource_combinations(overrides):
    cfg = Config(num_jobs=30, **overrides)
    sim, m = _run(cfg, "equal_share")
    assert m["num_finished"] == 30
    assert m["q_integral"] == pytest.approx(m["total_latency"])


def test_footprint_logging():
    sim, m = _run(Config(num_jobs=50), "equal_share")
    h = m["mean_hit_fraction"]["cache"]
    assert 0 < h < 1  # equal share mixes tasks, so the cache overflows at times
    assert 0 < m["footprint_overflow_fraction"]["cache"] < 1


def test_cache_overflow_costs_mixing_more_than_concentrating():
    """Shrinking the cache from unlimited to the default narrows equal share's lead
    over SRPT, since equal share runs more distinct tasks at once."""
    base = Config(num_jobs=60, workload_mode="streaming")
    unlimited = base.with_overrides(footprint_limits=(1e9,))
    assert _run(unlimited, "equal_share")[1]["mean_hit_fraction"]["cache"] == 1

    def lead(cfg):
        srpt = np.mean([_run(cfg, "srpt", s)[1]["mean_latency"] for s in range(3)])
        eq = np.mean([_run(cfg, "equal_share", s)[1]["mean_latency"] for s in range(3)])
        return srpt - eq

    assert lead(base) < lead(unlimited)


# ---------------------------------------------------------------- SRPT sanity check


def test_srpt_sanity_batch_matches_single_server_optimum():
    """One server of speed S: SRPT completes jobs in order of size."""
    cfg = SANITY.with_overrides(num_jobs=100)
    for seed in range(3):
        sim, m = _run(cfg, "srpt", seed)
        work = np.sort([j.total_work for j in sim.jobs])
        optimum = np.mean(np.cumsum(work) / cfg.num_slots)
        assert m["mean_latency"] == pytest.approx(optimum)
        for name in BASELINES:
            assert _run(cfg, name, seed)[1]["mean_latency"] >= optimum - 1e-9


def _preemptive_srpt(arrivals, sizes, speed):
    """Exact mean latency of preemptive SRPT on one server."""
    t, i, n = 0.0, 0, len(sizes)
    heap, total = [], 0.0
    order = np.argsort(arrivals, kind="stable")
    while i < n or heap:
        if not heap:
            t = max(t, arrivals[order[i]])
        while i < n and arrivals[order[i]] <= t:
            j = order[i]
            heapq.heappush(heap, [sizes[j], j])
            i += 1
        rem, j = heap[0]
        next_arrival = arrivals[order[i]] if i < n else np.inf
        finish = t + rem / speed
        if finish <= next_arrival:
            heapq.heappop(heap)
            t = finish
            total += t - arrivals[j]
        else:
            heap[0][0] -= (next_arrival - t) * speed
            heapq.heapify(heap)
            t = next_arrival
    return total / n


def test_srpt_sanity_streaming_with_small_interval():
    cfg = SANITY.with_overrides(
        num_jobs=200,
        workload_mode="streaming",
        arrival_rate=0.6,
        decision_interval=0.01,
    )
    sim, m = _run(cfg, "srpt", seed=0)
    arrivals = np.array([j.time_inserted for j in sim.jobs])
    sizes = np.array([j.total_work for j in sim.jobs])
    exact = _preemptive_srpt(arrivals, sizes, cfg.num_slots)
    assert m["mean_latency"] == pytest.approx(exact, rel=0.01)
    assert m["mean_latency"] >= exact - 1e-9
