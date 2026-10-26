import numpy as np
import pytest

from toysched.config import Config
from toysched.policies import (
    BASELINES,
    FIFOPolicy,
    RLAdapter,
    largest_remainder,
    shares_to_counts,
)
from toysched.simulator import Simulator, fifo_key
from toysched.workload import generate_workload


def test_largest_remainder():
    assert largest_remainder([0.5, 0.3, 0.2], 8) == [4, 2, 2]
    assert largest_remainder([1, 1, 1], 8) == [3, 3, 2]
    assert sum(largest_remainder([0.1] * 7, 8)) == 8
    assert largest_remainder([], 8) == []


def test_shares_to_counts_redistributes_capped_slots():
    # 0.75 * 8 = 6 slots for task 0, but its cap is 2. The excess goes to the others
    # in proportion to their shares (0.15 : 0.10).
    counts = shares_to_counts([0.75, 0.15, 0.10], [2, 8, 8], 8)
    assert counts[0] == 2 and sum(counts) == 8
    assert counts[1] >= counts[2]


def test_shares_to_counts_leaves_slots_when_all_capped():
    assert shares_to_counts([0.5, 0.5], [1, 2], 8) == [1, 2]


def test_shares_to_counts_ignores_zero_shares_when_redistributing():
    assert shares_to_counts([1.0, 0.0], [3, 8], 8) == [3, 0]


def _sim(policy, cfg=None, seed=0):
    cfg = cfg or Config(num_jobs=20)
    return Simulator(cfg, generate_workload(cfg, np.random.default_rng(seed)), policy)


def test_fifo_allocates_in_order():
    sim = _sim(FIFOPolicy())
    sim.advance_to_decision()
    alloc = sim.policy.allocate(sim)
    order = sorted(sim.eligible, key=fifo_key)
    given = [t for t in order if t in alloc]
    assert given == order[: len(given)]
    assert sum(alloc.values()) <= sim.S
    # Every task but the last gets its full k*
    for t in given[:-1]:
        assert alloc[t] == t.max_parallelism


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_complete_and_are_work_conserving(name):
    sim = _sim(BASELINES[name]())
    while sim.advance_to_decision():
        sim.decide()
        # After every decision and refill no slot is idle while some eligible task
        # could still take one
        assert not sim.can_refill()
    assert sim.done


def test_rl_adapter_follows_shares_and_logs_gap():
    adapter = RLAdapter()
    sim = _sim(adapter)
    sim.advance_to_decision()
    tasks = sim.eligible_tasks()
    shares = {t: 1.0 if i == 0 else 0.0 for i, t in enumerate(tasks)}
    adapter.set_shares(shares)
    sim.decide()
    first = tasks[0]
    assert first.slots == first.max_parallelism
    gap = sim.decisions[-1]["gap_l1"]
    assert gap == pytest.approx(1 - first.max_parallelism / sim.S)
    # Slots the shares could not use went to the FIFO fallback
    assert "fallback" in sim.slot_source or first.max_parallelism == sim.S


def test_rl_adapter_refill_tops_up_to_target():
    """A task from the last decision that lost a slot gets it back from refill step 1,
    and the slot is tagged as refill, not fallback."""
    adapter = RLAdapter()
    sim = _sim(adapter, Config(num_jobs=20, decision_interval=1000.0))
    sim.advance_to_decision()
    task = next(t for t in sim.eligible_tasks() if t.max_parallelism >= 2)
    adapter.set_shares({task: 1.0})
    sim.decide()
    assert task.slots == task.max_parallelism  # its target

    i = sim.slots.index(task)
    sim._set_assignment(
        [None if j == i else t for j, t in enumerate(sim.slots)],
        [None if j == i else s for j, s in enumerate(sim.slot_source)],
    )
    grants = adapter.refill_tagged(sim)
    assert grants["refill"] == {task: 1}
    assert task not in grants["fallback"]
