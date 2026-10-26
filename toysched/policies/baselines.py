"""Baseline policies. Each supplies allocate and refill directly, in slot counts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..simulator import fifo_key
from .base import Allocation, OrderPolicy, Policy, shares_to_counts

if TYPE_CHECKING:
    from ..simulator import Simulator, Task


class FIFOPolicy(OrderPolicy):
    """Job's time_inserted, job id, time eligible, task id. Like DuckDB's default."""

    name = "fifo"

    def key(self, sim: Simulator) -> Callable[[Task], tuple]:
        return fifo_key


class SJFPolicy(OrderPolicy):
    """Smallest total job work at insertion first, ties in FIFO order."""

    name = "sjf"

    def key(self, sim: Simulator) -> Callable[[Task], tuple]:
        return lambda t: (t.job.total_work, *fifo_key(t))


class SRPTPolicy(OrderPolicy):
    """Smallest remaining job work first, re-sorted at every call."""

    name = "srpt"

    def key(self, sim: Simulator) -> Callable[[Task], tuple]:
        remaining = {job: job.remaining_work for job in sim.active_jobs}
        return lambda t: (remaining[t.job], *fifo_key(t))


class EqualSharePolicy(Policy):
    """Splits slots evenly over eligible tasks, capped at k*."""

    name = "equal_share"

    def allocate(self, sim: Simulator) -> Allocation:
        tasks = sim.eligible_tasks()
        counts = shares_to_counts(
            [1.0] * len(tasks), [t.max_parallelism for t in tasks], sim.S
        )
        return dict(zip(tasks, counts))

    def refill(self, sim: Simulator) -> Allocation:
        tasks = [t for t in sim.eligible_tasks() if t.slots < t.max_parallelism]
        counts = shares_to_counts(
            [1.0] * len(tasks),
            [t.max_parallelism - t.slots for t in tasks],
            sim.free_slots,
        )
        return dict(zip(tasks, counts))


BASELINES: dict[str, type[Policy]] = {
    p.name: p for p in (FIFOPolicy, EqualSharePolicy, SJFPolicy, SRPTPolicy)
}
