"""Adapter that turns the RL policy's share vector into allocate and refill rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..simulator import fifo_key
from .base import Allocation, Policy, shares_to_counts

if TYPE_CHECKING:
    from ..simulator import Simulator, Task


class RLAdapter(Policy):
    """Call `set_shares` before each decision; the simulator then calls allocate.

    Only tasks of the `max_visible_jobs` oldest jobs can get slots, since those are
    the only ones the policy sees.
    """

    name = "rl"

    def __init__(self, max_visible_jobs: int | None = None):
        self.max_visible_jobs = max_visible_jobs
        self._pending: dict[Task, float] = {}
        self._shares: dict[Task, float] = {}  # p from the latest decision
        self._info: dict = {}

    def reset(self, sim: Simulator) -> None:
        self._pending, self._shares, self._info = {}, {}, {}

    def set_shares(self, shares: dict[Task, float]) -> None:
        """Share vector p for the next decision, over eligible tasks."""
        self._pending = dict(shares)

    def allocate(self, sim: Simulator) -> Allocation:
        p = {t: s for t, s in self._pending.items() if t in sim.eligible and s >= 0}
        self._pending = {}
        total = sum(p.values())
        self._shares = {t: s / total for t, s in p.items()} if total > 0 else {}
        if not self._shares:
            self._info = {}
            return {}

        tasks = list(self._shares)
        shares = [self._shares[t] for t in tasks]
        counts = shares_to_counts(shares, [t.max_parallelism for t in tasks], sim.S)
        self._info = {"gap_l1": sum(abs(s - n / sim.S) for s, n in zip(shares, counts))}
        return dict(zip(tasks, counts))

    def refill_tagged(self, sim: Simulator) -> dict[str, Allocation]:
        free = sim.free_slots
        grants: dict[Task, int] = {}

        # Step 1: top up tasks from the last decision that are below their target
        remaining = [t for t in self._shares if not t.finished]
        total = sum(self._shares[t] for t in remaining)
        if remaining and total > 0:
            shares = [self._shares[t] / total for t in remaining]
            targets = shares_to_counts(
                shares, [t.max_parallelism for t in remaining], sim.S
            )
            shortfall = {t: n - t.slots for t, n in zip(remaining, targets)}
            share_of = dict(zip(remaining, shares))
            while free > 0:
                best = max(
                    (t for t in remaining if shortfall[t] > 0),
                    key=lambda t: (shortfall[t], share_of[t]),
                    default=None,
                )
                if best is None:
                    break
                grants[best] = grants.get(best, 0) + 1
                shortfall[best] -= 1
                free -= 1

        # Step 2: FIFO fallback for whatever is still free
        fallback: dict[Task, int] = {}
        if free > 0:
            visible = {j for j in sim.visible_jobs(self.max_visible_jobs)}
            for task in sorted(sim.eligible, key=fifo_key):
                if free == 0:
                    break
                if task.job not in visible:
                    continue
                n = min(task.max_parallelism - task.slots - grants.get(task, 0), free)
                if n > 0:
                    fallback[task] = n
                    free -= n
        return {"refill": grants, "fallback": fallback}

    def refill(self, sim: Simulator) -> Allocation:
        merged: dict[Task, int] = {}
        for counts in self.refill_tagged(sim).values():
            for t, n in counts.items():
                merged[t] = merged.get(t, 0) + n
        return merged

    def decision_info(self) -> dict:
        return dict(self._info)
