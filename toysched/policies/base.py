"""Policy interface and the rounding helpers shared by policies."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulator import Simulator, Task

Allocation = dict["Task", int]


class Policy:
    """Supplies allocate, called at each decision, and refill, called when slots are
    free between decisions. Both work in slot counts."""

    name = "policy"

    def reset(self, sim: Simulator) -> None:
        pass

    def allocate(self, sim: Simulator) -> Allocation:
        """Slot count n_j for eligible tasks, n_j <= k*_j, summing to at most S."""
        raise NotImplementedError

    def refill(self, sim: Simulator) -> Allocation:
        """Extra slots, out of the free ones, for eligible tasks below k*."""
        raise NotImplementedError

    def refill_tagged(self, sim: Simulator) -> dict[str, Allocation]:
        """Refill grants keyed by a source label, for slot-time logging."""
        return {"refill": self.refill(sim)}

    def decision_info(self) -> dict:
        """Extra fields to log with the latest decision."""
        return {}


def largest_remainder(weights: Sequence[float], total: int) -> list[int]:
    """Integer counts summing to `total`, proportional to `weights`.

    Each weight's share of `total` is rounded down, then the leftover units go to the
    largest fractional parts, ties to the larger weight, then the earlier index.
    """
    if total <= 0 or not weights:
        return [0] * len(weights)
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights must have positive sum")
    raw = [w / s * total for w in weights]
    counts = [math.floor(x) for x in raw]
    left = total - sum(counts)
    order = sorted(
        range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), -weights[i], i)
    )
    for i in order[:left]:
        counts[i] += 1
    return counts


def shares_to_counts(
    shares: Sequence[float], caps: Sequence[int], total: int
) -> list[int]:
    """Largest-remainder counts for `shares`, cut down to `caps`.

    Slots freed by the caps go to uncapped tasks with a positive share, in proportion
    to their shares, until no task is over its cap or no such task remains. Slots
    that cannot be placed are left out, so the counts may sum to less than `total`.
    """
    counts = largest_remainder(shares, total)
    capped = [False] * len(shares)
    while True:
        excess = 0
        for i, (c, cap) in enumerate(zip(counts, caps)):
            if c > cap:
                excess += c - cap
                counts[i] = cap
                capped[i] = True
        if excess == 0:
            break
        open_ = [i for i in range(len(shares)) if not capped[i] and shares[i] > 0]
        if not open_:
            break
        extra = largest_remainder([shares[i] for i in open_], excess)
        for i, e in zip(open_, extra):
            counts[i] += e
    return counts


class OrderPolicy(Policy):
    """Gives each task slots up to k* in a priority order, until slots run out."""

    def key(self, sim: Simulator) -> Callable[[Task], tuple]:
        raise NotImplementedError

    def allocate(self, sim: Simulator) -> Allocation:
        left, alloc = sim.S, {}
        for task in sorted(sim.eligible, key=self.key(sim)):
            if left == 0:
                break
            n = min(task.max_parallelism, left)
            alloc[task] = n
            left -= n
        return alloc

    def refill(self, sim: Simulator) -> Allocation:
        free, grants = sim.free_slots, {}
        for task in sorted(sim.eligible, key=self.key(sim)):
            if free == 0:
                break
            n = min(task.max_parallelism - task.slots, free)
            if n > 0:
                grants[task] = n
                free -= n
        return grants
