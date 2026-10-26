"""Placement: turn slot counts into a task on each slot, changing as few as possible."""

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def place(
    prev: Sequence[T | None], counts: Mapping[T, int]
) -> tuple[list[T | None], int]:
    """Apply new slot counts to the previous assignment.

    Each task keeps min(n_j, previous count) of the slots it holds, every other slot
    is released, and released or empty slots go to tasks that still need slots, in
    the order of `counts`, lowest slot index first.

    Returns the new assignment and the number of slot changes, the sum over tasks of
    max(0, n_j - previous count).
    """
    new: list[T | None] = [None] * len(prev)
    kept: dict[T, int] = {}
    for i, task in enumerate(prev):
        if task is None:
            continue
        if kept.get(task, 0) < counts.get(task, 0):
            new[i] = task
            kept[task] = kept.get(task, 0) + 1

    free = (i for i, task in enumerate(new) if task is None)
    changes = 0
    for task, n in counts.items():
        need = n - kept.get(task, 0)
        for _ in range(need):
            new[next(free)] = task
        changes += max(0, need)
    return new, changes
