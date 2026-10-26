"""Event-driven simulator: runtime state, the event loop, and logging.

The loop jumps from event to event. Between events every running task progresses at a
constant rate, so the next event time is exact. At one instant, completions are
handled first, then arrivals, then the decision.

A decision is split into two calls so the Gymnasium environment can insert its action
between them: `advance_to_decision` runs until a decision is due, and `decide` applies
the policy's allocation. `run` does both in a loop, for the baselines.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from .config import Config
from .contention import apply_footprints, solve_contention
from .placement import place
from .workload import JobSpec, TaskSpec

if TYPE_CHECKING:
    from .policies.base import Policy

ALLOCATE = "allocate"


class Status(IntEnum):
    WAITING = 0
    ELIGIBLE = 1
    RUNNING = 2
    FINISHED = 3


class PolicyError(ValueError):
    """A policy returned an allocation or refill the simulator cannot apply."""


@dataclass(eq=False)
class Task:
    spec: TaskSpec
    job: Job
    remaining: float
    children_left: int
    eligible_time: float | None = None
    finish_time: float | None = None
    finished: bool = False
    slots: int = 0  # k_j
    speed: float = 0.0  # v per slot, after contention

    @property
    def id(self) -> int:
        return self.spec.id

    @property
    def max_parallelism(self) -> int:
        return self.spec.max_parallelism

    @property
    def rate(self) -> float:  # R_j
        return self.slots * self.speed

    @property
    def status(self) -> Status:
        if self.finished:
            return Status.FINISHED
        if self.eligible_time is None:
            return Status.WAITING
        return Status.RUNNING if self.slots > 0 else Status.ELIGIBLE

    def __repr__(self) -> str:
        return f"Task(job={self.job.id}, id={self.id}, status={self.status.name})"


@dataclass(eq=False)
class Job:
    spec: JobSpec
    tasks: list[Task] = field(default_factory=list)
    inserted: bool = False
    finished: bool = False
    completion_time: float | None = None
    tasks_finished: int = 0

    @property
    def id(self) -> int:
        return self.spec.id

    @property
    def time_inserted(self) -> float:
        return self.spec.time_inserted

    @property
    def total_work(self) -> float:
        return self.spec.total_work

    @property
    def remaining_work(self) -> float:
        return sum(t.remaining for t in self.tasks)

    @property
    def latency(self) -> float | None:
        if self.completion_time is None:
            return None
        return self.completion_time - self.time_inserted

    def __repr__(self) -> str:
        return f"Job(id={self.id}, finished={self.finished})"


def fifo_key(task: Task) -> tuple:
    """FIFO order: job's time_inserted, job id, time eligible, task id."""
    return (task.job.time_inserted, task.job.id, task.eligible_time, task.id)


class Simulator:
    def __init__(self, cfg: Config, workload: list[JobSpec], policy: Policy):
        self.cfg = cfg
        self.policy = policy
        self.S = cfg.num_slots
        self.capacities = np.asarray(cfg.capacity_limits, dtype=float)
        self.footprint_limits = np.asarray(cfg.footprint_limits, dtype=float)
        self.miss_penalty = np.asarray(cfg.miss_penalty, dtype=float)
        self.overflow = cfg.overflow_index

        self.jobs: list[Job] = []
        for spec in sorted(workload, key=lambda j: (j.time_inserted, j.id)):
            job = Job(spec)
            job.tasks = [
                Task(t, job, remaining=t.work, children_left=len(t.children))
                for t in spec.tasks
            ]
            self.jobs.append(job)

        self.t = 0.0
        self.decision_index = 0
        self.t_dec = 0.0
        self.decision_pending = False
        self._next_arrival = 0
        self.num_finished_jobs = 0
        self.completed_jobs: list[Job] = []  # in completion order

        self.slots: list[Task | None] = [None] * self.S
        self.slot_source: list[str | None] = [None] * self.S
        self.active_jobs: list[Job] = []  # in the system, in FIFO order
        self.eligible: dict[Task, None] = {}  # ordered set
        N, K = cfg.num_capacity_resources, cfg.num_footprint_resources
        self.rate_factor = np.ones(N)  # g_r
        self.load = np.zeros(N)  # U_r
        self.footprint_load = np.zeros(K)  # V_c
        self.hit_fraction = np.ones(K)  # h_c
        self._dirty = False

        # Logs and integrals
        self.q_integral = 0.0  # ∫ Q dt over the whole episode
        self.busy_slot_time = 0.0
        self.oversubscribed_time = np.zeros(N)
        self.overflow_time = np.zeros(K)
        self.hit_fraction_integral = np.zeros(K)
        self.slot_time_by_source: dict[str, float] = defaultdict(float)
        self.decisions: list[dict] = []
        self.intervals: list[dict] = []
        self._interval: dict | None = None
        self.contention_failures = 0
        self.num_contention_solves = 0

        self.policy.reset(self)

    # ------------------------------------------------------------------ queries

    @property
    def done(self) -> bool:
        return self.num_finished_jobs == len(self.jobs)

    @property
    def free_slots(self) -> int:
        return sum(1 for s in self.slots if s is None)

    @property
    def num_jobs_in_system(self) -> int:  # Q
        return len(self.active_jobs)

    def eligible_tasks(self) -> list[Task]:
        """Eligible tasks (running included), in FIFO order."""
        return sorted(self.eligible, key=fifo_key)

    def visible_jobs(self, limit: int | None = None) -> list[Job]:
        """The `limit` oldest jobs in the system."""
        return self.active_jobs if limit is None else self.active_jobs[:limit]

    def can_refill(self) -> bool:
        return self.free_slots > 0 and any(
            t.slots < t.max_parallelism for t in self.eligible
        )

    # ------------------------------------------------------------------ main loop

    def run(self) -> Simulator:
        """Run to the end of the episode, calling the policy at every decision."""
        while self.advance_to_decision():
            self.decide()
        return self

    def advance_to_decision(self) -> bool:
        """Simulate until a decision is due. Returns False once the episode is over."""
        if self.decision_pending:
            raise RuntimeError("A decision is pending; call decide() first")
        tol = self.cfg.time_tol
        while not self.done:
            running = [t for t in self.eligible if t.slots > 0]
            finish = {t: self.t + t.remaining / t.rate for t in running}
            t_next = self.t_dec
            if finish:
                t_next = min(t_next, min(finish.values()))
            if self._next_arrival < len(self.jobs):
                t_next = min(t_next, self.jobs[self._next_arrival].time_inserted)
            t_next = max(t_next, self.t)

            self._advance(t_next)
            for task in sorted(
                (t for t, ft in finish.items() if ft <= t_next + tol), key=fifo_key
            ):
                self._finish_task(task)
            while (
                self._next_arrival < len(self.jobs)
                and self.jobs[self._next_arrival].time_inserted <= self.t + tol
            ):
                self._insert_job(self.jobs[self._next_arrival])
                self._next_arrival += 1

            if self.done:
                break
            if self.t_dec <= self.t + tol:
                self._close_interval()
                self.decision_pending = True
                return True
            self._refill_if_needed()
            self._update_contention_if_dirty()

        self._close_interval()
        return False

    def decide(self) -> None:
        """Apply the policy's allocation at the pending decision."""
        if not self.decision_pending:
            raise RuntimeError("No decision is pending")
        alloc = {t: int(n) for t, n in self.policy.allocate(self).items() if n > 0}
        self._validate_allocation(alloc)
        new, changes = place(self.slots, alloc)
        self._set_assignment(new, [ALLOCATE if t is not None else None for t in new])

        self.decisions.append(
            {
                "index": self.decision_index,
                "time": self.t,
                "slot_changes": changes,
                "num_eligible": len(self.eligible),
                "num_jobs": self.num_jobs_in_system,
                **self.policy.decision_info(),
            }
        )
        self.decision_index += 1
        self.t_dec = self.decision_index * self.cfg.decision_interval
        self.decision_pending = False
        self._interval = {
            "start": self.t,
            "q_integral": 0.0,
            "slot_time": defaultdict(float),
        }
        self._refill_if_needed()
        self._update_contention_if_dirty()

    # ------------------------------------------------------------------ internals

    def _advance(self, t_next: float) -> None:
        dt = t_next - self.t
        if dt > 0:
            for task in self.eligible:
                if task.slots > 0:
                    task.remaining = max(0.0, task.remaining - task.rate * dt)
            q = self.num_jobs_in_system
            self.q_integral += q * dt
            self.oversubscribed_time += (
                self.rate_factor < 1 - self.cfg.oversubscribed_tol
            ) * dt
            self.overflow_time += (
                self.hit_fraction < 1 - self.cfg.oversubscribed_tol
            ) * dt
            self.hit_fraction_integral += self.hit_fraction * dt
            busy = self.S - self.free_slots
            self.busy_slot_time += busy * dt
            for src in self.slot_source:
                self.slot_time_by_source[src or "idle"] += dt
            if self._interval is not None:
                self._interval["q_integral"] += q * dt
                for src in self.slot_source:
                    self._interval["slot_time"][src or "idle"] += dt
        self.t = t_next

    def _close_interval(self) -> None:
        if self._interval is None:
            return
        iv = self._interval
        iv["end"] = self.t
        iv["slot_time"] = dict(iv["slot_time"])
        self.intervals.append(iv)
        self._interval = None

    def _make_eligible(self, task: Task) -> None:
        task.eligible_time = self.t
        self.eligible[task] = None

    def _insert_job(self, job: Job) -> None:
        job.inserted = True
        self.active_jobs.append(job)
        for task in job.tasks:
            if task.children_left == 0:
                self._make_eligible(task)

    def _finish_task(self, task: Task) -> None:
        task.remaining = 0.0
        task.finished = True
        task.finish_time = self.t
        task.speed = 0.0
        del self.eligible[task]
        new = [None if s is task else s for s in self.slots]
        self._set_assignment(
            new,
            [src if s is not None else None for s, src in zip(new, self.slot_source)],
        )
        job = task.job
        job.tasks_finished += 1
        for p in task.spec.parents:
            parent = job.tasks[p]
            parent.children_left -= 1
            if parent.children_left == 0:
                self._make_eligible(parent)
        if not task.spec.parents:  # the root
            job.finished = True
            job.completion_time = self.t
            self.active_jobs.remove(job)
            self.num_finished_jobs += 1
            self.completed_jobs.append(job)

    def _set_assignment(
        self, slots: list[Task | None], sources: list[str | None]
    ) -> None:
        if slots == self.slots and sources == self.slot_source:
            return
        touched = {t for t in self.slots if t is not None} | {
            t for t in slots if t is not None
        }
        self.slots = slots
        self.slot_source = sources
        for t in touched:
            t.slots = 0
        for t in slots:
            if t is not None:
                t.slots += 1
        self._dirty = True

    def _validate_allocation(self, alloc: dict[Task, int]) -> None:
        if sum(alloc.values()) > self.S:
            raise PolicyError(f"Allocation uses {sum(alloc.values())} > S slots")
        for task, n in alloc.items():
            if task not in self.eligible:
                raise PolicyError(f"{task} is not eligible")
            if n > task.max_parallelism:
                raise PolicyError(
                    f"{task} given {n} > k* = {task.max_parallelism} slots"
                )

    def _refill_if_needed(self) -> None:
        if not self.can_refill():
            return
        grants = self.policy.refill_tagged(self)
        slots = list(self.slots)
        sources = list(self.slot_source)
        free = iter([i for i, s in enumerate(slots) if s is None])
        added: dict[Task, int] = defaultdict(int)
        for source, counts in grants.items():
            for task, n in counts.items():
                if n <= 0:
                    continue
                if task not in self.eligible:
                    raise PolicyError(f"Refill gave slots to {task}, not eligible")
                added[task] += n
                if task.slots + added[task] > task.max_parallelism:
                    raise PolicyError(f"Refill pushed {task} past k*")
                for _ in range(n):
                    i = next(free, None)
                    if i is None:
                        raise PolicyError("Refill gave out more than the free slots")
                    slots[i] = task
                    sources[i] = source
        self._set_assignment(slots, sources)

    def _update_contention_if_dirty(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        running = [t for t in self.eligible if t.slots > 0]
        for t in self.eligible:
            if t.slots == 0:
                t.speed = 0.0
        N, K = self.cfg.num_capacity_resources, self.cfg.num_footprint_resources
        if not running:
            self.rate_factor = np.ones(N)
            self.load = np.zeros(N)
            self.footprint_load = np.zeros(K)
            self.hit_fraction = np.ones(K)
            return
        m = len(running)
        slot_counts = np.array([t.slots for t in running], dtype=float)
        fp = apply_footprints(
            breakdown=np.array([t.spec.breakdown for t in running]),
            rho=np.array([t.spec.rho for t in running]).reshape(m, N),
            footprint_shared=np.array(
                [t.spec.footprint_shared for t in running]
            ).reshape(m, K),
            footprint_slot=np.array([t.spec.footprint_slot for t in running]).reshape(
                m, K
            ),
            slot_counts=slot_counts,
            limits=self.footprint_limits,
            overflow=self.overflow,
            miss_penalty=self.miss_penalty,
        )
        self.footprint_load = fp.load
        self.hit_fraction = fp.hit_fraction
        res = solve_contention(
            unslowed=fp.unslowed,
            bound=fp.bound,
            usage=fp.usage,
            base_speed=np.array([t.spec.speedup[t.slots - 1] for t in running]),
            slot_counts=slot_counts,
            capacities=self.capacities,
            tol=self.cfg.contention_tol,
            max_iter=self.cfg.contention_max_iter,
        )
        self.num_contention_solves += 1
        if not res.converged:
            self.contention_failures += 1
        for task, v in zip(running, res.speed):
            task.speed = float(v)
        self.rate_factor = res.rate_factor
        self.load = res.load
