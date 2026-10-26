"""Workload generation: job DAGs, task attributes, and arrival times.

Everything here is static. The simulator builds its runtime state from these specs,
so one generated workload can be replayed under every policy.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass(frozen=True, eq=False)
class TaskSpec:
    id: int  # index within the job; ids are assigned level by level, top first
    level: int  # 1 at the root
    breakdown: np.ndarray  # f_{j,0}, f_{j,c} for c = 1..K, f_{j,r} for r = 1..N
    rho: np.ndarray  # ρ_{j,r}, length N
    usage: np.ndarray  # a_{j,r} = f_{j,r} ρ_{j,r}, length N
    footprint_shared: np.ndarray  # w^sh_{j,c}, length K
    footprint_slot: np.ndarray  # w^sl_{j,c}, length K
    work: float  # W_j
    serial_fraction: float  # α_j
    coherency: float  # κ_j
    peak: float  # k^peak_j, where total speed peaks; inf when κ_j = 0
    speedup: np.ndarray  # s_j[k] at index k - 1, for k = 1..S
    max_parallelism: int  # k*_j
    children: tuple[int, ...]  # tasks this one depends on
    parents: tuple[int, ...]  # tasks that depend on this one


@dataclass(frozen=True, eq=False)
class JobSpec:
    id: int
    time_inserted: float
    height: int  # L
    tasks: tuple[TaskSpec, ...]
    edges: tuple[tuple[int, int], ...]  # (parent, child) pairs

    @property
    def total_work(self) -> float:
        return sum(t.work for t in self.tasks)

    @property
    def root(self) -> TaskSpec:
        return self.tasks[0]


def _round(x: float) -> int:
    return math.floor(x + 0.5)


def usl_speedup(alpha: float, kappa: float, num_slots: int) -> np.ndarray:
    """Per-slot speed s[k] = 1 / (1 + α(k - 1) + κk(k - 1)) for k = 1..num_slots,
    stored at index k - 1. Total speed k s[k] follows the Universal Scalability
    Law."""
    k = np.arange(1, num_slots + 1, dtype=float)
    return 1.0 / (1.0 + alpha * (k - 1) + kappa * k * (k - 1))


def coherency_for_peak(alpha: float, peak: float) -> float:
    """κ that puts the peak of total speed, √((1 - α) / κ), at `peak`."""
    return (1.0 - alpha) / peak**2 if np.isfinite(peak) else 0.0


def max_parallelism(speedup: np.ndarray) -> int:
    """The k maximizing k * s[k], smallest k on ties."""
    total = np.arange(1, len(speedup) + 1) * speedup
    return int(np.argmax(total)) + 1


def sample_height(cfg: Config, rng: np.random.Generator) -> int:
    L = _round(rng.normal(cfg.height_mean, cfg.height_std))
    return int(np.clip(L, cfg.height_min, cfg.height_max))


def sample_task_count(cfg: Config, L: int, rng: np.random.Generator) -> int:
    t_min, t_max = L, 2 ** (L - 1)
    if t_max <= t_min:
        return L
    span = t_max - t_min
    T = _round(
        rng.normal(
            t_min + cfg.task_count_mean_frac * span, cfg.task_count_std_frac * span
        )
    )
    return int(np.clip(T, t_min, t_max))


def sample_levels(
    cfg: Config, L: int, T: int, rng: np.random.Generator
) -> list[list[int]]:
    """Task ids per level, top first. Level l (1-based) holds at most 3^(l-1)."""
    counts = [1] * L
    caps = [3**i for i in range(L)]
    for _ in range(T - L):
        while True:
            x = rng.exponential(cfg.level_mean_frac * L)
            level = L - math.floor(x)
            if level >= 1 and counts[level - 1] < caps[level - 1]:
                counts[level - 1] += 1
                break
    levels, next_id = [], 0
    for c in counts:
        levels.append(list(range(next_id, next_id + c)))
        next_id += c
    return levels


def sample_edges(
    cfg: Config, levels: list[list[int]], rng: np.random.Generator
) -> list[tuple[int, int]]:
    """(parent, child) edges, following the reference pseudocode."""
    L = len(levels)
    edges: set[tuple[int, int]] = set()
    order: list[tuple[int, int]] = []  # insertion order, for reproducible output
    child_count = {t: 0 for lvl in levels for t in lvl}

    def add_edge(parent: int, child: int) -> None:
        if (parent, child) not in edges:
            edges.add((parent, child))
            order.append((parent, child))
            child_count[parent] += 1

    def choice(xs: list[int]) -> int:
        return xs[int(rng.integers(len(xs)))]

    # Pass 1: connectivity
    for i in range(1, L):
        for v in levels[i]:
            add_edge(choice(levels[i - 1]), v)
    for i in range(L - 1):
        lower = [t for lvl in levels[i + 1 :] for t in lvl]
        for u in levels[i]:
            if child_count[u] == 0:
                add_edge(u, choice(lower))

    # Pass 2: distance-decayed extra edges
    for x in range(L):
        for y in range(x + 1, L):
            prob = cfg.edge_prob / (y - x) ** cfg.edge_decay
            for u in levels[x]:
                for v in levels[y]:
                    if (u, v) not in edges and rng.random() < prob:
                        add_edge(u, v)
    return order


def _lognormal_or_zero(
    median: float, log_std: float, rng: np.random.Generator
) -> float:
    """Lognormal draw with the given median; a median of 0 always gives 0."""
    if median <= 0:
        return 0.0
    return float(np.exp(rng.normal(math.log(median), log_std)))


def sample_breakdown(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.num_capacity_resources + cfg.num_footprint_resources == 0:
        return np.ones(1)
    heavy = rng.random() < cfg.heavy_prob
    alpha = cfg.dirichlet_heavy if heavy else cfg.dirichlet_light
    return rng.dirichlet(alpha)


def sample_task(
    cfg: Config,
    task_id: int,
    level: int,
    children: tuple[int, ...],
    parents: tuple[int, ...],
    rng: np.random.Generator,
) -> TaskSpec:
    S = cfg.num_slots
    breakdown = sample_breakdown(cfg, rng)
    rho = np.minimum(
        1.0, np.exp(rng.normal(np.log(cfg.rho_median), cfg.rho_log_std))
    ).reshape(cfg.num_capacity_resources)
    usage = breakdown[1 + cfg.num_footprint_resources :] * rho
    footprint_shared = np.array(
        [
            _lognormal_or_zero(m, cfg.footprint_log_std, rng)
            for m in cfg.footprint_shared_median
        ]
    )
    footprint_slot = np.array(
        [
            _lognormal_or_zero(m, cfg.footprint_log_std, rng)
            for m in cfg.footprint_slot_median
        ]
    )
    work = float(np.exp(rng.normal(cfg.work_log_mean, cfg.work_log_std)))

    alpha = _lognormal_or_zero(
        cfg.serial_fraction_median, cfg.serial_fraction_log_std, rng
    )
    alpha = min(alpha, 1.0)
    if cfg.peak_median_frac is None:
        peak = math.inf
    else:
        peak = _lognormal_or_zero(cfg.peak_median_frac * S, cfg.peak_log_std, rng)
    kappa = coherency_for_peak(alpha, peak)
    speedup = usl_speedup(alpha, kappa, S)

    return TaskSpec(
        id=task_id,
        level=level,
        breakdown=breakdown,
        rho=rho,
        usage=usage,
        footprint_shared=footprint_shared,
        footprint_slot=footprint_slot,
        work=work,
        serial_fraction=alpha,
        coherency=kappa,
        peak=peak,
        speedup=speedup,
        max_parallelism=max_parallelism(speedup),
        children=children,
        parents=parents,
    )


def sample_job(
    cfg: Config, job_id: int, time_inserted: float, rng: np.random.Generator
) -> JobSpec:
    L = sample_height(cfg, rng)
    T = sample_task_count(cfg, L, rng)
    levels = sample_levels(cfg, L, T, rng)
    edges = sample_edges(cfg, levels, rng)

    level_of = {t: i + 1 for i, lvl in enumerate(levels) for t in lvl}
    children = {t: [] for t in range(T)}
    parents = {t: [] for t in range(T)}
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)

    tasks = tuple(
        sample_task(
            cfg,
            t,
            level_of[t],
            tuple(sorted(children[t])),
            tuple(sorted(parents[t])),
            rng,
        )
        for t in range(T)
    )
    return JobSpec(
        id=job_id,
        time_inserted=time_inserted,
        height=L,
        tasks=tasks,
        edges=tuple(edges),
    )


def sample_arrival_times(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.workload_mode == "batch":
        return np.zeros(cfg.num_jobs)
    gaps = rng.exponential(1.0 / cfg.arrival_rate, size=cfg.num_jobs)
    return np.cumsum(gaps)


def generate_workload(cfg: Config, rng: np.random.Generator) -> list[JobSpec]:
    """J jobs, with ids in arrival order."""
    times = sample_arrival_times(cfg, rng)
    return [sample_job(cfg, i, float(t), rng) for i, t in enumerate(times)]
