"""Every knob of the toy simulator, in one configuration object.

Field comments give the symbol each knob has in the requirements document.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    # Machine
    num_slots: int = 8  # S

    # Capacity resources, consumed at a rate; load adds once per slot
    capacity_resources: tuple[str, ...] = ("memory_bandwidth", "storage_io")  # N = len
    capacity_limits: tuple[float, ...] = (1.0, 1.0)  # C_r

    # Footprint resources, occupied; load adds once per task plus a per-slot part.
    # Each overflows into the capacity resource named in footprint_overflow, or None.
    footprint_resources: tuple[str, ...] = ("cache",)  # K = len
    footprint_limits: tuple[float, ...] = (1.0,)  # B_c
    footprint_overflow: tuple[str | None, ...] = ("memory_bandwidth",)  # o(c)
    miss_penalty: tuple[float, ...] = (3.0,)  # η_c
    footprint_shared_median: tuple[float, ...] = (0.25,)  # w̃^sh, fraction of B_c
    footprint_slot_median: tuple[float, ...] = (0.02,)  # w̃^sl, fraction of B_c
    footprint_log_std: float = 1.0  # σ_w

    # Workload
    num_jobs: int = 100  # J
    workload_mode: str = "batch"  # "batch" or "streaming"
    arrival_rate: float = 0.1  # λ_a, streaming mode only

    # Scheduling
    decision_interval: float = 2.5  # Δ
    max_visible_jobs: int = 20  # M

    # DAG shape
    height_mean: float = 3.0  # μ_L
    height_std: float = 1.0  # σ_L
    height_min: int = 1  # L_min
    height_max: int = 6  # L_max
    task_count_mean_frac: float = 0.5  # μ_T
    task_count_std_frac: float = 0.25  # σ_T
    level_mean_frac: float = 0.3  # λ_L
    edge_prob: float = 0.4  # p_e
    edge_decay: float = 1.5  # γ

    # Time breakdown and capacity usage. Dirichlet parameters have 1 + K + N entries,
    # ordered compute, footprint resources, capacity resources.
    heavy_prob: float = 0.1  # π
    dirichlet_light: tuple[float, ...] = (5.0, 2.0, 2.0, 0.5)  # β^light
    dirichlet_heavy: tuple[float, ...] = (2.0, 1.0, 1.0, 6.0)  # β^heavy
    rho_median: tuple[float, ...] = (0.3, 0.6)  # ρ_r, length N
    rho_log_std: float = 0.3  # σ_ρ

    # Work time
    work_log_mean: float = math.log(10.0)  # μ_W
    work_log_std: float = 0.75  # σ_W

    # Speedup (Universal Scalability Law)
    serial_fraction_median: float = 0.02  # median of α_j; 0 gives α_j = 0
    serial_fraction_log_std: float = 0.75  # σ_α, std of ln α_j
    peak_median_frac: float | None = 4.0  # median of k^peak_j / S; None gives κ_j = 0
    peak_log_std: float = 1.0  # σ_peak, std of ln k^peak_j

    # Numerics
    time_tol: float = 1e-9  # events closer than this are simultaneous
    contention_tol: float = 1e-10
    contention_max_iter: int = 10_000
    oversubscribed_tol: float = 1e-9  # r counts as oversubscribed when g_r < 1 - tol

    def __post_init__(self):
        # YAML gives lists; store tuples so a frozen config cannot be mutated in place
        for name in (
            "capacity_resources",
            "capacity_limits",
            "footprint_resources",
            "footprint_limits",
            "footprint_overflow",
            "miss_penalty",
            "footprint_shared_median",
            "footprint_slot_median",
            "dirichlet_light",
            "dirichlet_heavy",
            "rho_median",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate()

    @property
    def num_capacity_resources(self) -> int:  # N
        return len(self.capacity_resources)

    @property
    def num_footprint_resources(self) -> int:  # K
        return len(self.footprint_resources)

    @property
    def overflow_index(self) -> tuple[int | None, ...]:
        """o(c) as an index into the capacity resources, or None."""
        return tuple(
            None if name is None else self.capacity_resources.index(name)
            for name in self.footprint_overflow
        )

    @property
    def max_tasks_per_job(self) -> int:
        return 2 ** (self.height_max - 1)

    def validate(self) -> None:
        n, k = self.num_capacity_resources, self.num_footprint_resources
        if self.num_slots < 1:
            raise ValueError("num_slots must be at least 1")
        if len(self.capacity_limits) != n or len(self.rho_median) != n:
            raise ValueError(
                "capacity_limits and rho_median need one entry per capacity resource"
            )
        for name in (
            "footprint_limits",
            "footprint_overflow",
            "miss_penalty",
            "footprint_shared_median",
            "footprint_slot_median",
        ):
            if len(getattr(self, name)) != k:
                raise ValueError(f"{name} needs one entry per footprint resource")
        for name in self.footprint_overflow:
            if name is not None and name not in self.capacity_resources:
                raise ValueError(f"footprint_overflow names unknown resource {name!r}")
        if n + k > 0 and (
            len(self.dirichlet_light) != 1 + k + n
            or len(self.dirichlet_heavy) != 1 + k + n
        ):
            raise ValueError(
                "Dirichlet parameters need 1 + K + N entries "
                "(compute, footprint resources, capacity resources)"
            )
        if self.workload_mode not in ("batch", "streaming"):
            raise ValueError(f"Unknown workload_mode {self.workload_mode!r}")
        if not 1 <= self.height_min <= self.height_max:
            raise ValueError("Need 1 <= height_min <= height_max")
        if not 0 <= self.serial_fraction_median <= 1:
            raise ValueError("serial_fraction_median must lie in [0, 1]")
        if self.peak_median_frac is not None and self.peak_median_frac <= 0:
            raise ValueError("peak_median_frac must be positive or None")
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be positive")

    def with_overrides(self, **overrides: Any) -> "Config":
        return dataclasses.replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        d.update(overrides)
        return cls.from_dict(d)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
