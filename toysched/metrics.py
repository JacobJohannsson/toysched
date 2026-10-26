"""Episode metrics computed from a finished simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulator import Simulator


def slot_time_fractions(slot_time: dict[str, float]) -> dict[str, float]:
    """Fraction of all slot-time (idle included) by the rule that assigned it."""
    total = sum(slot_time.values())
    if total <= 0:
        return {}
    return {src: v / total for src, v in sorted(slot_time.items())}


def episode_metrics(sim: Simulator) -> dict:
    latencies = np.array([j.latency for j in sim.jobs if j.finished], dtype=float)
    start = min((j.time_inserted for j in sim.jobs), default=0.0)
    end = sim.t
    makespan = end - start
    changes = [d["slot_changes"] for d in sim.decisions]
    gaps = [d["gap_l1"] for d in sim.decisions if "gap_l1" in d]

    metrics = {
        "num_jobs": len(sim.jobs),
        "num_finished": sim.num_finished_jobs,
        "makespan": makespan,
        "mean_latency": float(latencies.mean()) if len(latencies) else None,
        "p90_latency": float(np.percentile(latencies, 90)) if len(latencies) else None,
        "total_latency": float(latencies.sum()),
        "q_integral": sim.q_integral,
        "slot_utilization": (
            sim.busy_slot_time / (sim.S * makespan) if makespan > 0 else 0.0
        ),
        "oversubscribed_fraction": {
            name: (float(t) / end if end > 0 else 0.0)
            for name, t in zip(sim.cfg.capacity_resources, sim.oversubscribed_time)
        },
        "footprint_overflow_fraction": {
            name: (float(t) / end if end > 0 else 0.0)
            for name, t in zip(sim.cfg.footprint_resources, sim.overflow_time)
        },
        "mean_hit_fraction": {
            name: (float(h) / end if end > 0 else 1.0)
            for name, h in zip(sim.cfg.footprint_resources, sim.hit_fraction_integral)
        },
        "num_decisions": len(sim.decisions),
        "mean_slot_changes_per_decision": float(np.mean(changes)) if changes else 0.0,
        "slot_time_fraction": slot_time_fractions(sim.slot_time_by_source),
        "contention_failures": sim.contention_failures,
    }
    if gaps:
        metrics["mean_gap_l1"] = float(np.mean(gaps))
    return metrics
