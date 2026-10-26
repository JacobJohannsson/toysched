"""Run baseline policies on the same seeded workloads and save the results.

Example:
    python -m toysched.run_baselines --seeds 0 1 2 --set workload_mode=streaming
    python -m toysched.run_baselines --config configs/srpt_sanity.yaml --policies srpt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .config import Config
from .metrics import episode_metrics
from .policies import BASELINES
from .simulator import Simulator
from .workload import generate_workload


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Override {pair!r} is not of the form key=value")
        overrides[key] = yaml.safe_load(value)
    return overrides


def run_policy(cfg: Config, seed: int, policy_name: str) -> dict:
    workload = generate_workload(cfg, np.random.default_rng(seed))
    sim = Simulator(cfg, workload, BASELINES[policy_name]()).run()
    return episode_metrics(sim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="YAML file of config overrides")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override single knobs, e.g. --set num_slots=16 decision_interval=1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--policies", nargs="+", default=list(BASELINES), choices=list(BASELINES)
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    cfg = (
        Config.from_yaml(args.config, **overrides)
        if args.config
        else Config.from_dict(overrides)
    )

    runs = []
    for seed in args.seeds:
        for name in args.policies:
            start = time.perf_counter()
            metrics = run_policy(cfg, seed, name)
            runs.append(
                {
                    "policy": name,
                    "seed": seed,
                    "wall_time_s": time.perf_counter() - start,
                    "metrics": metrics,
                }
            )

    out_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {"config": cfg.to_dict(), "seeds": args.seeds, "runs": runs}, f, indent=2
        )

    print(
        f"{'policy':<12} {'mean lat':>10} {'p90 lat':>10} {'makespan':>10} {'util':>6}"
    )
    for name in args.policies:
        ms = [r["metrics"] for r in runs if r["policy"] == name]
        print(
            f"{name:<12}"
            f" {np.mean([m['mean_latency'] for m in ms]):>10.2f}"
            f" {np.mean([m['p90_latency'] for m in ms]):>10.2f}"
            f" {np.mean([m['makespan'] for m in ms]):>10.2f}"
            f" {np.mean([m['slot_utilization'] for m in ms]):>6.3f}"
        )
    print(f"Means over seeds {args.seeds}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
