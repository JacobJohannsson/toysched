# toy-scheduler

Toy simulator for practicing the RL scheduler before it runs against DuckDB. Jobs are
DAGs of tasks that compete for S slots and slow each other down on shared resources.
See `toy_scheduler_simulator_requirements.md` for the full design. This package is
Phase 1: the environment and the baselines.

## Setup

```bash
poetry install
poetry run pytest          # or: poetry run poe test
```

## Parameters

Every knob is a top-level field of `Config` in `toysched/config.py`, with its default
and the symbol it has in the requirements doc. `configs/default.yaml` lists all of them.
There are three ways to change one:

```bash
# 1. A YAML file with only the keys you want to change
poetry run python -m toysched.run_baselines --config configs/srpt_sanity.yaml
# 2. Single overrides on the command line (applied on top of --config, if given)
poetry run python -m toysched.run_baselines --set num_slots=16 peak_median_frac=2.0
```

```python
# 3. In Python
from toysched import Config
cfg = Config(serial_fraction_median=0.05, workload_mode="streaming")
cfg = cfg.with_overrides(decision_interval=1.0)
```

Resource knobs. Names are labels; the simulator treats every resource of a type alike.

| Knob | Meaning | Default |
| --- | --- | --- |
| `capacity_resources`, `capacity_limits` | Capacity resources (consumed at a rate, load per slot) and their limits C_r | memory_bandwidth, storage_io; 1, 1 |
| `footprint_resources`, `footprint_limits` | Footprint resources (occupied, load per task plus per slot) and their sizes B_c | cache; 1 |
| `footprint_overflow` | Capacity resource each footprint resource's misses go to, or `null` for latency only | memory_bandwidth |
| `miss_penalty` | η_c: how many times longer a missed access takes | 3 |
| `footprint_shared_median`, `footprint_slot_median`, `footprint_log_std` | Footprint held once per task, and once per slot (fractions of B_c) | 0.25, 0.02, 1.0 |
| `dirichlet_light`, `dirichlet_heavy` | Time breakdown, ordered compute, footprint resources, capacity resources | (5, 2, 2, 0.5), (2, 1, 1, 6) |

Speedup knobs (Universal Scalability Law, `s[k] = 1 / (1 + α(k-1) + κk(k-1))`):

| Knob | Meaning | Default |
| --- | --- | --- |
| `serial_fraction_median` | Median serial fraction α_j; 0 turns it off | 0.02 |
| `serial_fraction_log_std` | Std of ln α_j | 0.75 |
| `peak_median_frac` | Median slot count where total speed peaks, as a multiple of S; `null` turns coherency off (κ_j = 0) | 4.0 |
| `peak_log_std` | Std of ln of the peak | 1.0 |

## Layout

| Module | What it holds |
| --- | --- |
| `toysched/config.py` | `Config`, every knob from the Notation table, loadable from YAML |
| `toysched/workload.py` | DAG shape, task attributes, USL speedup curves, arrival times |
| `toysched/contention.py` | Footprint step (hit fractions, misses) and the fixed-point solve for g_r |
| `toysched/placement.py` | Minimum-change placement of slot counts onto slots |
| `toysched/simulator.py` | Runtime state and the event loop |
| `toysched/policies/` | `Policy` interface, FIFO, equal share, SJF, SRPT, and the RL adapter |
| `toysched/env.py` | Gymnasium environment (`ToyScheduler-v0`) wrapping simulator + RL adapter |
| `toysched/metrics.py` | Episode metrics |
| `toysched/run_baselines.py` | CLI to run baselines over seeds and save results |

## Running baselines

```bash
poetry run python -m toysched.run_baselines                       # defaults, seeds 0-4
poetry run python -m toysched.run_baselines --set workload_mode=streaming
poetry run python -m toysched.run_baselines --config configs/srpt_sanity.yaml
```

Each run writes `results/<timestamp>/config.yaml` and `results.json`, with the full
config, the seeds, and per-policy, per-seed metrics.

## Using the simulator directly

```python
import numpy as np
from toysched import BASELINES, Config, Simulator, episode_metrics, generate_workload

cfg = Config(num_jobs=50, workload_mode="streaming")
workload = generate_workload(cfg, np.random.default_rng(0))
sim = Simulator(cfg, workload, BASELINES["srpt"]()).run()
print(episode_metrics(sim))
```

A policy subclasses `toysched.policies.Policy` and implements `allocate(sim)` and
`refill(sim)`, each returning `{task: slot_count}`. The simulator validates both.

## Gymnasium environment

```python
from toysched import Config, ToySchedulerEnv

env = ToySchedulerEnv(Config())
obs, info = env.reset(seed=0)        # same workload as generate_workload(cfg, default_rng(0))
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- Observation: a dict of arrays padded to `M` jobs × `2^(L_max - 1)` tasks, with
  `job_mask`, `task_mask`, and `eligible_mask`.
- Action: shape `(M, max_tasks)`, entries in [0, 1]; only eligible positions count.
- Reward: Phase 2. `reward_fn` defaults to 0. Each step's `info` has `q_integral`,
  `elapsed`, `completed_jobs`, the decision log (with `gap_l1`), and the slot-time
  fractions by `allocate` / `refill` / `fallback` / `idle`. The reset info has the
  `q_integral` from before the first step.

## Interpretation choices

Places where the requirements leave room, and what the code does:

- The `M` visible jobs limit applies only to the RL environment. Tasks of invisible jobs
  never get slots under the RL adapter, including from the FIFO fallback. Baselines see
  every job.
- Decisions with no eligible visible task (an empty system in streaming mode) are
  applied automatically and are not env steps. Q is 0 over them.
- Refill step 1 gives free slots out one at a time to the task with the largest
  remaining shortfall, ties to the larger share.
- Order-based baselines break ties in their sort key with the FIFO key.
- "Oversubscribed" means g_r < 1, that is, demand on r would exceed its capacity.
- Makespan is the last completion minus the earliest insertion. Slot utilization is
  busy slot-time over S × makespan.
- In streaming mode, the first job arrives after one exponential gap, not at t = 0.
