import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import toysched  # noqa: F401  registers ToyScheduler-v0
from toysched.config import Config
from toysched.env import ToySchedulerEnv
from toysched.workload import generate_workload


def test_passes_gymnasium_checker():
    env = ToySchedulerEnv(Config(num_jobs=10))
    check_env(env, skip_render_check=True)


def test_registered():
    env = gym.make("ToyScheduler-v0", config=Config(num_jobs=5))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_same_workload_as_baselines_for_a_seed():
    cfg = Config(num_jobs=20)
    env = ToySchedulerEnv(cfg)
    env.reset(seed=11)
    baseline = generate_workload(cfg, np.random.default_rng(11))
    for job, spec in zip(env.sim.jobs, baseline):
        assert job.spec.edges == spec.edges
        assert [t.work for t in job.spec.tasks] == [t.work for t in spec.tasks]


@pytest.mark.parametrize("mode", ["batch", "streaming"])
def test_random_episode(mode):
    cfg = Config(num_jobs=30, workload_mode=mode)
    env = ToySchedulerEnv(cfg)
    obs, info = env.reset(seed=0)
    q_total, completed, steps = info["q_integral"], 0, 0
    terminated = False
    while not terminated:
        assert env.observation_space.contains(obs)
        assert obs["eligible_mask"].any()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert reward == 0.0 and not truncated
        q_total += info["q_integral"]
        completed += len(info["completed_jobs"])
        steps += 1
    m = info["episode"]
    assert completed == m["num_finished"] == 30
    assert q_total == pytest.approx(m["total_latency"])
    assert "mean_gap_l1" in m
    assert set(info["slot_time_fraction"]) <= {"allocate", "refill", "fallback", "idle"}


def test_zero_action_means_equal_shares():
    env = ToySchedulerEnv(Config(num_jobs=10))
    obs, _ = env.reset(seed=0)
    shares = env._shares(np.zeros(env.action_space.shape))
    assert len(shares) == obs["eligible_mask"].sum()
    assert all(s == pytest.approx(1 / len(shares)) for s in shares.values())


def test_only_visible_jobs_get_slots():
    # In batch mode a visible job stays visible until it finishes, so at every step
    # the running tasks must all belong to the M oldest jobs.
    env = ToySchedulerEnv(Config(num_jobs=40, max_visible_jobs=3))
    env.reset(seed=0)
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        running_jobs = {t.job for t in env.sim.slots if t is not None}
        assert running_jobs <= set(env.sim.visible_jobs(3))
