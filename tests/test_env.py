import numpy as np
import pytest

from tsp.env import TSPEnv
from tsp.instance import TSPInstance


@pytest.fixture
def env():
    instance = TSPInstance.from_json("instances/five_cities.json")
    return TSPEnv(instance, seed=42)


def test_reset(env):
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert obs["visited_mask"].sum() == 1
    assert obs["current_city"] == info["current_city"]
    assert obs["total_cost"][0] == 0.0
    assert len(info["available_actions"]) == 4


def test_action_space(env):
    assert env.action_space.n == 5


def test_step(env):
    _, info = env.reset(seed=42)
    action = info["available_actions"][0]

    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert reward < 0
    assert not terminated and not truncated
    assert action in info["tour"]
    assert obs["visited_mask"].sum() == 2


def test_invalid_action(env):
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(env.simulator.current_city)


def test_invalid_action_range(env):
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(10)


def test_complete_tour(env):
    obs, info = env.reset(seed=42)

    while info["available_actions"]:
        obs, reward, terminated, truncated, info = env.step(
            info["available_actions"][0]
        )

    assert terminated and not truncated
    assert len(info["tour"]) == 5
    assert len(set(info["tour"])) == 5
    assert obs["visited_mask"].sum() == 5
    assert info["available_actions"] == []
    assert info["total_cost"] > 0


def test_reward_matches_tour_cost(env):
    _, info = env.reset(seed=42)
    total_reward = 0.0

    while info["available_actions"]:
        _, reward, _, _, info = env.step(
            info["available_actions"][0]
        )
        total_reward += reward

    assert np.isclose(-total_reward, info["total_cost"])


def test_seed_reproducibility(env):
    _, info1 = env.reset(seed=123)
    _, info2 = env.reset(seed=123)

    assert info1["start_city"] == info2["start_city"]
