
import numpy as np
import pytest

from tsp.env import TSPEnv
from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator


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
    assert env.action_space.n == 6
    assert env.close_action == 5


def test_step(env):
    _, info = env.reset(seed=42)
    action = info["available_actions"][0]

    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert reward < 0
    assert not terminated and not truncated
    assert action in info["tour"]
    assert obs["visited_mask"].sum() == 2


def test_state_continuity(env):
    obs, info = env.reset(seed=42)

    for _ in range(3):
        action = info["available_actions"][0]
        obs, _, _, _, info = env.step(action)
        state = env.simulator.state()

        assert info["tour"] == state["tour"]
        assert info["current_city"] == state["current_city"]
        assert info["start_city"] == state["start_city"]
        assert info["available_actions"] == state["available_actions"]
        assert np.isclose(info["total_cost"], state["total_cost"])
        assert obs["current_city"] == state["current_city"]
        assert np.isclose(obs["total_cost"][0], state["total_cost"])


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
        obs, _, terminated, truncated, info = env.step(
            info["available_actions"][0]
        )

    assert not terminated
    assert not truncated
    assert len(info["tour"]) == 5
    assert len(set(info["tour"])) == 5
    assert obs["visited_mask"].sum() == 5
    assert not info["available_actions"]

    obs, reward, terminated, truncated, info = env.step(
        env.close_action
    )

    assert terminated and not truncated
    assert reward < 0
    assert len(info["tour"]) == 5
    assert obs["visited_mask"].sum() == 5
    assert not info["available_actions"]
    assert info["total_cost"] > 0
    assert info["current_city"] == info["start_city"]

def test_reward_matches_tour_cost(env):
    _, info = env.reset(seed=42)
    total_reward = 0.0

    while info["available_actions"]:
        _, reward, _, _, info = env.step(
            info["available_actions"][0]
        )
        total_reward += reward

    _, reward, terminated, truncated, info = env.step(
        env.close_action
    )
    total_reward += reward

    assert terminated and not truncated
    assert np.isclose(-total_reward, info["total_cost"])


def test_seed_reproducibility(env):
    _, info1 = env.reset(seed=123)
    _, info2 = env.reset(seed=123)

    assert info1["start_city"] == info2["start_city"]
    
def test_environment_owns_simulator(env):
    assert isinstance(env.simulator, TSPSimulator)
    assert env.simulator.instance is env.instance
    
def test_env_uses_same_simulator(env):
    _, info = env.reset(seed=42)

    action = info["available_actions"][0]
    obs, _, _, _, info = env.step(action)

    state = env.simulator.state()

    assert info["tour"] == state["tour"]
    assert obs["current_city"] == state["current_city"]
    assert np.isclose(obs["total_cost"][0], state["total_cost"])    

def test_premature_close_is_rejected(env):
    env.reset(seed=42)

    with pytest.raises(ValueError):
        env.step(env.close_action)

def test_available_actions_match_visited_mask(env):
    obs, info = env.reset(seed=42)

    for city in range(env.instance.num_cities):
        if city in info["tour"]:
            assert obs["visited_mask"][city] == 1
        else:
            assert obs["visited_mask"][city] == 0
            assert city in info["available_actions"]

def test_custom_cost_environment():
    instance = TSPInstance.from_json(
        "instances/five_cities_custom_cost.json"
    )
    env = TSPEnv(instance, seed=42)

    _, info = env.reset(seed=42)
    action = info["available_actions"][0]

    _, reward, _, _, _ = env.step(action)

    expected = -instance.cost(
        info["current_city"],
        action,
    )

    assert reward == pytest.approx(expected)
