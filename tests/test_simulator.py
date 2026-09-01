```python
from pathlib import Path

import numpy as np
import pytest

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator


INSTANCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "instances"
    / "five_cities.json"
)


@pytest.fixture
def instance():
    """Load the standard five-city test instance."""
    return TSPInstance.from_json(INSTANCE_PATH)


@pytest.fixture
def simulator(instance):
    """Create a simulator with a deterministic starting city."""
    return TSPSimulator(instance)


def test_simulator_initial_state(simulator):
    """A new simulator should start with exactly one random city."""
    assert len(simulator.tour) == 1
    assert simulator.current_city == simulator.tour[0]
    assert simulator.current_city in range(5)
    assert simulator.total_distance == 0.0
    assert simulator.done is False


def test_reset_selects_valid_starting_city(simulator):
    """Reset should select a valid city as the starting city."""
    state = simulator.reset(seed=1)

    assert len(state["tour"]) == 1
    assert state["current_city"] in range(5)
    assert state["tour"] == [state["current_city"]]
    assert state["visited"] == [state["current_city"]]
    assert state["total_distance"] == 0.0
    assert state["done"] is False


def test_reset_with_same_seed_is_reproducible(instance):
    """The same seed should produce the same starting city."""
    simulator_1 = TSPSimulator(instance)
    simulator_2 = TSPSimulator(instance)

    state_1 = simulator_1.reset(seed=42)
    state_2 = simulator_2.reset(seed=42)

    assert state_1["current_city"] == state_2["current_city"]
    assert state_1["tour"] == state_2["tour"]


def test_available_actions_exclude_starting_city(simulator):
    """The starting city must not be available as a next action."""
    simulator.reset(seed=42)

    start = simulator.current_city
    actions = simulator.available_actions()

    assert start not in actions
    assert sorted(actions) == sorted(
        set(range(simulator.instance.num_cities)) - {start}
    )


def test_available_actions_decrease_after_step(simulator):
    """Visiting a city should remove it from available actions."""
    simulator.reset(seed=42)

    actions_before = simulator.available_actions()
    next_city = actions_before[0]

    simulator.step(next_city)

    actions_after = simulator.available_actions()

    assert next_city not in actions_after
    assert len(actions_after) == len(actions_before) - 1


def test_step_adds_city(simulator):
    """Stepping to a city should add it to the tour."""
    simulator.reset(seed=42)

    start = simulator.current_city
    next_city = simulator.available_actions()[0]

    simulator.step(next_city)

    assert simulator.tour == [start, next_city]
    assert simulator.current_city == next_city


def test_multiple_steps_build_tour(simulator):
    """Multiple valid steps should build the expected tour."""
    simulator.reset(seed=42)

    start = simulator.current_city
    actions = simulator.available_actions()

    selected = actions[:3]

    for city in selected:
        simulator.step(city)

    assert simulator.tour == [start] + selected
    assert simulator.current_city == selected[-1]


def test_distance_updates_after_step(simulator):
    """Distance should update when moving between cities."""
    simulator.reset(seed=42)

    start = simulator.current_city
    next_city = simulator.available_actions()[0]

    simulator.step(next_city)

    expected = simulator.instance.distance(start, next_city)

    assert simulator.total_distance == pytest.approx(expected)


def test_close_tour_returns_to_start(simulator):
    """Closing the tour should add the final edge to the start."""
    simulator.reset(seed=42)

    while simulator.available_actions():
        simulator.step(simulator.available_actions()[0])

    distance_before_close = simulator.total_distance
    final_city = simulator.current_city
    start_city = simulator.tour[0]

    simulator.close_tour()

    final_edge = simulator.instance.distance(
        final_city,
        start_city,
    )

    assert simulator.total_distance == pytest.approx(
        distance_before_close + final_edge
    )
    assert simulator.done is True


def test_complete_five_city_tour(simulator):
    """A five-city tour should contain every city exactly once."""
    simulator.reset(seed=42)

    while simulator.available_actions():
        simulator.step(simulator.available_actions()[0])

    assert len(simulator.tour) == 5
    assert sorted(simulator.tour) == [0, 1, 2, 3, 4]

    simulator.close_tour()

    assert simulator.done is True


def test_available_actions_empty_after_all_cities_visited(simulator):
    """No next-city actions should remain after visiting every city."""
    simulator.reset(seed=42)

    while simulator.available_actions():
        simulator.step(simulator.available_actions()[0])

    assert simulator.available_actions() == []


def test_invalid_city_index_is_rejected(simulator):
    """A city index outside the instance should raise an error."""
    simulator.reset(seed=42)

    with pytest.raises((ValueError, IndexError)):
        simulator.step(5)


def test_negative_city_index_is_rejected(simulator):
    """Negative city indices should not be accepted."""
    simulator.reset(seed=42)

    with pytest.raises((ValueError, IndexError)):
        simulator.step(-1)


def test_duplicate_city_is_rejected(simulator):
    """A city should not be visited twice."""
    simulator.reset(seed=42)

    start = simulator.current_city

    with pytest.raises(ValueError):
        simulator.step(start)


def test_empty_tour_cannot_be_closed(simulator):
    """An empty tour should not be closable."""
    simulator.tour = []
    simulator.current_city = None

    with pytest.raises(ValueError):
        simulator.close_tour()


def test_step_after_completion_is_rejected(simulator):
    """No actions should be accepted after the tour is complete."""
    simulator.reset(seed=42)

    while simulator.available_actions():
        simulator.step(simulator.available_actions()[0])

    simulator.close_tour()

    with pytest.raises(RuntimeError):
        simulator.step(0)


def test_close_tour_is_idempotent(simulator):
    """Calling close_tour twice should not add the return edge twice."""
    simulator.reset(seed=42)

    while simulator.available_actions():
        simulator.step(simulator.available_actions()[0])

    simulator.close_tour()
    distance_after_first_close = simulator.total_distance

    simulator.close_tour()

    assert simulator.total_distance == pytest.approx(
        distance_after_first_close
    )
    assert simulator.done is True
```
