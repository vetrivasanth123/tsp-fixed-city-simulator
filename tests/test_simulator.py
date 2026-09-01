
"""Tests for the fixed-city TSP simulator."""

from pathlib import Path

import pytest

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "instances" / "five_cities.json"
CUSTOM_PATH = ROOT / "instances" / "five_cities_custom_cost.json"


@pytest.fixture
def instance():
    return TSPInstance.from_json(DEFAULT_PATH)


@pytest.fixture
def simulator(instance):
    return TSPSimulator(instance, seed=42)


def test_simulator_initial_state(simulator):
    assert simulator.tour == [simulator.start_city]
    assert simulator.current_city == simulator.start_city
    assert simulator.total_cost == pytest.approx(0.0)
    assert simulator.total_distance == pytest.approx(0.0)
    assert simulator.done is False


def test_reset_creates_random_start(instance):
    simulator = TSPSimulator(instance, seed=42)
    state = simulator.reset()

    assert len(simulator.tour) == 1
    assert simulator.tour[0] == simulator.start_city
    assert state["current_city"] == simulator.start_city
    assert simulator.total_cost == pytest.approx(0.0)


def test_available_actions_excludes_visited_city(simulator):
    available = simulator.available_actions()

    assert simulator.start_city not in available
    assert len(available) == simulator.instance.num_cities - 1

    for city in available:
        assert city not in simulator.tour


def test_step_adds_city(simulator):
    next_city = simulator.available_actions()[0]
    old_city = simulator.current_city

    simulator.step(next_city)

    assert simulator.tour == [
        simulator.start_city,
        next_city,
    ]
    assert simulator.current_city == next_city

    expected = simulator.instance.cost(old_city, next_city)

    assert simulator.total_cost == pytest.approx(expected)
    assert simulator.total_distance == pytest.approx(expected)


def test_cost_updates_after_step(simulator):
    first_city = simulator.start_city
    next_city = simulator.available_actions()[0]

    simulator.step(next_city)

    expected = simulator.instance.cost(
        first_city,
        next_city,
    )

    assert simulator.total_cost == pytest.approx(expected)


def test_simulator_uses_custom_cost():
    instance = TSPInstance.from_json(CUSTOM_PATH)
    simulator = TSPSimulator(instance, seed=42)

    current = simulator.current_city
    next_city = simulator.available_actions()[0]

    expected = instance.cost(current, next_city)

    simulator.step(next_city)

    assert simulator.total_cost == pytest.approx(expected)


def test_multiple_steps_build_tour(simulator):
    actions = simulator.available_actions()[:3]

    for city in actions:
        simulator.step(city)

    assert len(simulator.tour) == 4
    assert len(set(simulator.tour)) == 4
    assert simulator.current_city == actions[-1]


def test_close_tour_returns_to_start(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(simulator.available_actions()[0])

    cost_before = simulator.total_cost
    final_edge = simulator.instance.cost(
        simulator.current_city,
        simulator.start_city,
    )

    simulator.close_tour()

    assert simulator.total_cost == pytest.approx(
        cost_before + final_edge
    )
    assert simulator.done is True


def test_complete_five_city_tour(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(simulator.available_actions()[0])

    assert len(simulator.tour) == 5
    assert len(set(simulator.tour)) == 5
    assert sorted(simulator.tour) == [0, 1, 2, 3, 4]

    simulator.close_tour()

    assert simulator.done is True
    assert simulator.total_cost > 0.0


def test_invalid_city_index_is_rejected(simulator):
    with pytest.raises((ValueError, IndexError)):
        simulator.step(5)


def test_negative_city_index_is_rejected(simulator):
    with pytest.raises((ValueError, IndexError)):
        simulator.step(-1)


def test_duplicate_city_is_rejected(simulator):
    next_city = simulator.available_actions()[0]
    simulator.step(next_city)

    with pytest.raises(ValueError):
        simulator.step(next_city)


def test_close_empty_tour_is_not_allowed(instance):
    simulator = TSPSimulator(instance, seed=42)
    simulator.tour = []
    simulator.current_city = None

    with pytest.raises(ValueError):
        simulator.close_tour()


def test_step_after_completion_is_rejected(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(simulator.available_actions()[0])

    simulator.close_tour()

    with pytest.raises(RuntimeError):
        simulator.step(0)


def test_state_contains_cost(simulator):
    state = simulator.state()

    required = {
        "tour",
        "current_city",
        "start_city",
        "visited",
        "available_actions",
        "total_cost",
        "total_distance",
        "done",
    }

    assert required.issubset(state)
    assert state["available_actions"] == simulator.available_actions()

