"""
Tests for the fixed-city TSP simulator.
"""

from pathlib import Path

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
    """Create a fresh simulator."""
    return TSPSimulator(instance)


def test_simulator_initial_state(simulator):
    """A new simulator should start with an empty tour."""
    assert simulator.tour == []
    assert simulator.total_distance == pytest.approx(0.0)


def test_step_adds_city(simulator):
    """Stepping to a city should add it to the tour."""
    simulator.step(0)

    assert simulator.tour == [0]


def test_multiple_steps_build_tour(simulator):
    """Multiple valid steps should build the expected tour."""
    simulator.step(0)
    simulator.step(1)
    simulator.step(2)

    assert simulator.tour == [0, 1, 2]


def test_distance_updates_after_step(simulator):
    """Distance should update when moving between cities."""
    simulator.step(0)
    simulator.step(1)

    assert simulator.total_distance == pytest.approx(
        simulator.instance.distance_matrix[0, 1]
    )


def test_close_tour_returns_to_start(simulator):
    """Closing the tour should add the final edge to the start."""
    simulator.step(0)
    simulator.step(1)
    simulator.step(2)
    simulator.step(3)
    simulator.step(4)

    distance_before_close = simulator.total_distance

    simulator.close_tour()

    final_edge = simulator.instance.distance_matrix[4, 0]

    assert simulator.total_distance == pytest.approx(
        distance_before_close + final_edge
    )


def test_complete_five_city_tour(simulator):
    """A five-city tour should contain every city exactly once."""
    tour = [0, 1, 2, 3, 4]

    for city in tour:
        simulator.step(city)

    assert simulator.tour == tour

    simulator.close_tour()

    assert simulator.tour == tour


def test_invalid_city_index_is_rejected(simulator):
    """A city index outside the instance should raise an error."""
    with pytest.raises((ValueError, IndexError)):
        simulator.step(5)


def test_negative_city_index_is_rejected(simulator):
    """Negative city indices should not be accepted."""
    with pytest.raises((ValueError, IndexError)):
        simulator.step(-1)


def test_duplicate_city_is_rejected(simulator):
    """A city should not be visited twice."""
    simulator.step(0)

    with pytest.raises(ValueError):
        simulator.step(0)


def test_empty_tour_cannot_be_closed(simulator):
    """An empty tour should not be closable."""
    with pytest.raises(ValueError):
        simulator.close_tour()
