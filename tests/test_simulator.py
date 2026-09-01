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
    """Create a simulator with a fixed seed."""
    return TSPSimulator(instance, seed=42)


# ============================================================
# INITIAL STATE / RESET
# ============================================================

def test_simulator_initial_state(simulator):
    assert simulator.tour == [simulator.start_city]
    assert simulator.current_city == simulator.start_city
    assert simulator.total_distance == pytest.approx(0.0)
    assert simulator.done is False
    assert simulator.start_city in range(
        simulator.instance.num_cities
    )


def test_reset_creates_valid_start(instance):
    simulator = TSPSimulator(instance, seed=42)

    state = simulator.reset()

    assert len(simulator.tour) == 1
    assert simulator.tour[0] == simulator.start_city
    assert simulator.current_city == simulator.start_city
    assert state["current_city"] == simulator.start_city
    assert simulator.total_distance == pytest.approx(0.0)


def test_reset_with_seed_is_reproducible(instance):
    simulator = TSPSimulator(instance)

    state1 = simulator.reset(seed=42)
    state2 = simulator.reset(seed=42)

    assert state1["start_city"] == state2["start_city"]


def test_different_seed_can_change_start(instance):
    simulator = TSPSimulator(instance)

    state1 = simulator.reset(seed=1)
    state2 = simulator.reset(seed=3)

    assert state1["start_city"] != state2["start_city"]


# ============================================================
# AVAILABLE ACTIONS
# ============================================================

def test_available_actions_excludes_start(simulator):
    available = simulator.available_actions()

    assert simulator.start_city not in available
    assert len(available) == (
        simulator.instance.num_cities - 1
    )


def test_available_actions_contain_only_unvisited(simulator):
    available = simulator.available_actions()

    for city in available:
        assert city not in simulator.tour


def test_available_actions_shrink_after_step(simulator):
    next_city = simulator.available_actions()[0]

    before = simulator.available_actions()

    simulator.step(next_city)

    after = simulator.available_actions()

    assert len(after) == len(before) - 1
    assert next_city not in after


# ============================================================
# STEP
# ============================================================

def test_step_adds_city(simulator):
    next_city = simulator.available_actions()[0]

    old_city = simulator.current_city

    simulator.step(next_city)

    assert simulator.tour == [
        simulator.start_city,
        next_city,
    ]

    assert simulator.current_city == next_city

    expected_distance = simulator.instance.distance(
        old_city,
        next_city,
    )

    assert simulator.total_distance == pytest.approx(
        expected_distance
    )


def test_multiple_steps_build_tour(simulator):
    actions = simulator.available_actions()[:3]

    for city in actions:
        simulator.step(city)

    assert len(simulator.tour) == 4
    assert len(set(simulator.tour)) == 4
    assert simulator.current_city == actions[-1]


def test_distance_updates_after_step(simulator):
    first_city = simulator.start_city
    next_city = simulator.available_actions()[0]

    simulator.step(next_city)

    expected = simulator.instance.distance(
        first_city,
        next_city,
    )

    assert simulator.total_distance == pytest.approx(
        expected
    )


def test_duplicate_city_is_rejected(simulator):
    next_city = simulator.available_actions()[0]

    simulator.step(next_city)

    with pytest.raises(ValueError):
        simulator.step(next_city)


def test_invalid_city_index_is_rejected(simulator):
    with pytest.raises((ValueError, IndexError)):
        simulator.step(5)


def test_negative_city_index_is_rejected(simulator):
    with pytest.raises((ValueError, IndexError)):
        simulator.step(-1)


def test_non_integer_city_is_rejected(simulator):
    with pytest.raises(TypeError):
        simulator.step(1.5)


# ============================================================
# COMPLETE TOUR
# ============================================================

def test_complete_five_city_tour(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(
            simulator.available_actions()[0]
        )

    assert len(simulator.tour) == 5
    assert len(set(simulator.tour)) == 5
    assert sorted(simulator.tour) == [0, 1, 2, 3, 4]

    simulator.close_tour()

    assert simulator.done is True
    assert simulator.total_distance > 0.0


def test_close_tour_returns_to_start(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(
            simulator.available_actions()[0]
        )

    distance_before_close = simulator.total_distance
    last_city = simulator.current_city
    start_city = simulator.start_city

    simulator.close_tour()

    final_edge = simulator.instance.distance(
        last_city,
        start_city,
    )

    assert simulator.total_distance == pytest.approx(
        distance_before_close + final_edge
    )

    assert simulator.done is True


def test_close_tour_is_idempotent(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(
            simulator.available_actions()[0]
        )

    simulator.close_tour()

    distance = simulator.total_distance

    simulator.close_tour()

    assert simulator.done is True
    assert simulator.total_distance == pytest.approx(
        distance
    )


def test_step_after_completion_is_rejected(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(
            simulator.available_actions()[0]
        )

    simulator.close_tour()

    with pytest.raises(RuntimeError):
        simulator.step(0)


# ============================================================
# STATE
# ============================================================

def test_state_contains_required_fields(simulator):
    state = simulator.state()

    required = {
        "tour",
        "start_city",
        "current_city",
        "visited",
        "available_actions",
        "total_distance",
        "done",
    }

    assert required.issubset(state.keys())


def test_state_matches_simulator(simulator):
    state = simulator.state()

    assert state["tour"] == simulator.tour
    assert state["start_city"] == simulator.start_city
    assert state["current_city"] == simulator.current_city
    assert state["visited"] == simulator.tour
    assert (
        state["available_actions"]
        == simulator.available_actions()
    )
    assert state["total_distance"] == simulator.total_distance
    assert state["done"] == simulator.done


# ============================================================
# TRAJECTORY / VISUALIZATION SUPPORT
# ============================================================

def test_trajectory_records_start(simulator):
    trajectory = simulator.trajectory()

    assert len(trajectory) == 1

    record = trajectory[0]

    assert record["event"] == "start"
    assert record["current_city"] == simulator.start_city
    assert record["tour"] == [simulator.start_city]
    assert record["action"] is None


def test_trajectory_records_each_step(simulator):
    actions = simulator.available_actions()[:2]

    for city in actions:
        simulator.step(city)

    trajectory = simulator.trajectory()

    # start + 2 step events
    assert len(trajectory) == 3

    assert trajectory[1]["event"] == "step"
    assert trajectory[1]["action"] == actions[0]

    assert trajectory[2]["event"] == "step"
    assert trajectory[2]["action"] == actions[1]


def test_trajectory_records_close(simulator):
    while len(simulator.tour) < simulator.instance.num_cities:
        simulator.step(
            simulator.available_actions()[0]
        )

    simulator.close_tour()

    trajectory = simulator.trajectory()

    assert trajectory[-1]["event"] == "close"
    assert trajectory[-1]["action"] == simulator.start_city
    assert trajectory[-1]["done"] is True


def test_close_empty_tour_is_not_allowed(instance):
    simulator = TSPSimulator(instance, seed=42)

    simulator.tour = []
    simulator.current_city = None

    with pytest.raises(ValueError):
        simulator.close_tour()
