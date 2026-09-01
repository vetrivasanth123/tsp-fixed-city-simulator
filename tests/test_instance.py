
"""Tests for the TSP instance representation and cost abstraction."""

from pathlib import Path

import numpy as np
import pytest

from tsp.instance import TSPInstance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "instances" / "five_cities.json"
CUSTOM_PATH = ROOT / "instances" / "five_cities_custom_cost.json"


def test_load_five_city_instance():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    assert instance.name == "five_cities"
    assert instance.num_cities == 5
    assert instance.coordinates.shape == (5, 2)


def test_coordinates_are_loaded_correctly():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    expected = np.array([
        [0.0, 0.0],
        [2.0, 1.0],
        [4.0, 0.0],
        [3.0, 3.0],
        [0.5, 3.0],
    ])

    np.testing.assert_allclose(instance.coordinates, expected)


def test_distance_matrix_properties():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    assert instance.distance_matrix.shape == (5, 5)
    np.testing.assert_allclose(
        np.diag(instance.distance_matrix),
        0.0,
    )
    np.testing.assert_allclose(
        instance.distance_matrix,
        instance.distance_matrix.T,
    )
    assert np.all(instance.distance_matrix >= 0.0)


def test_known_distance():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    assert instance.distance_matrix[0, 1] == pytest.approx(
        np.sqrt(5.0)
    )


def test_distance_method_matches_matrix():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    assert instance.distance(0, 1) == pytest.approx(
        instance.distance_matrix[0, 1]
    )


def test_cost_method_exists():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    assert callable(instance.cost)


def test_default_cost_matches_euclidean():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    np.testing.assert_allclose(
        instance.cost_matrix,
        instance.distance_matrix,
    )


def test_default_cost_is_nonnegative_and_zero_diagonal():
    instance = TSPInstance.from_json(DEFAULT_PATH)

    assert np.all(instance.cost_matrix >= 0.0)
    np.testing.assert_allclose(
        np.diag(instance.cost_matrix),
        0.0,
    )


def test_custom_cost_matrix_is_loaded_from_json():
    instance = TSPInstance.from_json(CUSTOM_PATH)

    expected = np.array([
        [0.0, 10.0, 20.0, 15.0, 8.0],
        [10.0, 0.0, 12.0, 18.0, 14.0],
        [20.0, 12.0, 0.0, 9.0, 16.0],
        [15.0, 18.0, 9.0, 0.0, 11.0],
        [8.0, 14.0, 16.0, 11.0, 0.0],
    ])

    np.testing.assert_allclose(
        instance.cost_matrix,
        expected,
    )


def test_custom_cost_is_distinct_from_euclidean():
    instance = TSPInstance.from_json(CUSTOM_PATH)

    assert instance.cost(0, 1) == pytest.approx(10.0)
    assert instance.distance(0, 1) == pytest.approx(np.sqrt(5.0))
    assert instance.cost(0, 1) != pytest.approx(
        instance.distance(0, 1)
    )

