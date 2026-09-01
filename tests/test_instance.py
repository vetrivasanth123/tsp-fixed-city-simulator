
"""
Tests for the TSP instance representation and cost abstraction.
"""

from pathlib import Path

import numpy as np
import pytest

from tsp.instance import TSPInstance


INSTANCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "instances"
    / "five_cities.json"
)


def test_load_five_city_instance():
    """The five-city JSON file should load correctly."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    assert instance.name == "five_cities"
    assert instance.num_cities == 5
    assert instance.coordinates.shape == (5, 2)


def test_coordinates_are_loaded_correctly():
    """City coordinates should match the JSON instance."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    expected = np.array(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [4.0, 0.0],
            [3.0, 3.0],
            [0.5, 3.0],
        ]
    )

    np.testing.assert_allclose(
        instance.coordinates,
        expected,
    )


def test_distance_matrix_shape():
    """The Euclidean distance matrix should be N x N."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    assert instance.distance_matrix.shape == (5, 5)


def test_distance_matrix_diagonal_is_zero():
    """Distance from every city to itself should be zero."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    np.testing.assert_allclose(
        np.diag(instance.distance_matrix),
        0.0,
    )


def test_distance_matrix_is_symmetric():
    """The Euclidean distance matrix should be symmetric."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    np.testing.assert_allclose(
        instance.distance_matrix,
        instance.distance_matrix.T,
    )


def test_distance_matrix_is_nonnegative():
    """All Euclidean distances should be nonnegative."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    assert np.all(instance.distance_matrix >= 0.0)


def test_known_distance():
    """Check one manually known Euclidean distance."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    # City 0 = (0, 0)
    # City 1 = (2, 1)
    # Distance = sqrt(2^2 + 1^2) = sqrt(5)

    expected = np.sqrt(5.0)

    assert instance.distance_matrix[0, 1] == pytest.approx(
        expected
    )


def test_distance_method_matches_distance_matrix():
    """The distance method should return the stored Euclidean distance."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    expected = instance.distance_matrix[0, 1]

    assert instance.distance(0, 1) == pytest.approx(expected)


def test_cost_method_exists():
    """The instance should expose cost as the primary edge-cost interface."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    assert callable(instance.cost)


def test_default_cost_matches_euclidean_distance():
    """
    Euclidean distance should remain the default cost for
    the current five-city instance.
    """

    instance = TSPInstance.from_json(INSTANCE_PATH)

    for city_a in range(instance.num_cities):
        for city_b in range(instance.num_cities):
            assert instance.cost(city_a, city_b) == pytest.approx(
                instance.distance(city_a, city_b)
            )


def test_cost_is_nonnegative_for_default_instance():
    """The default Euclidean-based cost should be nonnegative."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    costs = np.array(
        [
            [
                instance.cost(i, j)
                for j in range(instance.num_cities)
            ]
            for i in range(instance.num_cities)
        ]
    )

    assert np.all(costs >= 0.0)


def test_cost_diagonal_is_zero_for_default_instance():
    """The default cost from a city to itself should be zero."""

    instance = TSPInstance.from_json(INSTANCE_PATH)

    for city in range(instance.num_cities):
        assert instance.cost(city, city) == pytest.approx(0.0)

