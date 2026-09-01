"""
Tests for the TSP instance representation.
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
    """The distance matrix should be N x N."""
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
    """All pairwise distances should be nonnegative."""
    instance = TSPInstance.from_json(INSTANCE_PATH)

    assert np.all(instance.distance_matrix >= 0.0)


def test_known_distance():
    """Check one manually known Euclidean distance."""
    instance = TSPInstance.from_json(INSTANCE_PATH)

    # City 0 = (0, 0)
    # City 1 = (2, 1)
    # Distance = sqrt(2^2 + 1^2) = sqrt(5)
    expected = np.sqrt(5.0)

    assert instance.distance_matrix[0, 1] == pytest.approx(expected)
