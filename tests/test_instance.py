
"""Tests for the TSP instance representation and cost abstraction."""

import numpy as np
import pytest

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance


def make_coordinates(width=10, height=10, n_cities=8, seed=42):
    return np.asarray(
        CityLocationGenerator(
            width, height, n_cities, seed=seed
        ).generate(),
        dtype=float,
    )


def make_cost_matrix(n_cities, seed=42):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 100.0, (n_cities, n_cities))
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_coordinates_and_city_count():
    coordinates = make_coordinates(20, 15, 10)
    instance = TSPInstance(coordinates)

    assert instance.num_cities == len(coordinates)
    assert instance.coordinates.shape == (len(coordinates), 2)
    np.testing.assert_allclose(instance.coordinates, coordinates)


def test_distance_matrix_properties():
    coordinates = make_coordinates(20, 15, 10)
    instance = TSPInstance(coordinates)

    n = instance.num_cities

    assert instance.distance_matrix.shape == (n, n)
    np.testing.assert_allclose(
        instance.distance_matrix,
        instance.distance_matrix.T,
    )
    np.testing.assert_allclose(
        np.diag(instance.distance_matrix),
        0.0,
    )
    assert np.all(instance.distance_matrix >= 0.0)


def test_distance_matches_coordinates():
    coordinates = make_coordinates()
    instance = TSPInstance(coordinates)

    city_a = 0
    city_b = instance.num_cities - 1

    expected = np.linalg.norm(
        coordinates[city_a] - coordinates[city_b]
    )

    assert instance.distance(city_a, city_b) == pytest.approx(expected)


def test_distance_method_matches_matrix():
    instance = TSPInstance(make_coordinates())

    city_a = 0
    city_b = instance.num_cities - 1

    assert instance.distance(city_a, city_b) == pytest.approx(
        instance.distance_matrix[city_a, city_b]
    )


def test_default_cost_matches_euclidean():
    instance = TSPInstance(make_coordinates())

    np.testing.assert_allclose(
        instance.cost_matrix,
        instance.distance_matrix,
    )


def test_default_cost_is_valid():
    instance = TSPInstance(make_coordinates())

    assert np.all(instance.cost_matrix >= 0.0)
    np.testing.assert_allclose(
        np.diag(instance.cost_matrix),
        0.0,
    )


def test_custom_cost_matrix_is_supported():
    coordinates = make_coordinates()
    cost_matrix = make_cost_matrix(len(coordinates))

    instance = TSPInstance(
        coordinates,
        cost_matrix=cost_matrix,
    )

    np.testing.assert_allclose(
        instance.cost_matrix,
        cost_matrix,
    )


def test_custom_cost_is_used():
    coordinates = make_coordinates()
    cost_matrix = make_cost_matrix(len(coordinates))
    instance = TSPInstance(
        coordinates,
        cost_matrix=cost_matrix,
    )

    city_a = 0
    city_b = instance.num_cities - 1

    assert instance.cost(city_a, city_b) == pytest.approx(
        cost_matrix[city_a, city_b]
    )


@pytest.mark.parametrize(
    "width,height,n_cities",
    [
        (5, 5, 3),
        (20, 10, 6),
        (100, 50, 15),
    ],
)
def test_instance_supports_dynamic_problem_sizes(
    width, height, n_cities
):
    coordinates = make_coordinates(
        width, height, n_cities
    )
    instance = TSPInstance(coordinates)

    assert instance.num_cities == n_cities
    assert instance.coordinates.shape == (n_cities, 2)
    assert instance.distance_matrix.shape == (
        n_cities, n_cities
    )


@pytest.mark.parametrize(
    "coordinates",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.array([[0.0, 0.0]]),
    ],
)
def test_invalid_coordinates_are_rejected(coordinates):
    with pytest.raises(ValueError):
        TSPInstance(coordinates)


def test_nonfinite_coordinates_are_rejected():
    coordinates = make_coordinates()
    coordinates[0, 0] = np.nan

    with pytest.raises(ValueError):
        TSPInstance(coordinates)


def test_invalid_cost_matrix_shape_is_rejected():
    coordinates = make_coordinates(n_cities=6)
    n = len(coordinates)

    with pytest.raises(ValueError):
        TSPInstance(
            coordinates,
            cost_matrix=np.zeros((n - 1, n - 1)),
        )


def test_nonfinite_cost_matrix_is_rejected():
    coordinates = make_coordinates()
    cost_matrix = make_cost_matrix(len(coordinates))
    cost_matrix[0, 1] = np.inf

    with pytest.raises(ValueError):
        TSPInstance(
            coordinates,
            cost_matrix=cost_matrix,
        )


def test_negative_cost_matrix_is_rejected():
    coordinates = make_coordinates()
    cost_matrix = make_cost_matrix(len(coordinates))
    cost_matrix[0, 1] = -1.0

    with pytest.raises(ValueError):
        TSPInstance(
            coordinates,
            cost_matrix=cost_matrix,
        )

