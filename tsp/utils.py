
from __future__ import annotations

import numpy as np


def validate_tour(
    tour: list[int] | np.ndarray,
    n_cities: int,
) -> None:
    """Validate that a tour visits every city exactly once."""

    tour = np.asarray(tour, dtype=int)

    if tour.ndim != 1:
        raise ValueError("tour must be one-dimensional")

    if len(tour) != n_cities:
        raise ValueError(
            f"tour must contain exactly {n_cities} cities"
        )

    if set(tour.tolist()) != set(range(n_cities)):
        raise ValueError(
            "tour must contain every city exactly once"
        )


def close_tour(
    tour: list[int] | np.ndarray,
) -> np.ndarray:
    """Return a closed tour by appending the starting city."""

    tour = np.asarray(tour, dtype=int)

    if tour.ndim != 1 or len(tour) == 0:
        raise ValueError(
            "tour must be a non-empty one-dimensional sequence"
        )

    return np.append(tour, tour[0])


def tour_cost(
    tour: list[int] | np.ndarray,
    instance,
) -> float:
    """
    Calculate the total cost of a closed TSP tour.

    The instance provides the edge-cost abstraction through
    instance.cost(city_a, city_b).
    """

    tour = np.asarray(tour, dtype=int)

    validate_tour(
        tour,
        instance.num_cities,
    )

    closed_tour = close_tour(tour)

    return float(
        sum(
            instance.cost(
                closed_tour[i],
                closed_tour[i + 1],
            )
            for i in range(len(tour))
        )
    )


def tour_length(
    tour: list[int] | np.ndarray,
    distance_matrix: np.ndarray,
) -> float:
    """
    Backward-compatible Euclidean tour length.

    This function is retained so existing code using
    distance_matrix continues to work.
    """

    distance_matrix = np.asarray(
        distance_matrix,
        dtype=float,
    )

    tour = np.asarray(
        tour,
        dtype=int,
    )

    if distance_matrix.ndim != 2:
        raise ValueError(
            "distance_matrix must be two-dimensional"
        )

    if (
        distance_matrix.shape[0]
        != distance_matrix.shape[1]
    ):
        raise ValueError(
            "distance_matrix must be square"
        )

    validate_tour(
        tour,
        distance_matrix.shape[0],
    )

    closed_tour = close_tour(tour)

    return float(
        sum(
            distance_matrix[
                closed_tour[i],
                closed_tour[i + 1],
            ]
            for i in range(len(tour))
        )
    )


def euclidean_distance_matrix(
    coordinates: np.ndarray,
) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix.
    """

    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    if coordinates.ndim != 2:
        raise ValueError(
            "coordinates must be a two-dimensional array"
        )

    if coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (n_cities, 2)"
        )

    differences = (
        coordinates[:, None, :]
        - coordinates[None, :, :]
    )

    return np.linalg.norm(
        differences,
        axis=2,
    )

