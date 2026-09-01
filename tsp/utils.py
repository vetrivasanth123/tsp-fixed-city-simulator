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
    """Append the starting city to close the tour."""

    tour = np.asarray(tour, dtype=int)

    if tour.ndim != 1 or len(tour) == 0:
        raise ValueError(
            "tour must be a non-empty one-dimensional sequence"
        )

    return np.append(tour, tour[0])


def tour_cost(
    tour: list[int] | np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """Calculate the total cost of a closed TSP tour."""

    cost_matrix = np.asarray(cost_matrix, dtype=float)
    tour = np.asarray(tour, dtype=int)

    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")

    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be square")

    validate_tour(tour, cost_matrix.shape[0])

    closed = close_tour(tour)

    return float(
        sum(
            cost_matrix[closed[i], closed[i + 1]]
            for i in range(len(tour))
        )
    )


def tour_length(
    tour: list[int] | np.ndarray,
    distance_matrix: np.ndarray,
) -> float:
    """Backward-compatible alias for tour_cost()."""

    return tour_cost(tour, distance_matrix)


def euclidean_distance_matrix(
    coordinates: np.ndarray,
) -> np.ndarray:
    """Compute a pairwise Euclidean distance matrix."""

    coordinates = np.asarray(coordinates, dtype=float)

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

    return np.linalg.norm(differences, axis=2)
