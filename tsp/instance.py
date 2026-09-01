from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class TSPInstance:
    """Fixed-city Travelling Salesman Problem instance."""

    coordinates: np.ndarray
    distance_matrix: np.ndarray
    city_names: list[str]

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

        n = len(self.coordinates)

        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n_cities, 2)")

        if self.distance_matrix.shape != (n, n):
            raise ValueError(
                "distance_matrix must have shape "
                f"({n}, {n})"
            )

        if len(self.city_names) != n:
            raise ValueError("city_names must contain one name per city")

        if not np.allclose(self.distance_matrix, self.distance_matrix.T):
            raise ValueError("distance_matrix must be symmetric")

        if not np.allclose(np.diag(self.distance_matrix), 0.0):
            raise ValueError("distance_matrix diagonal must be zero")

        if np.any(self.distance_matrix < 0):
            raise ValueError("distance_matrix cannot contain negative values")

    @property
    def n_cities(self) -> int:
        return len(self.coordinates)

    def tour_length(self, tour: list[int] | np.ndarray) -> float:
        """Return the closed-tour length for a city ordering."""

        tour = np.asarray(tour, dtype=int)

        if tour.ndim != 1 or len(tour) != self.n_cities:
            raise ValueError("tour must contain every city exactly once")

        if set(tour.tolist()) != set(range(self.n_cities)):
            raise ValueError("tour must be a permutation of all city indices")

        return float(
            sum(
                self.distance_matrix[tour[i], tour[(i + 1) % self.n_cities]]
                for i in range(self.n_cities)
            )
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TSPInstance":
        """Load a fixed TSP instance from a JSON file."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        coordinates = np.asarray(
            [city["coordinates"] for city in data["cities"]],
            dtype=float,
        )

        city_names = [
            city.get("name", str(i))
            for i, city in enumerate(data["cities"])
        ]

        if "distance_matrix" in data:
            distance_matrix = np.asarray(
                data["distance_matrix"],
                dtype=float,
            )
        else:
            distance_matrix = cls._compute_distance_matrix(coordinates)

        return cls(
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            city_names=city_names,
        )

    @staticmethod
    def _compute_distance_matrix(
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Compute Euclidean pairwise distances."""

        differences = coordinates[:, None, :] - coordinates[None, :, :]
        return np.linalg.norm(differences, axis=2)
