from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TSPInstance:
    """Fixed-city Travelling Salesman Problem instance."""

    def __init__(
        self,
        coordinates: np.ndarray,
        name: str = "tsp_instance",
    ) -> None:
        coordinates = np.asarray(coordinates, dtype=float)

        if coordinates.ndim != 2:
            raise ValueError("coordinates must be a 2D array.")

        if coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n_cities, 2).")

        if coordinates.shape[0] < 2:
            raise ValueError("A TSP instance must contain at least two cities.")

        self.name = name
        self.coordinates = coordinates
        self.num_cities = coordinates.shape[0]

        self.distance_matrix = self._compute_distance_matrix()

    @classmethod
    def from_json(cls, path: str | Path) -> "TSPInstance":
        """Load a TSP instance from a JSON file."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "cities" not in data:
            raise ValueError("JSON file must contain a 'cities' field.")

        coordinates = np.asarray(
            [city["coordinates"] for city in data["cities"]],
            dtype=float,
        )

        name = data.get("name", path.stem)

        return cls(
            coordinates=coordinates,
            name=name,
        )

    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise Euclidean distances between cities."""

        differences = (
            self.coordinates[:, np.newaxis, :]
            - self.coordinates[np.newaxis, :, :]
        )

        return np.linalg.norm(differences, axis=2)

    def distance(self, city_a: int, city_b: int) -> float:
        """Return the Euclidean distance between two cities."""

        self._validate_city_index(city_a)
        self._validate_city_index(city_b)

        return float(self.distance_matrix[city_a, city_b])

    def _validate_city_index(self, city: int) -> None:
        if not isinstance(city, (int, np.integer)):
            raise TypeError("City index must be an integer.")

        if city < 0 or city >= self.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.num_cities} cities."
            )
