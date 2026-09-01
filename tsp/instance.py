from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TSPInstance:
    """Fixed Travelling Salesman Problem instance."""

    def __init__(
        self,
        coordinates: np.ndarray,
        name: str = "tsp_instance",
    ) -> None:
        coordinates = np.asarray(coordinates, dtype=float)

        if coordinates.ndim != 2:
            raise ValueError("coordinates must be a 2D array.")

        if coordinates.shape[1] != 2:
            raise ValueError("Each city must have exactly two coordinates.")

        if coordinates.shape[0] < 2:
            raise ValueError("A TSP instance must contain at least two cities.")

        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values.")

        self.name = name
        self.coordinates = coordinates
        self.num_cities = coordinates.shape[0]

        # Compute the complete Euclidean distance matrix.
        differences = (
            coordinates[:, np.newaxis, :]
            - coordinates[np.newaxis, :, :]
        )

        self.distance_matrix = np.linalg.norm(differences, axis=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "TSPInstance":
        """Load a fixed TSP instance from a JSON file."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "cities" not in data:
            raise ValueError(
                "JSON file must contain a top-level 'cities' field."
            )

        cities = data["cities"]

        if not isinstance(cities, list):
            raise ValueError("'cities' must be a list.")

        coordinates = np.asarray(
            [city["coordinates"] for city in cities],
            dtype=float,
        )

        # Use the JSON filename as the instance name.
        # Example: five_cities.json -> five_cities
        name = path.stem

        return cls(
            coordinates=coordinates,
            name=name,
        )

    def distance(self, city_a: int, city_b: int) -> float:
        """Return the Euclidean distance between two cities."""

        self._validate_city_index(city_a)
        self._validate_city_index(city_b)

        return float(self.distance_matrix[city_a, city_b])

    def _validate_city_index(self, city_index: int) -> None:
        """Validate a city index."""

        if not isinstance(city_index, (int, np.integer)):
            raise TypeError("city index must be an integer.")

        if city_index < 0 or city_index >= self.num_cities:
            raise IndexError(
                f"City index {city_index} is out of range "
                f"for {self.num_cities} cities."
            )
