
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import euclidean_distance_matrix


class TSPInstance:
    """TSP instance with configurable edge costs."""

    def __init__(
        self,
        coordinates: np.ndarray,
        name: str = "tsp_instance",
        cost_matrix: np.ndarray | None = None,
    ) -> None:
        coordinates = np.asarray(coordinates, dtype=float)

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must be an N x 2 array.")
        if len(coordinates) < 2:
            raise ValueError("A TSP instance must contain at least two cities.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values.")

        self.name = name
        self.coordinates = coordinates
        self.num_cities = len(coordinates)

        self.distance_matrix = euclidean_distance_matrix(coordinates)

        if cost_matrix is None:
            cost_matrix = self.distance_matrix.copy()
        else:
            cost_matrix = np.asarray(cost_matrix, dtype=float)

        if cost_matrix.shape != (self.num_cities, self.num_cities):
            raise ValueError("cost_matrix must be N x N.")
        if not np.all(np.isfinite(cost_matrix)):
            raise ValueError("cost_matrix must contain finite values.")
        if np.any(cost_matrix < 0):
            raise ValueError("cost_matrix cannot contain negative values.")

        self.cost_matrix = cost_matrix

    @classmethod
    def from_json(cls, path: str | Path) -> "TSPInstance":
        """Load a TSP instance from JSON."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "cities" not in data or not isinstance(data["cities"], list):
            raise ValueError("JSON file must contain a 'cities' list.")

        coordinates = np.asarray(
            [city["coordinates"] for city in data["cities"]],
            dtype=float,
        )

        cost_matrix = data.get("cost_matrix")

        return cls(
            coordinates=coordinates,
            name=data.get("name", path.stem),
            cost_matrix=cost_matrix,
        )

    def cost(self, city_a: int, city_b: int) -> float:
        """Return the edge cost from city_a to city_b."""

        self._validate_city_index(city_a)
        self._validate_city_index(city_b)

        return float(self.cost_matrix[city_a, city_b])

    def distance(self, city_a: int, city_b: int) -> float:
        """Return the Euclidean distance between two cities."""

        self._validate_city_index(city_a)
        self._validate_city_index(city_b)

        return float(self.distance_matrix[city_a, city_b])

    def _validate_city_index(self, city_index: int) -> None:
        if not isinstance(city_index, (int, np.integer)):
            raise TypeError("city index must be an integer.")

        if not 0 <= city_index < self.num_cities:
            raise IndexError(
                f"City index {city_index} is out of range "
                f"for {self.num_cities} cities."
            )

