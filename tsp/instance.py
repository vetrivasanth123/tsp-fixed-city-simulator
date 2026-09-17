
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import euclidean_distance_matrix


class TSPInstance:
    """TSP instance with configurable edge costs, spatial bounds, and city data."""

    def __init__(
        self,
        coordinates: np.ndarray,
        name: str = "tsp_instance",
        cost_matrix: np.ndarray | None = None,
        width: float | None = None,
        height: float | None = None,
        cities: list | None = None,
    ) -> None:
        coordinates = np.asarray(coordinates, dtype=float)

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must be an N x 2 array.")
        if len(coordinates) < 2:
            raise ValueError("A TSP instance must contain at least two cities.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values.")

        if width is not None and width <= 0:
            raise ValueError("width must be positive.")
        if height is not None and height <= 0:
            raise ValueError("height must be positive.")

        self.name = name
        self.coordinates = coordinates
        self.num_cities = len(coordinates)
        self.width = float(width) if width is not None else None
        self.height = float(height) if height is not None else None

        if self.width is not None and np.any(coordinates[:, 0] > self.width):
            raise ValueError("coordinates exceed the specified width.")
        if self.height is not None and np.any(coordinates[:, 1] > self.height):
            raise ValueError("coordinates exceed the specified height.")

        if cities is not None:
            if not isinstance(cities, list):
                raise ValueError("cities must be a list.")
            if len(cities) != self.num_cities:
                raise ValueError(
                    "Number of city records must match number of coordinates."
                )

        self.cities = cities

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

        city_records = data["cities"]

        # Support new city-region format and old fixed-city format.
        if city_records and "center" in city_records[0]:
            coordinates = np.asarray(
                [city["center"] for city in city_records],
                dtype=float,
            )
            cities = city_records
        else:
            coordinates = np.asarray(
                [city["coordinates"] for city in city_records],
                dtype=float,
            )
            cities = None

        return cls(
            coordinates=coordinates,
            name=data.get("name", path.stem),
            cost_matrix=data.get("cost_matrix"),
            width=data.get("width"),
            height=data.get("height"),
            cities=cities,
        )

    def cost(self, city_a: int, city_b: int) -> float:
        self._validate_city_index(city_a)
        self._validate_city_index(city_b)
        return float(self.cost_matrix[city_a, city_b])

    def distance(self, city_a: int, city_b: int) -> float:
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

