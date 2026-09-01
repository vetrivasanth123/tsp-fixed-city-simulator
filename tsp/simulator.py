from __future__ import annotations

import numpy as np

from .instance import TSPInstance


class TSPSimulator:
    """
    Lightweight simulator for a fixed-city TSP.

    The simulator maintains the state of a tour as cities are selected.
    City coordinates and distances remain fixed throughout an episode.
    """

    def __init__(self, instance: TSPInstance):
        self.instance = instance
        self.n_cities = instance.n_cities

        self.current_city: int | None = None
        self.visited: list[int] = []
        self.total_distance: float = 0.0
        self.done: bool = False

    def reset(self, start_city: int = 0) -> dict:
        """
        Reset the simulator and start a new tour.

        Parameters
        ----------
        start_city:
            City from which the tour begins.

        Returns
        -------
        dict
            Initial simulator state.
        """

        self._validate_city(start_city)

        self.current_city = start_city
        self.visited = [start_city]
        self.total_distance = 0.0
        self.done = False

        return self.get_state()

    def step(self, next_city: int) -> dict:
        """
        Move from the current city to an unvisited city.

        The final step automatically returns to the starting city.

        Parameters
        ----------
        next_city:
            Index of the next city to visit.

        Returns
        -------
        dict
            Updated simulator state.
        """

        if self.done:
            raise RuntimeError("Episode is already complete. Call reset().")

        if self.current_city is None:
            raise RuntimeError("Simulator has not been reset.")

        self._validate_city(next_city)

        if next_city in self.visited:
            raise ValueError(
                f"City {next_city} has already been visited."
            )

        self.total_distance += self.instance.distance_matrix[
            self.current_city, next_city
        ]

        self.current_city = next_city
        self.visited.append(next_city)

        if len(self.visited) == self.n_cities:
            start_city = self.visited[0]

            self.total_distance += self.instance.distance_matrix[
                self.current_city, start_city
            ]

            self.done = True

        return self.get_state()

    def get_state(self) -> dict:
        """Return the current simulator state."""

        return {
            "current_city": self.current_city,
            "visited": self.visited.copy(),
            "unvisited": [
                city
                for city in range(self.n_cities)
                if city not in self.visited
            ],
            "total_distance": self.total_distance,
            "done": self.done,
        }

    def get_tour(self) -> list[int]:
        """
        Return the completed closed tour.

        Raises
        ------
        RuntimeError
            If the tour has not yet been completed.
        """

        if not self.done:
            raise RuntimeError("Tour is not complete.")

        return self.visited + [self.visited[0]]

    def get_tour_length(self) -> float:
        """Return the length of the completed tour."""

        if not self.done:
            raise RuntimeError("Tour is not complete.")

        return self.total_distance

    def _validate_city(self, city: int) -> None:
        """Validate a city index."""

        if not isinstance(city, (int, np.integer)):
            raise TypeError("city must be an integer")

        if not 0 <= city < self.n_cities:
            raise ValueError(
                f"city must be between 0 and {self.n_cities - 1}"
            )
