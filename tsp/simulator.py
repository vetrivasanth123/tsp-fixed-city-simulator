from __future__ import annotations

from typing import Any

import random

from .instance import TSPInstance


class TSPSimulator:
    """Simulator for constructing TSP tours on a fixed set of cities."""

    def __init__(
        self,
        instance: TSPInstance,
        seed: int | None = None,
    ) -> None:
        self.instance = instance
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> dict[str, Any]:
        """
        Reset the simulator and randomly select a starting city.

        Returns
        -------
        dict
            Initial simulator state.
        """
        self.start_city = self._rng.randrange(self.instance.num_cities)

        self.tour: list[int] = [self.start_city]
        self.current_city: int = self.start_city
        self.total_distance: float = 0.0
        self.done: bool = False

        return self.state()

    def available_actions(self) -> list[int]:
        """
        Return the cities that can be selected next.

        The starting city and all previously visited cities are excluded
        until the tour is closed.
        """
        if self.done:
            return []

        visited = set(self.tour)

        return [
            city
            for city in range(self.instance.num_cities)
            if city not in visited
        ]

    def step(self, next_city: int) -> dict[str, Any]:
        """
        Move from the current city to an unvisited city.

        The simulator does not automatically close the tour. Once all
        cities have been visited, call close_tour() explicitly.
        """
        if self.done:
            raise RuntimeError(
                "Episode is already complete. Call reset()."
            )

        self._validate_city(next_city)

        if next_city in self.tour:
            raise ValueError(
                f"City {next_city} has already been visited."
            )

        self.total_distance += self.instance.distance(
            self.current_city,
            next_city,
        )

        self.tour.append(next_city)
        self.current_city = next_city

        return self.state()

    def close_tour(self) -> dict[str, Any]:
        """
        Return to the starting city and complete the tour.

        The tour must contain at least one city.
        """
        if not self.tour:
            raise ValueError("Cannot close an empty tour.")

        if self.done:
            return self.state()

        if len(self.tour) > 1:
            self.total_distance += self.instance.distance(
                self.current_city,
                self.start_city,
            )

        self.done = True

        return self.state()

    def state(self) -> dict[str, Any]:
        """Return the current simulator state."""
        return {
            "tour": list(self.tour),
            "start_city": self.start_city,
            "current_city": self.current_city,
            "visited": list(self.tour),
            "available_actions": self.available_actions(),
            "total_distance": self.total_distance,
            "done": self.done,
        }

    def _validate_city(self, city: int) -> None:
        """Validate a city index."""
        if not isinstance(city, int):
            raise TypeError("City index must be an integer.")

        if city < 0 or city >= self.instance.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.instance.num_cities} cities."
            )
