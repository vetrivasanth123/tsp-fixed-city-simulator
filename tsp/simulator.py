from __future__ import annotations

from typing import Any
import random

from .instance import TSPInstance


class TSPSimulator:
    """Simulator for constructing tours using edge costs."""

    def __init__(
        self,
        instance: TSPInstance,
        seed: int | None = None,
    ) -> None:
        self.instance = instance
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Start a new episode at a random city."""

        self.start_city = self._rng.randrange(self.instance.num_cities)
        self.tour = [self.start_city]
        self.current_city = self.start_city
        self.total_cost = 0.0
        self.done = False

        return self.state()

    @property
    def total_distance(self) -> float:
        """Backward-compatible alias for total_cost."""
        return self.total_cost

    def available_actions(self) -> list[int]:
        """Return unvisited cities that can be selected."""

        if self.done:
            return []

        visited = set(self.tour)

        return [
            city
            for city in range(self.instance.num_cities)
            if city not in visited
        ]

    def step(self, next_city: int) -> dict[str, Any]:
        """Move to an unvisited city."""

        if self.done:
            raise RuntimeError(
                "Episode is already complete. Call reset()."
            )

        self._validate_city(next_city)

        if next_city in self.tour:
            raise ValueError(
                f"City {next_city} has already been visited."
            )

        self.total_cost += self.instance.cost(
            self.current_city,
            next_city,
        )

        self.tour.append(next_city)
        self.current_city = next_city

        return self.state()

    def close_tour(self) -> dict[str, Any]:
        """Return to the starting city and complete the tour."""

        if not self.tour:
            raise ValueError("Cannot close an empty tour.")

        if not self.done and len(self.tour) > 1:
            self.total_cost += self.instance.cost(
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
            "total_cost": self.total_cost,
            "total_distance": self.total_distance,
            "done": self.done,
        }

    def _validate_city(self, city: int) -> None:
        if not isinstance(city, int):
            raise TypeError("City index must be an integer.")

        if not 0 <= city < self.instance.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.instance.num_cities} cities."
            )
