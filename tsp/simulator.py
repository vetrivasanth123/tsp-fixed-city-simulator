from __future__ import annotations

from typing import Any
import random

from .instance import TSPInstance

_LAST_SIMULATOR = None


def get_last_simulator():
    """Return the most recently executed simulator."""
    return _LAST_SIMULATOR


class TSPSimulator:
    """Simulator for constructing TSP tours on fixed cities."""

    def __init__(
        self,
        instance: TSPInstance,
        seed: int | None = None,
    ) -> None:
        global _LAST_SIMULATOR

        self.instance = instance
        self._rng = random.Random(seed)
        _LAST_SIMULATOR = self
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Reset and randomly select the starting city."""

        self.start_city = self._rng.randrange(
            self.instance.num_cities
        )
        self.tour = [self.start_city]
        self.current_city = self.start_city
        self.total_distance = 0.0
        self.done = False

        self.history = [{
            "tour": list(self.tour),
            "current_city": self.current_city,
            "action": None,
            "distance": self.total_distance,
            "done": False,
        }]

        return self.state()

    def available_actions(self) -> list[int]:
        """Return unvisited cities."""

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

        self.total_distance += self.instance.distance(
            self.current_city,
            next_city,
        )

        self.tour.append(next_city)
        self.current_city = next_city

        self.history.append({
            "tour": list(self.tour),
            "current_city": self.current_city,
            "action": next_city,
            "distance": self.total_distance,
            "done": False,
        })

        return self.state()

    def close_tour(self) -> dict[str, Any]:
        """Return to the start and complete the tour."""

        if not self.tour:
            raise ValueError("Cannot close an empty tour.")

        if self.done:
            return self.state()

        self.total_distance += self.instance.distance(
            self.current_city,
            self.start_city,
        )

        self.done = True

        self.history.append({
            "tour": list(self.tour),
            "current_city": self.current_city,
            "action": self.start_city,
            "distance": self.total_distance,
            "done": True,
        })

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
        if not isinstance(city, int):
            raise TypeError("City index must be an integer.")

        if city < 0 or city >= self.instance.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.instance.num_cities} cities."
            )
