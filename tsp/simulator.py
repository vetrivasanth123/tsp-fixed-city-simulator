from __future__ import annotations

from typing import Any

import random

from .instance import TSPInstance


class TSPSimulator:
    """
    Simulator for constructing TSP tours on a fixed set of cities.

    The simulator:
    - uses a fixed set of city coordinates,
    - randomly selects a starting city,
    - exposes valid next-city actions,
    - accepts one action at a time,
    - tracks the partial tour and distance,
    - explicitly closes the tour,
    - records every transition for visualization,
    - can later be used as the environment interface for RL.

    No optimization or RL logic is included here.
    """

    def __init__(
        self,
        instance: TSPInstance,
        seed: int | None = None,
    ) -> None:
        self.instance = instance
        self._rng = random.Random(seed)
        self._seed = seed

        self.reset()

    def reset(
        self,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Reset the simulator.

        If seed is supplied, the random generator is reseeded.
        Otherwise, the existing random generator continues.

        A starting city is selected randomly.
        """

        if seed is not None:
            self._rng.seed(seed)
            self._seed = seed

        self.start_city = self._rng.randrange(
            self.instance.num_cities
        )

        self.tour: list[int] = [
            self.start_city
        ]

        self.current_city: int = self.start_city

        self.total_distance: float = 0.0

        self.done: bool = False

        # Complete simulator trajectory.
        self.history: list[dict[str, Any]] = []

        self._record_history(
            event="start",
            action=None,
            previous_city=None,
            distance_added=0.0,
        )

        return self.state()

    def available_actions(self) -> list[int]:
        """
        Return all currently valid next-city actions.

        Previously visited cities are excluded.
        """

        if self.done:
            return []

        visited = set(self.tour)

        return [
            city
            for city in range(
                self.instance.num_cities
            )
            if city not in visited
        ]

    def step(
        self,
        next_city: int,
    ) -> dict[str, Any]:
        """
        Move from the current city to an unvisited city.

        The action is recorded in the simulator trajectory.
        """

        if self.done:
            raise RuntimeError(
                "Episode is already complete. "
                "Call reset()."
            )

        self._validate_city(next_city)

        if next_city in self.tour:
            raise ValueError(
                f"City {next_city} has already been visited."
            )

        previous_city = self.current_city

        distance_added = self.instance.distance(
            previous_city,
            next_city,
        )

        self.total_distance += distance_added

        self.tour.append(next_city)

        self.current_city = next_city

        self._record_history(
            event="step",
            action=next_city,
            previous_city=previous_city,
            distance_added=distance_added,
        )

        return self.state()

    def close_tour(self) -> dict[str, Any]:
        """
        Return to the starting city and complete the tour.
        """

        if not self.tour:
            raise ValueError(
                "Cannot close an empty tour."
            )

        if self.done:
            return self.state()

        previous_city = self.current_city

        distance_added = 0.0

        if len(self.tour) > 1:

            distance_added = self.instance.distance(
                previous_city,
                self.start_city,
            )

            self.total_distance += distance_added

        self.done = True

        self._record_history(
            event="close",
            action=self.start_city,
            previous_city=previous_city,
            distance_added=distance_added,
        )

        return self.state()

    def state(self) -> dict[str, Any]:
        """
        Return the current simulator state.

        This structure is intentionally suitable for
        future RL integration.
        """

        return {
            "tour": list(self.tour),
            "start_city": self.start_city,
            "current_city": self.current_city,
            "visited": list(self.tour),
            "available_actions": self.available_actions(),
            "total_distance": self.total_distance,
            "done": self.done,
        }

    def trajectory(self) -> list[dict[str, Any]]:
        """
        Return the complete simulator trajectory.

        The visualization uses this trajectory to reproduce
        exactly what happened during simulation.
        """

        return [
            dict(record)
            for record in self.history
        ]

    def _record_history(
        self,
        event: str,
        action: int | None,
        previous_city: int | None,
        distance_added: float,
    ) -> None:
        """
        Record one simulator event.
        """

        self.history.append(
            {
                "event": event,
                "action": action,
                "previous_city": previous_city,
                "current_city": self.current_city,
                "tour": list(self.tour),
                "visited": list(self.tour),
                "available_actions": self.available_actions(),
                "total_distance": self.total_distance,
                "distance_added": distance_added,
                "done": self.done,
            }
        )

    def _validate_city(
        self,
        city: int,
    ) -> None:
        """Validate a city index."""

        if not isinstance(city, int):
            raise TypeError(
                "City index must be an integer."
            )

        if (
            city < 0
            or city >= self.instance.num_cities
        ):
            raise IndexError(
                f"City index {city} is out of range "
                f"for {self.instance.num_cities} cities."
            )
