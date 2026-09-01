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
    - records the trajectory for visualization and later RL use.

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
        Reset the simulator and randomly select a starting city.

        Parameters
        ----------
        seed:
            Optional seed. If supplied, the simulator's random
            generator is reseeded before selecting the start city.

        Returns
        -------
        dict
            Initial simulator state.
        """

        if seed is not None:
            self._rng.seed(seed)
            self._seed = seed

        self.start_city = self._rng.randrange(
            self.instance.num_cities
        )

        self.tour: list[int] = [self.start_city]

        self.current_city: int = self.start_city

        self.total_distance: float = 0.0

        self.done: bool = False

        # Record the complete trajectory.
        #
        # This is useful for visualization because the visualizer
        # can reproduce exactly what the simulator did.
        self.history: list[dict[str, Any]] = []

        self._record_history(
            action=None,
            event="start",
        )

        return self.state()

    def available_actions(self) -> list[int]:
        """
        Return the cities that can currently be selected.

        Previously visited cities are excluded.

        Returns
        -------
        list[int]
            Valid next-city actions.
        """

        if self.done:
            return []

        visited = set(self.tour)

        return [
            city
            for city in range(self.instance.num_cities)
            if city not in visited
        ]

    def step(
        self,
        next_city: int,
    ) -> dict[str, Any]:
        """
        Select the next city.

        Parameters
        ----------
        next_city:
            City index selected as the next action.

        Returns
        -------
        dict
            Updated simulator state.

        Raises
        ------
        RuntimeError
            If the episode is already complete.

        ValueError
            If the city was already visited.

        IndexError
            If the city index is invalid.
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

        previous_city = self.current_city

        distance_added = self.instance.distance(
            previous_city,
            next_city,
        )

        self.total_distance += distance_added

        self.tour.append(next_city)

        self.current_city = next_city

        self._record_history(
            action=next_city,
            event="step",
            previous_city=previous_city,
            distance_added=distance_added,
        )

        return self.state()

    def close_tour(self) -> dict[str, Any]:
        """
        Return to the starting city and complete the tour.

        The closing edge is recorded in the trajectory.

        Returns
        -------
        dict
            Final simulator state.
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
            action=self.start_city,
            event="close",
            previous_city=previous_city,
            distance_added=distance_added,
        )

        return self.state()

    def state(self) -> dict[str, Any]:
        """
        Return the current simulator state.

        This dictionary is intentionally suitable for a future
        RL agent interface.
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
        Return a copy of the recorded simulator trajectory.

        Each entry contains the state transition information needed
        to reproduce the simulation visually.
        """

        return [
            dict(record)
            for record in self.history
        ]

    def _record_history(
        self,
        action: int | None,
        event: str,
        previous_city: int | None = None,
        distance_added: float = 0.0,
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

        if city < 0 or city >= self.instance.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.instance.num_cities} cities."
            )
