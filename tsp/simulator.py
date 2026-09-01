```python
from __future__ import annotations

from typing import Any

import numpy as np

from .instance import TSPInstance


class TSPSimulator:
    """
    Simulator for constructing tours on a fixed TSP instance.

    The city locations are fixed by the supplied TSPInstance.

    At reset(), a starting city is selected randomly. The simulator
    then exposes the valid next-city actions through available_actions().
    An external policy, algorithm, or RL agent can choose one of those
    actions and pass it to step().

    The simulator itself does not choose subsequent cities.
    """

    def __init__(self, instance: TSPInstance) -> None:
        self.instance = instance
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """
        Reset the simulator and randomly select a starting city.

        Parameters
        ----------
        seed:
            Optional random seed. Supplying the same seed produces the
            same starting city, which is useful for testing.

        Returns
        -------
        dict
            Initial simulator state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.instance.num_cities <= 0:
            raise ValueError("TSP instance must contain at least one city.")

        start_city = int(
            self._rng.integers(0, self.instance.num_cities)
        )

        self.tour: list[int] = [start_city]
        self.current_city: int = start_city
        self.total_distance: float = 0.0
        self.done: bool = False

        return self.state()

    def available_actions(self) -> list[int]:
        """
        Return the cities that can be selected as the next action.

        The current/visited cities are excluded.

        Returns
        -------
        list[int]
            Unvisited city indices.

        Notes
        -----
        If all cities have been visited, an empty list is returned.
        The final return to the starting city is handled by close_tour()
        rather than appearing as a normal action.
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
        Visit the selected next city.

        The first city is selected automatically during reset().
        Subsequent calls add travel distance from the current city.

        Parameters
        ----------
        next_city:
            Index of an unvisited city.

        Returns
        -------
        dict
            Updated simulator state.

        Raises
        ------
        RuntimeError
            If the episode has already been completed.
        ValueError
            If the city has already been visited.
        IndexError
            If the city index is outside the instance.
        TypeError
            If the city index is not an integer.
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

        For a multi-city tour, the distance from the final city back
        to the starting city is added exactly once.

        Returns
        -------
        dict
            Final simulator state.
        """
        if not self.tour:
            raise ValueError("Cannot close an empty tour.")

        if self.done:
            return self.state()

        start_city = self.tour[0]

        if len(self.tour) > 1:
            self.total_distance += self.instance.distance(
                self.current_city,
                start_city,
            )

        self.done = True

        return self.state()

    def state(self) -> dict[str, Any]:
        """
        Return the current simulator state.

        Returns
        -------
        dict
            Current tour, city, distance, visited cities,
            available actions, and completion status.
        """
        return {
            "tour": list(self.tour),
            "current_city": self.current_city,
            "total_distance": self.total_distance,
            "done": self.done,
            "visited": list(self.tour),
            "available_actions": self.available_actions(),
        }

    def _validate_city(self, city: int) -> None:
        """Validate a city index."""

        if not isinstance(city, (int, np.integer)):
            raise TypeError("City index must be an integer.")

        if city < 0 or city >= self.instance.num_cities:
            raise IndexError(
                f"City index {city} is out of range for "
                f"{self.instance.num_cities} cities."
            )
```
