from __future__ import annotations

from typing import Any

from .instance import TSPInstance


class TSPSimulator:
    """Deterministic simulator for constructing a TSP tour."""

    def __init__(self, instance: TSPInstance) -> None:
        self.instance = instance
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Reset the simulator to an empty tour."""

        self.tour: list[int] = []
        self.current_city: int | None = None
        self.total_distance: float = 0.0
        self.done: bool = False

        return self.state()

    def step(self, next_city: int) -> dict[str, Any]:
        """
        Visit the next city.

        The first selected city becomes the starting city.
        Subsequent cities add travel distance from the current city.

        Visiting every city does not automatically close the tour.
        Call close_tour() explicitly to return to the starting city.
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

        # First city establishes the starting point.
        if self.current_city is None:
            self.tour.append(next_city)
            self.current_city = next_city

        else:
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

        start_city = self.tour[0]

        # Add the final edge back to the starting city.
        if len(self.tour) > 1:
            self.total_distance += self.instance.distance(
                self.current_city,
                start_city,
            )

        self.done = True

        return self.state()

    def state(self) -> dict[str, Any]:
        """Return the current simulator state."""

        return {
            "tour": list(self.tour),
            "current_city": self.current_city,
            "total_distance": self.total_distance,
            "done": self.done,
            "visited": list(self.tour),
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
