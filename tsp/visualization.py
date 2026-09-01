"""
Visualization utilities for the fixed-city TSP simulator.

This module contains lightweight Matplotlib-based plotting functions.
It does not contain simulation logic, optimization algorithms, or RL code.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .instance import TSPInstance
from .utils import tour_length


def plot_cities(
    instance: TSPInstance,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    title: str = "TSP City Locations",
) -> plt.Axes:

    if ax is None:
        _, ax = plt.subplots()

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=80,
        zorder=3,
    )

    if show_labels:
        for city_index, (x, y) in enumerate(coordinates):
            ax.annotate(
                str(city_index),
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    return ax


def plot_tour(
    instance: TSPInstance,
    tour: Sequence[int],
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    show_distance: bool = True,
    close_tour: bool = True,
    title: str = "TSP Tour",
) -> plt.Axes:

    if ax is None:
        _, ax = plt.subplots()

    tour = list(tour)

    _validate_tour(instance, tour)

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    plot_cities(
        instance,
        ax=ax,
        show_labels=show_labels,
        title=title,
    )

    route = tour.copy()

    if close_tour:
        route.append(tour[0])

    route_coordinates = coordinates[route]

    ax.plot(
        route_coordinates[:, 0],
        route_coordinates[:, 1],
        marker="o",
        linewidth=1.5,
        zorder=2,
    )

    if show_distance:
        distance = tour_length(
            tour,
            instance.distance_matrix,
        )

        ax.set_title(
            f"{title} — Distance: {distance:.4f}"
        )

    return ax


def plot_tour_from_simulator(
    simulator,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    show_distance: bool = True,
    title: str = "TSP Tour",
) -> plt.Axes:

    return plot_tour(
        instance=simulator.instance,
        tour=simulator.tour,
        ax=ax,
        show_labels=show_labels,
        show_distance=show_distance,
        close_tour=simulator.done,
        title=title,
    )


def _validate_tour(
    instance: TSPInstance,
    tour: Iterable[int],
) -> None:

    tour = list(tour)

    num_cities = instance.num_cities

    if len(tour) != num_cities:
        raise ValueError(
            f"Tour must contain exactly {num_cities} cities; "
            f"received {len(tour)}."
        )

    if sorted(tour) != list(range(num_cities)):
        raise ValueError(
            "Tour must contain every city index exactly once."
        )
