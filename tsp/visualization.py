"""
Visualization utilities for the fixed-city TSP simulator.

This module provides:
- static city plots,
- static tour plots,
- simulator trajectory visualization,
- animated TSP simulation visualization.

It contains no optimization or RL logic.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from .instance import TSPInstance
from .utils import tour_length


def plot_cities(
    instance: TSPInstance,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    title: str = "TSP City Locations",
) -> plt.Axes:
    """
    Plot the fixed city locations.
    """

    if ax is None:
        _, ax = plt.subplots()

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=100,
        zorder=3,
    )

    if show_labels:
        for city_index, (x, y) in enumerate(
            coordinates
        ):
            ax.annotate(
                str(city_index),
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

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
    """
    Plot a completed TSP tour.
    """

    if ax is None:
        _, ax = plt.subplots()

    tour = list(tour)

    _validate_tour(
        instance,
        tour,
    )

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
    """
    Plot the current tour stored by a TSPSimulator.
    """

    return plot_tour(
        instance=simulator.instance,
        tour=simulator.tour,
        ax=ax,
        show_labels=show_labels,
        show_distance=show_distance,
        close_tour=simulator.done,
        title=title,
    )


def animate_simulation(
    simulator,
    interval: int = 1200,
    show_labels: bool = True,
    title: str = "Fixed-City TSP Simulation",
):
    """
    Animate the exact trajectory recorded by the simulator.

    The simulator must already have executed its actions.

    The animation reproduces:
        1. starting city selection,
        2. available actions,
        3. selected city,
        4. route connection,
        5. accumulated distance,
        6. final return to the start.

    Parameters
    ----------
    simulator:
        TSPSimulator containing a recorded trajectory.

    interval:
        Delay between animation frames in milliseconds.

    show_labels:
        Whether to display city labels.

    title:
        Base animation title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object.
    """

    history = simulator.trajectory()

    if not history:
        raise ValueError(
            "Simulator contains no recorded trajectory."
        )

    instance = simulator.instance

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # --------------------------------------------------
    # Determine plotting limits
    # --------------------------------------------------

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    x_range = max(
        np.ptp(x),
        1.0,
    )

    y_range = max(
        np.ptp(y),
        1.0,
    )

    margin_x = 0.15 * x_range
    margin_y = 0.15 * y_range

    ax.set_xlim(
        x.min() - margin_x,
        x.max() + margin_x,
    )

    ax.set_ylim(
        y.min() - margin_y,
        y.max() + margin_y,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------
    # Fixed city locations
    # --------------------------------------------------

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=100,
        zorder=3,
    )

    if show_labels:
        for city_index, (cx, cy) in enumerate(
            coordinates
        ):
            ax.annotate(
                str(city_index),
                (cx, cy),
                xytext=(7, 7),
                textcoords="offset points",
            )

    # --------------------------------------------------
    # Dynamic graphical elements
    # --------------------------------------------------

    route_line, = ax.plot(
        [],
        [],
        linewidth=2.0,
        zorder=2,
    )

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=16,
        linestyle="None",
        zorder=5,
    )

    available_scatter = ax.scatter(
        [],
        [],
        s=180,
        marker="o",
        zorder=4,
    )

    selected_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=20,
        linestyle="None",
        zorder=6,
    )

    # --------------------------------------------------
    # Initialize
    # --------------------------------------------------

    def init():
        route_line.set_data(
            [],
            [],
        )

        current_marker.set_data(
            [],
            [],
        )

        available_scatter.set_offsets(
            np.empty((0, 2))
        )

        selected_marker.set_data(
            [],
            [],
        )

        ax.set_title(title)

        return (
            route_line,
            current_marker,
            available_scatter,
            selected_marker,
        )

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------

    def update(frame: int):

        record = history[frame]

        tour = record["tour"]

        current_city = record[
            "current_city"
        ]

        available_actions = record[
            "available_actions"
        ]

        action = record[
            "action"
        ]

        event = record[
            "event"
        ]

        # ----------------------------------------------
        # Draw current route
        # ----------------------------------------------

        route_coordinates = coordinates[
            tour
        ]

        route_line.set_data(
            route_coordinates[:, 0],
            route_coordinates[:, 1],
        )

        # ----------------------------------------------
        # Current city
        # ----------------------------------------------

        if current_city is not None:

            current_marker.set_data(
                [coordinates[current_city, 0]],
                [coordinates[current_city, 1]],
            )

        # ----------------------------------------------
        # Available actions
        # ----------------------------------------------

        if available_actions:

            available_coordinates = coordinates[
                available_actions
            ]

            available_scatter.set_offsets(
                available_coordinates
            )

        else:

            available_scatter.set_offsets(
                np.empty((0, 2))
            )

        # ----------------------------------------------
        # Selected action
        # ----------------------------------------------

        if (
            action is not None
            and event == "step"
        ):

            selected_marker.set_data(
                [coordinates[action, 0]],
                [coordinates[action, 1]],
            )

        else:

            selected_marker.set_data(
                [],
                [],
            )

        # ----------------------------------------------
        # Text information
        # ----------------------------------------------

        if event == "start":

            ax.set_title(
                f"{title}\n"
                f"Starting city: {record['current_city']}"
            )

        elif event == "step":

            ax.set_title(
                f"{title}\n"
                f"Action: City {action}  |  "
                f"Current city: {current_city}  |  "
                f"Distance: "
                f"{record['total_distance']:.4f}"
            )

        elif event == "close":

            ax.set_title(
                f"{title}\n"
                f"Tour complete — "
                f"Returned to City {record['current_city']}"
                f"  |  "
                f"Total distance: "
                f"{record['total_distance']:.4f}"
            )

        return (
            route_line,
            current_marker,
            available_scatter,
            selected_marker,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        init_func=init,
        interval=interval,
        blit=False,
        repeat=False,
    )

    return animation


def _validate_tour(
    instance: TSPInstance,
    tour: Iterable[int],
) -> None:
    """
    Validate that a tour contains every city exactly once.
    """

    tour = list(tour)

    num_cities = instance.num_cities

    if len(tour) != num_cities:
        raise ValueError(
            f"Tour must contain exactly "
            f"{num_cities} cities; "
            f"received {len(tour)}."
        )

    if sorted(tour) != list(
        range(num_cities)
    ):
        raise ValueError(
            "Tour must contain every city "
            "index exactly once."
        )
