"""
Visualization utilities for the fixed-city TSP simulator.

The animation reproduces the actual simulator trajectory.

For every action it shows:

    current city
          ↓
    selected action
          ↓
    new current city
          ↓
    updated route

The final frame shows the completed closed tour.

No optimization or RL logic is contained here.
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
    """Plot the fixed city locations."""

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
    """Plot a complete TSP tour."""

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
        linewidth=2.0,
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
    """Plot the current simulator tour."""

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
    interval: int = 1500,
    show_labels: bool = True,
    title: str = "Fixed-City TSP Simulation",
):
    """
    Animate the exact trajectory produced by the simulator.

    The animation contains one visual decision frame for every
    simulator action.

    The displayed information is:

        Current city
        Available actions
        Selected action
        Distance added
        Total distance

    The route is updated only after the corresponding action
    has been selected.

    Parameters
    ----------
    simulator:
        A TSPSimulator that has already executed its trajectory.

    interval:
        Time between animation frames in milliseconds.

    show_labels:
        Whether to show city indices.

    title:
        Animation title.

    Returns
    -------
    FuncAnimation
        Matplotlib animation object.
    """

    history = simulator.trajectory()

    if not history:
        raise ValueError(
            "Simulator contains no trajectory."
        )

    instance = simulator.instance

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    num_cities = instance.num_cities

    # --------------------------------------------------
    # Build explicit visual frames
    # --------------------------------------------------
    #
    # We do NOT simply use the raw history as frames.
    #
    # Each simulator action produces TWO visual states:
    #
    #   1. decision:
    #      current city + selected city
    #
    #   2. transition:
    #      selected city becomes current city
    #
    # This makes the action-selection process visible.
    # --------------------------------------------------

    frames = []

    for record in history:

        event = record["event"]

        if event == "start":

            frames.append(
                {
                    "type": "start",
                    "current_city": record[
                        "current_city"
                    ],
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": list(
                        record["available_actions"]
                    ),
                    "action": None,
                    "total_distance": record[
                        "total_distance"
                    ],
                }
            )

        elif event == "step":

            previous_city = record[
                "previous_city"
            ]

            action = record["action"]

            # ------------------------------------------
            # Decision frame
            # ------------------------------------------

            frames.append(
                {
                    "type": "decision",
                    "current_city": previous_city,
                    "tour": list(
                        record["tour"][:-1]
                    ),
                    "available_actions": list(
                        record[
                            "available_actions"
                        ]
                    ),
                    "action": action,
                    "total_distance": (
                        record["total_distance"]
                        - record["distance_added"]
                    ),
                }
            )

            # ------------------------------------------
            # Transition/result frame
            # ------------------------------------------

            frames.append(
                {
                    "type": "transition",
                    "current_city": record[
                        "current_city"
                    ],
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": list(
                        record[
                            "available_actions"
                        ]
                    ),
                    "action": action,
                    "total_distance": record[
                        "total_distance"
                    ],
                }
            )

        elif event == "close":

            frames.append(
                {
                    "type": "close",
                    "current_city": record[
                        "previous_city"
                    ],
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": [],
                    "action": record[
                        "action"
                    ],
                    "total_distance": (
                        record["total_distance"]
                        - record["distance_added"]
                    ),
                }
            )

            # ------------------------------------------
            # Final completed frame
            # ------------------------------------------

            frames.append(
                {
                    "type": "complete",
                    "current_city": record[
                        "previous_city"
                    ],
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": [],
                    "action": record[
                        "action"
                    ],
                    "total_distance": record[
                        "total_distance"
                    ],
                }
            )

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

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

    margin_x = 0.20 * x_range
    margin_y = 0.20 * y_range

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
    # Fixed cities
    # --------------------------------------------------

    ax.scatter(
        x,
        y,
        s=100,
        zorder=3,
    )

    if show_labels:

        for city_index, (
            cx,
            cy,
        ) in enumerate(coordinates):

            ax.annotate(
                str(city_index),
                (cx, cy),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
            )

    # --------------------------------------------------
    # Dynamic route
    # --------------------------------------------------

    route_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        zorder=2,
    )

    # --------------------------------------------------
    # Current-city marker
    # --------------------------------------------------

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=18,
        linestyle="None",
        zorder=6,
    )

    # --------------------------------------------------
    # Selected-action marker
    # --------------------------------------------------

    selected_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=22,
        linestyle="None",
        zorder=7,
    )

    # --------------------------------------------------
    # Start-city marker
    # --------------------------------------------------

    start_marker, = ax.plot(
        [],
        [],
        marker="s",
        markersize=15,
        linestyle="None",
        zorder=5,
    )

    # --------------------------------------------------
    # Available action markers
    # --------------------------------------------------

    available_marker = ax.scatter(
        [],
        [],
        s=160,
        marker="o",
        zorder=4,
    )

    # --------------------------------------------------
    # Helper for current city
    # --------------------------------------------------

    def set_marker(
        marker,
        city,
    ):

        if city is None:

            marker.set_data(
                [],
                [],
            )

            return

        marker.set_data(
            [coordinates[city, 0]],
            [coordinates[city, 1]],
        )

    # --------------------------------------------------
    # Initialization
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

        selected_marker.set_data(
            [],
            [],
        )

        start_marker.set_data(
            [],
            [],
        )

        available_marker.set_offsets(
            np.empty(
                (0, 2)
            )
        )

        ax.set_title(title)

        return (
            route_line,
            current_marker,
            selected_marker,
            start_marker,
            available_marker,
        )

    # --------------------------------------------------
    # Frame update
    # --------------------------------------------------

    def update(frame_index):

        frame = frames[
            frame_index
        ]

        frame_type = frame[
            "type"
        ]

        current_city = frame[
            "current_city"
        ]

        tour = frame[
            "tour"
        ]

        available_actions = frame[
            "available_actions"
        ]

        action = frame[
            "action"
        ]

        total_distance = frame[
            "total_distance"
        ]

        # ----------------------------------------------
        # Route
        # ----------------------------------------------

        if len(tour) >= 1:

            route_coordinates = coordinates[
                tour
            ]

            route_line.set_data(
                route_coordinates[:, 0],
                route_coordinates[:, 1],
            )

        else:

            route_line.set_data(
                [],
                [],
            )

        # ----------------------------------------------
        # Current city
        # ----------------------------------------------

        set_marker(
            current_marker,
            current_city,
        )

        # ----------------------------------------------
        # Start city
        # ----------------------------------------------

        start_city = simulator.start_city

        set_marker(
            start_marker,
            start_city,
        )

        # ----------------------------------------------
        # Available actions
        # ----------------------------------------------

        if available_actions:

            available_coordinates = coordinates[
                available_actions
            ]

            available_marker.set_offsets(
                available_coordinates
            )

        else:

            available_marker.set_offsets(
                np.empty(
                    (0, 2)
                )
            )

        # ----------------------------------------------
        # Selected action
        # ----------------------------------------------

        if (
            frame_type == "decision"
            or frame_type == "close"
        ):

            set_marker(
                selected_marker,
                action,
            )

        else:

            selected_marker.set_data(
                [],
                [],
            )

        # ----------------------------------------------
        # Titles
        # ----------------------------------------------

        if frame_type == "start":

            ax.set_title(
                f"{title}\n"
                f"START → City {current_city}\n"
                f"Available actions: "
                f"{available_actions}"
            )

        elif frame_type == "decision":

            ax.set_title(
                f"{title}\n"
                f"Current City: {current_city}   "
                f"→   Selected Action: City {action}\n"
                f"Available actions: "
                f"{available_actions}   |   "
                f"Distance: {total_distance:.4f}"
            )

        elif frame_type == "transition":

            ax.set_title(
                f"{title}\n"
                f"Moved to City {current_city}\n"
                f"Tour: {tour}   |   "
                f"Distance: {total_distance:.4f}"
            )

        elif frame_type == "close":

            ax.set_title(
                f"{title}\n"
                f"Closing tour: "
                f"City {current_city} → "
                f"City {action}\n"
                f"Current distance: "
                f"{total_distance:.4f}"
            )

        elif frame_type == "complete":

            ax.set_title(
                f"{title}\n"
                f"✓ COMPLETE TOUR\n"
                f"Tour: {tour} → {simulator.start_city}\n"
                f"Total distance: "
                f"{total_distance:.4f}"
            )

        return (
            route_line,
            current_marker,
            selected_marker,
            start_marker,
            available_marker,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return animation


def _validate_tour(
    instance: TSPInstance,
    tour: Iterable[int],
) -> None:
    """Validate that a tour contains every city exactly once."""

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
