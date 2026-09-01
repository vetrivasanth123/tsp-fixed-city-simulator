"""
Visualization utilities for the fixed-city TSP simulator.

The animation reproduces the actual simulator trajectory.

For each simulator action the visualization shows:

    1. Current city
    2. Available actions
    3. Selected next city
    4. Proposed movement
    5. Completed movement
    6. Updated tour

The final frame shows the complete closed tour.

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
    Animate the exact trajectory produced by TSPSimulator.

    The simulator must already contain a completed trajectory.

    Every action is visualized as:

        DECISION
        current city
              ↓
        selected city
              ↓
        TRANSITION
        edge becomes part of route

    The animation does NOT choose any new actions.

    It only visualizes what the simulator actually did.
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

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # --------------------------------------------------
    # Build visual frames
    # --------------------------------------------------

    frames = []

    for record in history:

        event = record["event"]

        # ----------------------------------------------
        # START
        # ----------------------------------------------

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
                        record[
                            "available_actions"
                        ]
                    ),
                    "action": None,
                    "previous_city": None,
                    "distance_added": 0.0,
                    "total_distance": record[
                        "total_distance"
                    ],
                }
            )

        # ----------------------------------------------
        # NORMAL ACTION
        # ----------------------------------------------

        elif event == "step":

            previous_city = record[
                "previous_city"
            ]

            action = record["action"]

            distance_before = (
                record["total_distance"]
                - record["distance_added"]
            )

            # ------------------------------------------
            # DECISION FRAME
            # ------------------------------------------

            frames.append(
                {
                    "type": "decision",
                    "current_city": previous_city,
                    "previous_city": previous_city,
                    "tour": list(
                        record["tour"][:-1]
                    ),
                    "available_actions": list(
                        record[
                            "available_actions"
                        ]
                    ),
                    "action": action,
                    "distance_added": record[
                        "distance_added"
                    ],
                    "total_distance": distance_before,
                }
            )

            # ------------------------------------------
            # TRANSITION FRAME
            # ------------------------------------------

            frames.append(
                {
                    "type": "transition",
                    "current_city": record[
                        "current_city"
                    ],
                    "previous_city": previous_city,
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": list(
                        record[
                            "available_actions"
                        ]
                    ),
                    "action": action,
                    "distance_added": record[
                        "distance_added"
                    ],
                    "total_distance": record[
                        "total_distance"
                    ],
                }
            )

        # ----------------------------------------------
        # CLOSE TOUR
        # ----------------------------------------------

        elif event == "close":

            previous_city = record[
                "previous_city"
            ]

            action = record[
                "action"
            ]

            distance_before = (
                record["total_distance"]
                - record["distance_added"]
            )

            # ------------------------------------------
            # CLOSING DECISION
            # ------------------------------------------

            frames.append(
                {
                    "type": "close_decision",
                    "current_city": previous_city,
                    "previous_city": previous_city,
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": [],
                    "action": action,
                    "distance_added": record[
                        "distance_added"
                    ],
                    "total_distance": distance_before,
                }
            )

            # ------------------------------------------
            # COMPLETE
            # ------------------------------------------

            frames.append(
                {
                    "type": "complete",
                    "current_city": action,
                    "previous_city": previous_city,
                    "tour": list(
                        record["tour"]
                    ),
                    "available_actions": [],
                    "action": action,
                    "distance_added": record[
                        "distance_added"
                    ],
                    "total_distance": record[
                        "total_distance"
                    ],
                }
            )

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    x_range = max(
        np.ptp(x),
        1.0,
    )

    y_range = max(
        np.ptp(y),
        1.0,
    )

    margin_x = 0.25 * x_range
    margin_y = 0.25 * y_range

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
        s=120,
        zorder=3,
    )

    if show_labels:

        for city_index, (
            cx,
            cy,
        ) in enumerate(coordinates):

            ax.annotate(
                f"City {city_index}",
                (cx, cy),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
            )

    # --------------------------------------------------
    # Completed route
    # --------------------------------------------------

    route_line, = ax.plot(
        [],
        [],
        marker="o",
        linewidth=2.5,
        zorder=2,
    )

    # --------------------------------------------------
    # Proposed action edge
    # --------------------------------------------------

    proposed_line, = ax.plot(
        [],
        [],
        linestyle="--",
        linewidth=2.0,
        zorder=1,
    )

    # --------------------------------------------------
    # Current city marker
    # --------------------------------------------------

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=18,
        linestyle="None",
        zorder=7,
    )

    # --------------------------------------------------
    # Selected city marker
    # --------------------------------------------------

    selected_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=23,
        linestyle="None",
        zorder=8,
    )

    # --------------------------------------------------
    # Start city marker
    # --------------------------------------------------

    start_marker, = ax.plot(
        [],
        [],
        marker="s",
        markersize=14,
        linestyle="None",
        zorder=6,
    )

    # --------------------------------------------------
    # Available actions
    # --------------------------------------------------

    available_marker = ax.scatter(
        [],
        [],
        s=180,
        marker="o",
        zorder=4,
    )

    # --------------------------------------------------
    # Marker helper
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

        proposed_line.set_data(
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
            np.empty((0, 2))
        )

        ax.set_title(title)

        return (
            route_line,
            proposed_line,
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

        previous_city = frame[
            "previous_city"
        ]

        action = frame[
            "action"
        ]

        tour = frame[
            "tour"
        ]

        available_actions = frame[
            "available_actions"
        ]

        total_distance = frame[
            "total_distance"
        ]

        # ----------------------------------------------
        # Completed route
        # ----------------------------------------------

        if len(tour) >= 2:

            route_coordinates = coordinates[
                tour
            ]

            route_line.set_data(
                route_coordinates[:, 0],
                route_coordinates[:, 1],
            )

        elif len(tour) == 1:

            route_line.set_data(
                [coordinates[tour[0], 0]],
                [coordinates[tour[0], 1]],
            )

        else:

            route_line.set_data(
                [],
                [],
            )

        # ----------------------------------------------
        # Proposed edge
        # ----------------------------------------------

        if (
            frame_type
            in (
                "decision",
                "close_decision",
            )
            and previous_city is not None
            and action is not None
        ):

            proposed_line.set_data(
                [
                    coordinates[
                        previous_city,
                        0,
                    ],
                    coordinates[
                        action,
                        0,
                    ],
                ],
                [
                    coordinates[
                        previous_city,
                        1,
                    ],
                    coordinates[
                        action,
                        1,
                    ],
                ],
            )

        else:

            proposed_line.set_data(
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

        set_marker(
            start_marker,
            simulator.start_city,
        )

        # ----------------------------------------------
        # Available cities
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
                np.empty((0, 2))
            )

        # ----------------------------------------------
        # Selected action
        # ----------------------------------------------

        if (
            frame_type
            in (
                "decision",
                "close_decision",
            )
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
        # Text
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
                f"Current City {current_city}"
                f"  →  "
                f"SELECTED City {action}\n"
                f"Available actions: "
                f"{available_actions}\n"
                f"Distance so far: "
                f"{total_distance:.4f}"
            )

        elif frame_type == "transition":

            ax.set_title(
                f"{title}\n"
                f"Moved: City "
                f"{previous_city}"
                f" → City "
                f"{current_city}\n"
                f"Edge distance: "
                f"{frame['distance_added']:.4f}  |  "
                f"Total distance: "
                f"{total_distance:.4f}\n"
                f"Tour: {tour}"
            )

        elif frame_type == "close_decision":

            ax.set_title(
                f"{title}\n"
                f"CLOSING TOUR\n"
                f"City {previous_city}"
                f"  →  "
                f"Start City {action}\n"
                f"Closing distance: "
                f"{frame['distance_added']:.4f}\n"
                f"Distance so far: "
                f"{total_distance:.4f}"
            )

        elif frame_type == "complete":

            ax.set_title(
                f"{title}\n"
                f"✓ COMPLETE TOUR\n"
                f"Tour: {tour}"
                f" → "
                f"{simulator.start_city}\n"
                f"Total distance: "
                f"{total_distance:.4f}"
            )

        return (
            route_line,
            proposed_line,
            current_marker,
            selected_marker,
            start_marker,
            available_marker,
        )

    # --------------------------------------------------
    # Animation
    # --------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=interval,
        repeat=False,
        blit=False,
    )

    # Keep animation object attached to figure.
    # This prevents Matplotlib from deleting it.
    fig._tsp_animation = animation

    return animation


def _validate_tour(
    instance: TSPInstance,
    tour: Iterable[int],
) -> None:

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
