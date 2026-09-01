
"""
Visualization utilities for the fixed-city TSP simulator.

The animation replays the exact trajectory produced by the simulator.

For every decision, the animation shows:

    current city
          ↓
    selected next city
          ↓
    movement to selected city
          ↓
    updated route

The final animation shows the completed closed tour.

No optimization or RL logic is contained here.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .instance import TSPInstance
from .utils import tour_length


# ============================================================
# STATIC CITY PLOT
# ============================================================

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
        for city_index, (x, y) in enumerate(coordinates):
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


# ============================================================
# STATIC TOUR PLOT
# ============================================================

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


# ============================================================
# SIMULATOR TOUR PLOT
# ============================================================

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


# ============================================================
# ANIMATION
# ============================================================

def animate_simulation(
    simulator,
    interval: int = 1200,
    show_labels: bool = True,
    title: str = "Fixed-City TSP Simulation",
):
    """
    Animate the exact trajectory produced by the simulator.

    The simulator must already contain a completed trajectory.

    Each simulator action is represented by two frames:

    1. Decision frame
       Current city and selected next city are shown.

    2. Result frame
       The selected city becomes the current city and the
       route is updated.

    The final closing edge back to the start city is also shown.

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

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # --------------------------------------------------------
    # Build visual frames
    # --------------------------------------------------------

    frames = []

    for record in history:

        event = record["event"]

        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        if event == "start":

            frames.append(
                {
                    "type": "start",
                    "current_city": record["current_city"],
                    "tour": list(record["tour"]),
                    "available_actions": list(
                        record["available_actions"]
                    ),
                    "action": None,
                    "total_distance": record[
                        "total_distance"
                    ],
                    "distance_added": 0.0,
                }
            )

        # ----------------------------------------------------
        # NORMAL ACTION
        # ----------------------------------------------------

        elif event == "step":

            previous_city = record["previous_city"]
            action = record["action"]
            distance_added = record["distance_added"]

            # -----------------------------------------------
            # Decision frame
            # -----------------------------------------------

            frames.append(
                {
                    "type": "decision",
                    "current_city": previous_city,
                    "tour": list(record["tour"][:-1]),
                    "available_actions": list(
                        record["available_actions"]
                    ),
                    "action": action,
                    "total_distance": (
                        record["total_distance"]
                        - distance_added
                    ),
                    "distance_added": 0.0,
                }
            )

            # -----------------------------------------------
            # Result frame
            # -----------------------------------------------

            frames.append(
                {
                    "type": "transition",
                    "current_city": record["current_city"],
                    "tour": list(record["tour"]),
                    "available_actions": list(
                        record["available_actions"]
                    ),
                    "action": action,
                    "total_distance": record[
                        "total_distance"
                    ],
                    "distance_added": distance_added,
                }
            )

        # ----------------------------------------------------
        # CLOSE TOUR
        # ----------------------------------------------------

        elif event == "close":

            previous_city = record["previous_city"]
            start_city = record["action"]
            distance_added = record["distance_added"]

            # -----------------------------------------------
            # Closing decision
            # -----------------------------------------------

            frames.append(
                {
                    "type": "close_decision",
                    "current_city": previous_city,
                    "tour": list(record["tour"]),
                    "available_actions": [],
                    "action": start_city,
                    "total_distance": (
                        record["total_distance"]
                        - distance_added
                    ),
                    "distance_added": 0.0,
                }
            )

            # -----------------------------------------------
            # Completed tour
            # -----------------------------------------------

            frames.append(
                {
                    "type": "complete",
                    "current_city": start_city,
                    "tour": list(record["tour"]),
                    "available_actions": [],
                    "action": start_city,
                    "total_distance": record[
                        "total_distance"
                    ],
                    "distance_added": distance_added,
                }
            )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fixed cities
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=110,
        zorder=3,
    )

    if show_labels:

        for city_index, (cx, cy) in enumerate(
            coordinates
        ):

            ax.annotate(
                str(city_index),
                (cx, cy),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
            )

    # --------------------------------------------------------
    # Route line
    # --------------------------------------------------------

    route_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        zorder=2,
    )

    # --------------------------------------------------------
    # Current city
    # --------------------------------------------------------

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=17,
        linestyle="None",
        zorder=6,
    )

    # --------------------------------------------------------
    # Selected city
    # --------------------------------------------------------

    selected_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=22,
        linestyle="None",
        zorder=7,
    )

    # --------------------------------------------------------
    # Start city
    # --------------------------------------------------------

    start_marker, = ax.plot(
        [],
        [],
        marker="s",
        markersize=14,
        linestyle="None",
        zorder=5,
    )

    # --------------------------------------------------------
    # Available actions
    # --------------------------------------------------------

    available_marker = ax.scatter(
        [],
        [],
        s=160,
        marker="o",
        zorder=4,
    )

    # --------------------------------------------------------
    # Marker helper
    # --------------------------------------------------------

    def set_marker(marker, city):

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

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

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
            np.empty((0, 2))
        )

        ax.set_title(title)

        return (
            route_line,
            current_marker,
            selected_marker,
            start_marker,
            available_marker,
        )

    # --------------------------------------------------------
    # Update frame
    # --------------------------------------------------------

    def update(frame_index):

        frame = frames[frame_index]

        frame_type = frame["type"]

        current_city = frame["current_city"]
        tour = frame["tour"]
        available_actions = frame[
            "available_actions"
        ]
        action = frame["action"]
        total_distance = frame[
            "total_distance"
        ]

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        if tour:

            route_coordinates = coordinates[tour]

            route_line.set_data(
                route_coordinates[:, 0],
                route_coordinates[:, 1],
            )

        else:

            route_line.set_data(
                [],
                [],
            )

        # ----------------------------------------------------
        # Current city
        # ----------------------------------------------------

        set_marker(
            current_marker,
            current_city,
        )

        # ----------------------------------------------------
        # Start city
        # ----------------------------------------------------

        set_marker(
            start_marker,
            simulator.start_city,
        )

        # ----------------------------------------------------
        # Available actions
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Selected action
        # ----------------------------------------------------

        if frame_type in (
            "decision",
            "close_decision",
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

        # ----------------------------------------------------
        # Titles
        # ----------------------------------------------------

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
                f"  →  Selected City {action}\n"
                f"Available actions: "
                f"{available_actions}"
            )

        elif frame_type == "transition":

            ax.set_title(
                f"{title}\n"
                f"Moved: City {current_city}\n"
                f"Tour: {tour}\n"
                f"Added distance: "
                f"{frame['distance_added']:.4f}  |  "
                f"Total: {total_distance:.4f}"
            )

        elif frame_type == "close_decision":

            ax.set_title(
                f"{title}\n"
                f"CLOSING TOUR\n"
                f"City {current_city}"
                f"  →  Start City {action}\n"
                f"Current distance: "
                f"{total_distance:.4f}"
            )

        elif frame_type == "complete":

            # Draw the final return edge.
            final_route = list(tour)

            if final_route:
                final_route.append(
                    simulator.start_city
                )

                final_coordinates = coordinates[
                    final_route
                ]

                route_line.set_data(
                    final_coordinates[:, 0],
                    final_coordinates[:, 1],
                )

            ax.set_title(
                f"{title}\n"
                f"✓ COMPLETE TOUR\n"
                f"Tour: {tour} → "
                f"{simulator.start_city}\n"
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

    # --------------------------------------------------------
    # Create animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=interval,
        repeat=False,
        blit=False,
    )

    # Store frames so the animation object retains them.
    animation._simulation_frames = frames

    return animation


# ============================================================
# TOUR VALIDATION
# ============================================================

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

