"""
Visualization utilities for the fixed-city TSP simulator.

The animation shows the exact trajectory recorded by the simulator:

    start city
        ↓
    choose next city
        ↓
    connect cities
        ↓
    repeat
        ↓
    close complete tour

No RL or optimization is performed here.
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
    """Plot fixed city locations."""

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
        for city, (x, y) in enumerate(coordinates):
            ax.annotate(
                str(city),
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
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
    """Plot a complete TSP tour."""

    tour = list(tour)
    _validate_tour(instance, tour)

    if ax is None:
        _, ax = plt.subplots()

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

    route = tour + [tour[0]] if close_tour else tour

    points = coordinates[route]

    ax.plot(
        points[:, 0],
        points[:, 1],
        marker="o",
        linewidth=2,
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
        simulator.instance,
        simulator.tour,
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
    Animate the exact trajectory produced by the simulator.

    The animation is generated from simulator.history, so it shows
    the same actions and city sequence that actually occurred.
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

    # --------------------------------------------------
    # Convert simulator events into visual frames
    # --------------------------------------------------

    frames = []

    for record in history:

        event = record["event"]

        if event == "start":

            frames.append({
                "type": "start",
                "current": record["current_city"],
                "selected": None,
                "tour": list(record["tour"]),
                "available": list(
                    record["available_actions"]
                ),
                "distance": record["total_distance"],
            })

        elif event == "step":

            # Decision frame:
            # current city chooses the next city.
            frames.append({
                "type": "decision",
                "current": record["previous_city"],
                "selected": record["action"],
                "tour": list(record["tour"][:-1]),
                "available": list(
                    record["available_actions"]
                ),
                "distance": (
                    record["total_distance"]
                    - record["distance_added"]
                ),
            })

            # Movement/result frame:
            # the selected city becomes current.
            frames.append({
                "type": "move",
                "current": record["current_city"],
                "selected": None,
                "tour": list(record["tour"]),
                "available": list(
                    record["available_actions"]
                ),
                "distance": record["total_distance"],
            })

        elif event == "close":

            # Closing decision.
            frames.append({
                "type": "decision",
                "current": record["previous_city"],
                "selected": record["action"],
                "tour": list(record["tour"]),
                "available": [],
                "distance": (
                    record["total_distance"]
                    - record["distance_added"]
                ),
            })

            # Final result.
            frames.append({
                "type": "complete",
                "current": record["action"],
                "selected": None,
                "tour": list(record["tour"]),
                "available": [],
                "distance": record["total_distance"],
            })

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    margin_x = max(np.ptp(x) * 0.25, 1.0)
    margin_y = max(np.ptp(y) * 0.25, 1.0)

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
    ax.grid(True, alpha=0.3)

    # Fixed cities.
    ax.scatter(
        x,
        y,
        s=100,
        zorder=3,
    )

    if show_labels:
        for city, (cx, cy) in enumerate(coordinates):
            ax.annotate(
                str(city),
                (cx, cy),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
            )

    # Route.
    route_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        zorder=2,
    )

    # Current city.
    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=18,
        linestyle="None",
        zorder=5,
    )

    # Selected city.
    selected_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=22,
        linestyle="None",
        zorder=6,
    )

    # Start city.
    start_marker, = ax.plot(
        [],
        [],
        marker="s",
        markersize=14,
        linestyle="None",
        zorder=4,
    )

    # Available cities.
    available_marker = ax.scatter(
        [],
        [],
        s=140,
        zorder=4,
    )

    def set_marker(marker, city):

        if city is None:
            marker.set_data([], [])
            return

        marker.set_data(
            [coordinates[city, 0]],
            [coordinates[city, 1]],
        )

    def init():

        route_line.set_data([], [])
        current_marker.set_data([], [])
        selected_marker.set_data([], [])
        start_marker.set_data([], [])

        available_marker.set_offsets(
            np.empty((0, 2))
        )

        return (
            route_line,
            current_marker,
            selected_marker,
            start_marker,
            available_marker,
        )

    def update(index):

        frame = frames[index]

        frame_type = frame["type"]
        current = frame["current"]
        selected = frame["selected"]
        tour = frame["tour"]
        available = frame["available"]
        distance = frame["distance"]

        # ----------------------------------------------
        # Route
        # ----------------------------------------------

        if tour:

            points = coordinates[tour]

            route_line.set_data(
                points[:, 0],
                points[:, 1],
            )

        else:
            route_line.set_data([], [])

        # ----------------------------------------------
        # Markers
        # ----------------------------------------------

        set_marker(
            current_marker,
            current,
        )

        set_marker(
            start_marker,
            simulator.start_city,
        )

        set_marker(
            selected_marker,
            selected,
        )

        if available:

            available_marker.set_offsets(
                coordinates[available]
            )

        else:

            available_marker.set_offsets(
                np.empty((0, 2))
            )

        # ----------------------------------------------
        # Text
        # ----------------------------------------------

        if frame_type == "start":

            ax.set_title(
                f"{title}\n"
                f"START → City {current}\n"
                f"Available: {available}"
            )

        elif frame_type == "decision":

            ax.set_title(
                f"{title}\n"
                f"City {current}  →  "
                f"Choosing City {selected}\n"
                f"Available: {available}  |  "
                f"Distance: {distance:.4f}"
            )

        elif frame_type == "move":

            ax.set_title(
                f"{title}\n"
                f"Moved to City {current}\n"
                f"Tour: {tour}  |  "
                f"Distance: {distance:.4f}"
            )

        elif frame_type == "complete":

            # Draw final return edge.
            closed_route = tour + [simulator.start_city]
            points = coordinates[closed_route]

            route_line.set_data(
                points[:, 0],
                points[:, 1],
            )

            ax.set_title(
                f"{title}\n"
                f"COMPLETE TOUR\n"
                f"{tour} → {simulator.start_city}\n"
                f"Total distance: {distance:.4f}"
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

    # Keep the figure alive for notebook rendering.
    animation._fig = fig

    return animation


def _validate_tour(
    instance: TSPInstance,
    tour: Iterable[int],
) -> None:
    """Validate a complete TSP tour."""

    tour = list(tour)

    if len(tour) != instance.num_cities:
        raise ValueError(
            f"Tour must contain exactly "
            f"{instance.num_cities} cities; "
            f"received {len(tour)}."
        )

    if sorted(tour) != list(
        range(instance.num_cities)
    ):
        raise ValueError(
            "Tour must contain every city "
            "index exactly once."
        )
