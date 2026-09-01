from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .instance import TSPInstance


def plot_cities(
    instance: TSPInstance,
    ax: Optional[plt.Axes] = None,
    title: str = "TSP Cities",
) -> plt.Axes:
    """Plot the fixed city locations."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

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

    for city, (x, y) in enumerate(coordinates):
        ax.annotate(
            f"City {city}",
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    return ax


def animate_simulation(
    simulator,
    interval: int = 1200,
    title: str = "Fixed-City TSP Simulation",
):
    """
    Animate the exact trajectory produced by the simulator.

    The animation shows:
    1. Starting city.
    2. Current city.
    3. Selected next city.
    4. Route after the action.
    5. Final return to the starting city.
    """

    history = simulator.trajectory()

    if not history:
        raise ValueError(
            "Simulator has no recorded trajectory."
        )

    instance = simulator.instance

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    # --------------------------------------------------
    # Build visual frames from the actual trajectory
    # --------------------------------------------------

    frames = []

    for record in history:

        if record["event"] == "start":

            frames.append({
                "type": "start",
                "current": record["current_city"],
                "selected": None,
                "tour": record["tour"],
                "available": record["available_actions"],
                "distance": record["total_distance"],
            })

        elif record["event"] == "step":

            # Decision frame:
            # current city selects the next city.
            frames.append({
                "type": "decision",
                "current": record["previous_city"],
                "selected": record["action"],
                "tour": record["tour"][:-1],
                "available": record["available_actions"],
                "distance": (
                    record["total_distance"]
                    - record["distance_added"]
                ),
            })

            # Result frame:
            # action has now happened.
            frames.append({
                "type": "move",
                "current": record["current_city"],
                "selected": None,
                "tour": record["tour"],
                "available": record["available_actions"],
                "distance": record["total_distance"],
            })

        elif record["event"] == "close":

            # Decision to return to start.
            frames.append({
                "type": "decision",
                "current": record["previous_city"],
                "selected": record["action"],
                "tour": record["tour"],
                "available": [],
                "distance": (
                    record["total_distance"]
                    - record["distance_added"]
                ),
            })

            # Completed tour.
            frames.append({
                "type": "complete",
                "current": record["action"],
                "selected": None,
                "tour": record["tour"],
                "available": [],
                "distance": record["total_distance"],
            })

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    margin_x = max(np.ptp(x) * 0.20, 0.5)
    margin_y = max(np.ptp(y) * 0.20, 0.5)

    ax.set_xlim(
        x.min() - margin_x,
        x.max() + margin_x,
    )

    ax.set_ylim(
        y.min() - margin_y,
        y.max() + margin_y,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # Fixed cities
    # --------------------------------------------------

    ax.scatter(
        x,
        y,
        s=100,
        zorder=3,
    )

    for city, (cx, cy) in enumerate(coordinates):
        ax.annotate(
            f"City {city}",
            (cx, cy),
            xytext=(7, 7),
            textcoords="offset points",
        )

    # --------------------------------------------------
    # Dynamic objects
    # --------------------------------------------------

    route_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        zorder=2,
    )

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=18,
        linestyle="None",
        zorder=5,
    )

    selected_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=22,
        linestyle="None",
        zorder=6,
    )

    start_marker, = ax.plot(
        [],
        [],
        marker="s",
        markersize=14,
        linestyle="None",
        zorder=4,
    )

    # --------------------------------------------------
    # Marker helper
    # --------------------------------------------------

    def set_marker(marker, city):

        if city is None:
            marker.set_data([], [])
            return

        marker.set_data(
            [coordinates[city, 0]],
            [coordinates[city, 1]],
        )

    # --------------------------------------------------
    # Initialize
    # --------------------------------------------------

    def init():

        route_line.set_data([], [])
        current_marker.set_data([], [])
        selected_marker.set_data([], [])
        start_marker.set_data([], [])

        return (
            route_line,
            current_marker,
            selected_marker,
            start_marker,
        )

    # --------------------------------------------------
    # Update animation
    # --------------------------------------------------

    def update(frame_index):

        frame = frames[frame_index]

        tour = frame["tour"]
        current = frame["current"]
        selected = frame["selected"]
        distance = frame["distance"]

        # Route
        if tour:
            route = coordinates[tour]

            route_line.set_data(
                route[:, 0],
                route[:, 1],
            )
        else:
            route_line.set_data([], [])

        # Current city
        set_marker(
            current_marker,
            current,
        )

        # Start city
        set_marker(
            start_marker,
            simulator.start_city,
        )

        # Selected city
        set_marker(
            selected_marker,
            selected,
        )

        # Text
        if frame["type"] == "start":

            ax.set_title(
                f"{title}\n"
                f"Start → City {current}\n"
                f"Available: {frame['available']}"
            )

        elif frame["type"] == "decision":

            ax.set_title(
                f"{title}\n"
                f"City {current} → City {selected}\n"
                f"Selecting next city | "
                f"Distance: {distance:.3f}"
            )

        elif frame["type"] == "move":

            ax.set_title(
                f"{title}\n"
                f"Moved to City {current}\n"
                f"Tour: {tour} | "
                f"Distance: {distance:.3f}"
            )

        elif frame["type"] == "complete":

            # Draw final edge back to start.
            closed_route = tour + [simulator.start_city]
            route = coordinates[closed_route]

            route_line.set_data(
                route[:, 0],
                route[:, 1],
            )

            ax.set_title(
                f"{title}\n"
                f"✓ COMPLETE TOUR\n"
                f"{tour} → {simulator.start_city}\n"
                f"Total distance: {distance:.3f}"
            )

        return (
            route_line,
            current_marker,
            selected_marker,
            start_marker,
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

    return animation


def plot_tour_from_simulator(
    simulator,
    ax: Optional[plt.Axes] = None,
):
    """Plot the simulator's current completed tour."""

    if not simulator.done:
        raise ValueError(
            "The simulator tour must be completed first."
        )

    coordinates = np.asarray(
        simulator.instance.coordinates,
        dtype=float,
    )

    tour = list(simulator.tour)
    route = tour + [simulator.start_city]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=100,
    )

    for city, (x, y) in enumerate(coordinates):
        ax.annotate(
            f"City {city}",
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
        )

    route_coordinates = coordinates[route]

    ax.plot(
        route_coordinates[:, 0],
        route_coordinates[:, 1],
        marker="o",
        linewidth=2.5,
    )

    ax.set_title(
        f"Completed TSP Tour — "
        f"Distance: {simulator.total_distance:.3f}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    return ax
