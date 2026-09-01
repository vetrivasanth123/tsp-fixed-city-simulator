from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .instance import TSPInstance
from .utils import tour_length


def plot_cities(
    instance: TSPInstance,
    ax=None,
    show_labels: bool = True,
    title: str = "TSP City Locations",
):
    """Plot fixed city locations."""

    if ax is None:
        _, ax = plt.subplots()

    coordinates = np.asarray(instance.coordinates, dtype=float)

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
    ax=None,
    show_labels: bool = True,
    show_distance: bool = True,
    close_tour: bool = True,
    title: str = "TSP Tour",
):
    """Plot a completed TSP tour."""

    tour = list(tour)

    if len(tour) != instance.num_cities:
        raise ValueError("Tour must contain every city exactly once.")

    if sorted(tour) != list(range(instance.num_cities)):
        raise ValueError("Tour contains invalid or duplicate cities.")

    if ax is None:
        _, ax = plt.subplots()

    coordinates = np.asarray(instance.coordinates, dtype=float)

    plot_cities(instance, ax, show_labels, title)

    route = tour + [tour[0]] if close_tour else tour

    xy = coordinates[route]

    ax.plot(
        xy[:, 0],
        xy[:, 1],
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


def animate_simulation(
    instance: TSPInstance,
    start_city: int,
    actions: Sequence[int],
    interval: int = 1000,
):
    """
    Animate an already-recorded simulator run.

    No simulation or action selection happens here.
    """

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    actions = list(actions)

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_cities(
        instance,
        ax=ax,
        title="Fixed-City TSP Simulation",
    )

    # Starting city highlight.
    ax.scatter(
        coordinates[start_city, 0],
        coordinates[start_city, 1],
        s=240,
        facecolors="none",
        linewidths=3,
        zorder=5,
    )

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
        markersize=15,
        linestyle="None",
        zorder=6,
    )

    info = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    def update(frame):
        route = [start_city] + actions[:frame]

        # Add return edge only after the last action.
        plotted = route.copy()

        if frame == len(actions):
            plotted.append(start_city)

        xy = coordinates[plotted]

        route_line.set_data(
            xy[:, 0],
            xy[:, 1],
        )

        current = route[-1]

        current_marker.set_data(
            [coordinates[current, 0]],
            [coordinates[current, 1]],
        )

        if frame == 0:
            info.set_text(
                f"Start city: {start_city}\n"
                f"Available actions: {actions}"
            )

        elif frame < len(actions):
            previous = route[-2]
            selected = actions[frame - 1]

            info.set_text(
                f"Current city: {previous}\n"
                f"Selected action: {selected}\n"
                f"Moved to city: {current}"
            )

        else:
            info.set_text(
                f"Tour complete\n"
                f"Tour: {route}\n"
                f"Distance: "
                f"{tour_length(route, instance.distance_matrix):.4f}"
            )

        return route_line, current_marker, info

    animation = FuncAnimation(
        fig,
        update,
        frames=len(actions) + 1,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation
