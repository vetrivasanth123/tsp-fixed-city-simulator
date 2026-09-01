from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .instance import TSPInstance


def animate_simulation(
    simulator,
    interval: int = 1000,
):
    """
    Animate the exact simulation already executed.

    No actions are selected and no simulation is run here.
    """

    instance: TSPInstance = simulator.instance
    history = simulator.history
    coordinates = np.asarray(instance.coordinates, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=100,
        zorder=3,
    )

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

    # Highlight starting city.
    start = simulator.start_city

    ax.scatter(
        coordinates[start, 0],
        coordinates[start, 1],
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
        record = history[frame]
        tour = record["tour"]

        route = list(tour)

        if record["done"]:
            route.append(start)

        xy = coordinates[route]

        route_line.set_data(
            xy[:, 0],
            xy[:, 1],
        )

        current = record["current_city"]

        current_marker.set_data(
            [coordinates[current, 0]],
            [coordinates[current, 1]],
        )

        if frame == 0:
            info.set_text(
                f"Start city: {start}\n"
                f"Current city: {current}"
            )

        elif record["done"]:
            info.set_text(
                f"Tour complete\n"
                f"Tour: {tour} → {start}\n"
                f"Distance: {record['distance']:.4f}"
            )

        else:
            previous = history[frame - 1]["current_city"]

            info.set_text(
                f"Current city: {previous}\n"
                f"Selected action: {record['action']}\n"
                f"Moved to city: {current}"
            )

        return route_line, current_marker, info

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        repeat=False,
        blit=False,
    )

    return animation
