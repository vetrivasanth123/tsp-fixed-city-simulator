from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .instance import TSPInstance


def animate_simulation(
    instance: TSPInstance,
    actions: list[int],
    start_city: int,
    interval: int = 1000,
):
    """Animate the exact actions produced by one simulation."""

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
    ax.scatter(
        coordinates[start_city, 0],
        coordinates[start_city, 1],
        s=220,
        facecolors="none",
        linewidths=3,
        zorder=5,
    )

    line, = ax.plot([], [], linewidth=2.5, zorder=2)

    current, = ax.plot(
        [], [],
        marker="o",
        markersize=14,
        linestyle="None",
        zorder=6,
    )

    text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
    )

    route = [start_city]

    def update(frame):
        route[:] = [start_city] + actions[:frame]

        plotted = route.copy()

        if frame == len(actions):
            plotted.append(start_city)

        xy = coordinates[plotted]
        line.set_data(xy[:, 0], xy[:, 1])

        city = route[-1]
        current.set_data(
            [coordinates[city, 0]],
            [coordinates[city, 1]],
        )

        if frame == 0:
            text.set_text(
                f"Start city: {start_city}\n"
                f"Available actions: {actions}"
            )
        elif frame == len(actions):
            text.set_text(
                f"Tour complete\n"
                f"Tour: {route} → {start_city}"
            )
        else:
            text.set_text(
                f"Current city: {city}\n"
                f"Selected action: {actions[frame - 1]}"
            )

        return line, current, text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(actions) + 1,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation
