
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .instance import TSPInstance
from .utils import tour_length


def plot_cities(instance, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    xy = np.asarray(instance.coordinates, dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], s=100, zorder=3)

    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), xytext=(7, 7),
                    textcoords="offset points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    return ax


def plot_tour(instance, tour, ax=None, title="TSP Tour"):
    tour = list(tour)

    if len(tour) != instance.num_cities:
        raise ValueError("Tour must contain every city exactly once.")

    ax = plot_cities(instance, ax)

    route = tour + [tour[0]]
    xy = np.asarray(instance.coordinates)[route]

    ax.plot(xy[:, 0], xy[:, 1], marker="o", linewidth=2)
    ax.set_title(
        f"{title} — Distance: "
        f"{tour_length(tour, instance.distance_matrix):.4f}"
    )

    return ax


def save_simulation(simulator, project_root):
    path = Path(project_root) / ".simulation.json"

    data = {
        "start_city": simulator.start_city,
        "actions": simulator.tour[1:],
        "tour": simulator.tour,
        "total_distance": simulator.total_distance,
    }

    path.write_text(json.dumps(data, indent=2))


def load_saved_simulation(project_root):
    path = Path(project_root) / ".simulation.json"

    if not path.exists():
        return None

    return json.loads(path.read_text())


def animate_simulation(instance, actions, start_city, interval=900):
    xy = np.asarray(instance.coordinates, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=100, zorder=3)

    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), xytext=(7, 7),
                    textcoords="offset points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    start = ax.scatter(
        [xy[start_city, 0]],
        [xy[start_city, 1]],
        s=220,
        facecolors="none",
        linewidths=3,
        zorder=5,
    )

    line, = ax.plot([], [], linewidth=2.5)
    current, = ax.plot([], [], "o", markersize=14)
    text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        va="top"
    )

    route = [start_city]

    def update(frame):
        if frame > 0:
            route.append(actions[frame - 1])

        plotted = route.copy()

        if frame == len(actions):
            plotted.append(start_city)

        points = xy[plotted]
        line.set_data(points[:, 0], points[:, 1])

        city = route[-1]
        current.set_data(
            [xy[city, 0]],
            [xy[city, 1]],
        )

        if frame == 0:
            text.set_text(
                f"Start city: {start_city}\n"
                f"Next action: {actions[0]}"
            )
        elif frame < len(actions):
            text.set_text(
                f"Current city: {city}\n"
                f"Selected action: {actions[frame - 1]}"
            )
        else:
            text.set_text(
                f"Tour complete\n"
                f"Tour: {route}"
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

