
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .utils import tour_length


def plot_cities(instance, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    xy = np.asarray(instance.coordinates, dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], s=100, zorder=3)

    for i, (x, y) in enumerate(xy):
        ax.annotate(
            str(i), (x, y),
            xytext=(7, 7),
            textcoords="offset points",
        )

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

    path.write_text(json.dumps({
        "start_city": simulator.start_city,
        "actions": simulator.tour[1:],
        "tour": simulator.tour,
        "total_distance": simulator.total_distance,
    }, indent=2))


def load_saved_simulation(project_root):
    path = Path(project_root) / ".simulation.json"

    if not path.exists():
        return None

    return json.loads(path.read_text())


def animate_simulation(instance, actions, start_city, interval=900):
    xy = np.asarray(instance.coordinates, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cities(instance, ax)

    ax.scatter(
        [xy[start_city, 0]],
        [xy[start_city, 1]],
        s=220,
        facecolors="none",
        linewidths=3,
        zorder=5,
    )

    line, = ax.plot([], [], linewidth=2.5)
    current, = ax.plot([], [], "o", markersize=14)

    route = [start_city]
    labels = []

    def update(frame):
        if frame > 0:
            route.append(actions[frame - 1])

        # Clear previous edge-distance labels.
        for label in labels:
            label.remove()
        labels.clear()

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

        # Show distance on every edge already travelled.
        for i in range(len(plotted) - 1):
            a, b = plotted[i], plotted[i + 1]

            distance = np.linalg.norm(xy[b] - xy[a])

            mx = (xy[a, 0] + xy[b, 0]) / 2
            my = (xy[a, 1] + xy[b, 1]) / 2

            labels.append(
                ax.annotate(
                    f"{distance:.2f}",
                    (mx, my),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            )

        return line, current, *labels

    animation = FuncAnimation(
        fig,
        update,
        frames=len(actions) + 1,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation

