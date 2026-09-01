
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .utils import tour_cost


def plot_cities(instance, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    xy = np.asarray(instance.coordinates, dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], s=100, zorder=3)

    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), xytext=(7, 7), textcoords="offset points")

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
    ax.set_title(f"{title} — Cost: {tour_cost(tour, instance):.4f}")

    return ax


def save_simulation(simulator, project_root):
    path = Path(project_root) / ".simulation.json"

    data = {
        "start_city": simulator.start_city,
        "actions": simulator.tour[1:],
        "tour": simulator.tour,
        "total_cost": simulator.total_cost,
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_saved_simulation(project_root):
    path = Path(project_root) / ".simulation.json"

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def animate_simulation(instance, actions, start_city, interval=900):
    xy = np.asarray(instance.coordinates, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(xy[:, 0], xy[:, 1], s=100, zorder=3)

    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), xytext=(7, 7), textcoords="offset points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    ax.scatter(
        [xy[start_city, 0]],
        [xy[start_city, 1]],
        s=220,
        facecolors="none",
        linewidths=3,
        zorder=5,
    )

    line, = ax.plot([], [], linewidth=2.5)
    current, = ax.plot([], [], "o", markersize=14, zorder=6)
    status = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    cost_labels = []
    route = [start_city]

    def add_cost_label(a, b):
        x1, y1 = xy[a]
        x2, y2 = xy[b]
        label = ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            f"{instance.cost(a, b):.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )
        cost_labels.append(label)

    def update(frame):
        if frame > 0:
            previous = route[-1]
            city = actions[frame - 1]
            route.append(city)
            add_cost_label(previous, city)

        plotted = route.copy()

        if frame == len(actions):
            plotted.append(start_city)
            if len(route) > 1:
                add_cost_label(route[-1], start_city)

        points = xy[plotted]
        line.set_data(points[:, 0], points[:, 1])

        city = route[-1]
        current.set_data([xy[city, 0]], [xy[city, 1]])

        if frame == 0:
            status.set_text(f"Start city: {start_city}")
        elif frame < len(actions):
            status.set_text(
                f"Current city: {city}   Next action: {actions[frame]}"
            )
        else:
            status.set_text("Tour complete")

        return line, current, status, *cost_labels

    animation = FuncAnimation(
        fig,
        update,
        frames=len(actions) + 1,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation

