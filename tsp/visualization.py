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

    if instance.width is not None:
        ax.set_xlim(0, instance.width)
    if instance.height is not None:
        ax.set_ylim(0, instance.height)

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


def save_simulation(simulator, project_root, trajectory):
    path = Path(project_root) / ".simulation.json"
    instance = simulator.instance
    rewards = [float(step["reward"]) for step in trajectory]

    data = {
        "instance": {
            "name": instance.name,
            "coordinates": instance.coordinates.tolist(),
            "cost_matrix": instance.cost_matrix.tolist(),
            "width": instance.width,
            "height": instance.height,
        },
        "trajectory": trajectory,
        "summary": {
            "start_city": int(simulator.start_city),
            "tour": simulator.tour + [simulator.start_city],
            "total_cost": float(simulator.total_cost),
            "total_reward": float(sum(rewards)),
        },
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_saved_simulation(project_root):
    path = Path(project_root) / ".simulation.json"

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def animate_simulation(
    instance,
    actions,
    start_city,
    rewards,
    interval=900,
):
    xy = np.asarray(instance.coordinates, dtype=float)

    fig, (summary_ax, ax) = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        gridspec_kw={"width_ratios": [1, 3]},
    )

    summary_ax.axis("off")
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
    current, = ax.plot([], [], "o", markersize=14, zorder=6)

    summary = summary_ax.text(
        0.02,
        0.95,
        "",
        transform=summary_ax.transAxes,
        va="top",
        ha="left",
    )

    cost_labels = []
    route = [start_city]

    def add_cost_label(a, b):
        x1, y1 = xy[a]
        x2, y2 = xy[b]

        cost_labels.append(
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                f"{instance.cost(a, b):.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
        )

    def update(frame):
        if frame > 0:
            previous = route[-1]
            action = actions[frame - 1]

            if action == "CLOSE":
                add_cost_label(previous, start_city)
            else:
                route.append(action)
                add_cost_label(previous, action)

        plotted = route.copy()

        if frame == len(actions):
            plotted.append(start_city)

        points = xy[plotted]
        line.set_data(points[:, 0], points[:, 1])

        city = route[-1]
        current.set_data([xy[city, 0]], [xy[city, 1]])

        if frame == 0:
            summary.set_text(
                f"SUMMARY\n\n"
                f"Start city: {start_city}\n"
                f"Total cost: 0.0000\n"
                f"Total reward: 0.0000"
            )

        elif actions[frame - 1] == "CLOSE":
            summary.set_text(
                f"SUMMARY\n\n"
                f"Tour complete\n"
                f"Total cost: "
                f"{sum(instance.cost(a, b) for a, b in zip(route, route[1:] + [start_city])):.4f}\n"
                f"Total reward: {sum(rewards):.4f}"
            )

        else:
            summary.set_text(
                f"SUMMARY\n\n"
                f"Current city: {city}\n"
                f"Action: {action}\n"
                f"Cost: {instance.cost(previous, city):.4f}\n"
                f"Reward: {rewards[frame - 1]:.4f}\n"
                f"Total cost: {sum(-r for r in rewards[:frame]):.4f}\n"
                f"Total reward: {sum(rewards[:frame]):.4f}"
            )

        return line, current, summary, *cost_labels

    animation = FuncAnimation(
        fig,
        update,
        frames=len(actions) + 1,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation
