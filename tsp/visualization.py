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
    interval=180,
    frames_per_move=12,
):
    xy = np.asarray(instance.coordinates, dtype=float)

    fig, (summary_ax, ax) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 3]}
    )
    summary_ax.axis("off")
    plot_cities(instance, ax)

    blue, green, red, gray, edge = (
        "#2563EB", "#16A34A", "#DC2626", "#D1D5DB", "#374151"
    )
    final_city = next((a for a in reversed(actions) if a != "CLOSE"), start_city)

    ax.scatter(xy[:, 0], xy[:, 1], s=100, color=gray, zorder=3)
    ax.scatter(
        [xy[start_city, 0]], [xy[start_city, 1]],
        s=220, color=green, edgecolors=edge, linewidths=1.5, zorder=6
    )
    if final_city != start_city:
        ax.scatter(
            [xy[final_city, 0]], [xy[final_city, 1]],
            s=220, color=red, edgecolors=edge, linewidths=1.5, zorder=6
        )

    line, = ax.plot([], [], linewidth=2.8, color=blue)
    current, = ax.plot([], [], "o", markersize=12, color=green, zorder=7)
    summary = summary_ax.text(
        0.02, 0.95, "", transform=summary_ax.transAxes, va="top", ha="left"
    )

    route = [start_city]
    drawn = [False] * len(actions)
    labels = []

    def add_arrow(a, b):
        p = 0.5 * (xy[a] + xy[b])
        d = xy[b] - xy[a]
        q1, q2 = p - 0.03 * d, p + 0.03 * d

        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color=blue,
            linewidth=2.8,
            zorder=4,
        )
        ax.annotate(
            "",
            xy=q2,
            xytext=q1,
            arrowprops={
                "arrowstyle": "->",
                "color": black,
                "linewidth": 2,
            },
            zorder=6,
        )
        labels.append(
            ax.text(
                p[0], p[1],
                f"{instance.cost(a, b):.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=red
            )
        )

    def update(frame):
        step, sub = divmod(frame, frames_per_move)

        if step == 0:
            current.set_data([xy[start_city, 0]], [xy[start_city, 1]])
            summary.set_text(
                f"SUMMARY\n\nStart city: {start_city}\n"
                "Total cost: 0.0000\nTotal reward: 0.0000"
            )
            return line, current, summary, *labels

        i = min(step - 1, len(actions) - 1)
        action = actions[i]
        a = route[-1]
        b = start_city if action == "CLOSE" else action

        p = (sub + 1) / frames_per_move
        pos = xy[a] + p * (xy[b] - xy[a])
        current.set_data([pos[0]], [pos[1]])

        if sub == frames_per_move - 1 and not drawn[i]:
            add_arrow(a, b)
            drawn[i] = True
            if action != "CLOSE":
                route.append(action)

        summary.set_text(
            f"SUMMARY\n\n"
            f"{'Returning to start' if action == 'CLOSE' else f'Current city: {b}'}\n"
            f"Action: {action}\n"
            f"Cost: {-rewards[i]:.4f}\n"
            f"Reward: {rewards[i]:.4f}\n"
            f"Total cost: {-sum(rewards[:i + 1]):.4f}\n"
            f"Total reward: {sum(rewards[:i + 1]):.4f}"
        )

        if step > len(actions):
            current.set_data([xy[start_city, 0]], [xy[start_city, 1]])
            route_plot = route + [start_city]
            line.set_data(xy[route_plot, 0], xy[route_plot, 1])
            summary.set_text(
                f"SUMMARY\n\nTour complete\n"
                f"Total cost: {-sum(rewards):.4f}\n"
                f"Total reward: {sum(rewards):.4f}"
            )

        return line, current, summary, *labels

    animation = FuncAnimation(
        fig,
        update,
        frames=(len(actions) + 1) * frames_per_move,
        interval=interval,
        repeat=False,
        blit=False,
    )

    return fig, animation
