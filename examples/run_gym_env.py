import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsp.env import TSPEnv
from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import (
    create_live_visualization,
    update_live_visualization,
)


def choose_instance(root):
    print("\nSelect cost model")
    print("-----------------")
    print("1. Euclidean cost")
    print("2. Custom cost matrix")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == "1":
            return TSPInstance.from_json(
                root / "instances/five_cities.json"
            )

        if choice == "2":
            return TSPInstance.from_json(
                root / "instances/five_cities_custom_cost.json"
            )

        if choice == "3":
            sys.exit(0)

        print("Invalid choice.")


def main():
    root = Path(__file__).resolve().parents[1]

    # One instance
    instance = choose_instance(root)

    # One simulator
    simulator = TSPSimulator(instance)

    # Gym wraps the SAME simulator
    env = TSPEnv(simulator)

    # Start one episode
    _, info = env.reset()

    print(f"\nInstance: {instance.name}")
    print(f"Start city: {info['start_city']}")

    frames = root / "frames"
    frames.mkdir(exist_ok=True)

    fig, _, line, current, status = create_live_visualization(
        instance, simulator.start_city
    )

    frame = 0
    fig.savefig(frames / f"frame_{frame:03d}.png")

    while info["available_actions"]:
        action = random.choice(info["available_actions"])

        _, reward, terminated, truncated, info = env.step(action)

        frame += 1
        update_live_visualization(
            instance, simulator, line, current, status
        )
        fig.savefig(frames / f"frame_{frame:03d}.png")

        print(
            f"Action: {action} | "
            f"Current: {info['current_city']} | "
            f"Reward: {reward:.4f}"
        )

    print("\nFinal")
    print("-----")
    print(f"Tour: {simulator.tour + [simulator.start_city]}")
    print(f"Closed: {simulator.done}")
    print(f"Cost: {simulator.total_cost:.6f}")
    print(f"Frames: {frame + 1}")

    plt.show()


if __name__ == "__main__":
    main()
