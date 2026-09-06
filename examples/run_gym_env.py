import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsp.env import TSPEnv
from tsp.instance import TSPInstance
from tsp.visualization import create_live_visualization, update_live_visualization


def main():
    root = Path(__file__).resolve().parents[1]

    print("Select cost model")
    print("-----------------")
    print("1. Euclidean cost")
    print("2. Custom cost matrix")
    print("3. Exit")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == "3":
        return
    if choice not in {"1", "2"}:
        print("Invalid choice.")
        return

    filename = (
        "five_cities.json"
        if choice == "1"
        else "five_cities_custom_cost.json"
    )

    instance = TSPInstance.from_json(root / "instances" / filename)
    env = TSPEnv(instance, seed=42)
    _, info = env.reset(seed=42)

    print(f"\nInstance: {instance.name}")
    print(f"Number of cities: {instance.num_cities}")
    print(f"Start city: {info['start_city']}")

    frames = root / "frames"
    frames.mkdir(exist_ok=True)

    fig, _, line, current, status = create_live_visualization(
        instance, info["start_city"]
    )

    frame = 0
    fig.savefig(frames / f"frame_{frame:03d}.png")

    while not env.simulator.done:
        action = random.choice(info["available_actions"])
        _, reward, _, _, info = env.step(action)

        frame += 1
        update_live_visualization(
            instance, env.simulator, line, current, status
        )
        fig.savefig(frames / f"frame_{frame:03d}.png")

        print(
            f"Action: {action} | Current: {info['current_city']} | "
            f"Reward: {reward:.4f}"
        )

    print("\nFinal")
    print(f"Tour: {info['tour']}")
    print(f"Cost: {info['total_cost']:.6f}")
    print(f"Frames: {frame + 1}")

    plt.show()


if __name__ == "__main__":
    main()
