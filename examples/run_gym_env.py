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
    instance = TSPInstance.from_json(root / "instances/five_cities.json")
    env = TSPEnv(instance, seed=42)

    _, info = env.reset(seed=42)

    frames = root / "frames"
    frames.mkdir(exist_ok=True)

    fig, _, line, current, status = create_live_visualization(
        instance, info["start_city"]
    )

    frame = 0
    fig.savefig(frames / f"frame_{frame:03d}.png")

    while not env.simulator.done:
        action = random.choice(info["available_actions"])
        _, reward, terminated, truncated, info = env.step(action)

        frame += 1
        update_live_visualization(
            instance, env.simulator, line, current, status
        )
        fig.savefig(frames / f"frame_{frame:03d}.png")

        print(
            f"Action: {action} | "
            f"Current: {info['current_city']} | "
            f"Reward: {reward:.4f}"
        )

    print("\nFinal")
    print(f"Tour: {info['tour']}")
    print(f"Cost: {info['total_cost']:.6f}")
    print(f"Frames: {frame + 1}")

    plt.show()


if __name__ == "__main__":
    main()
