import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsp.env import TSPEnv
from tsp.instance import TSPInstance


def main():
    instance = TSPInstance.from_json("instances/five_cities.json")
    env = TSPEnv(instance, seed=42)

    _, info = env.reset(seed=42)

    print("Initial state")
    print("------------")
    print(f"Start city: {info['start_city']}")
    print(f"Available actions: {info['available_actions']}")

    total_reward = 0.0

    print("\nAction sequence")
    print("---------------")

    while info["available_actions"]:
        action = info["available_actions"][0]
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Selected: {action} | "
            f"Current: {info['current_city']} | "
            f"Available: {info['available_actions']} | "
            f"Reward: {reward:.4f}"
        )

    print("\nFinal result")
    print("------------")
    print(f"Tour: {info['tour']}")
    print(f"Total cost: {info['total_cost']:.6f}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")


if __name__ == "__main__":
    main()
