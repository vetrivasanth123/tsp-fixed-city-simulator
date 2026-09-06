from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.env import TSPEnv


def choose_instance():
    print("\nSelect cost model")
    print("-----------------")
    print("1. Euclidean cost")
    print("2. Custom cost matrix")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice in ("1", "2"):
            file = "five_cities.json" if choice == "1" else "five_cities_custom_cost.json"
            return TSPInstance.from_json(PROJECT_ROOT / "instances" / file)

        if choice == "3":
            sys.exit(0)

        print("Invalid choice.")


def main():
    instance = choose_instance()
    simulator = TSPSimulator(instance, seed=42)
    env = TSPEnv(simulator)

    obs, info = env.reset(seed=42)
    rng = random.Random(42)

    print("\nInstance:", instance.name)
    print("Cost matrix:\n", instance.cost_matrix)

    print("\nInitial state")
    print("Start city:", info["start_city"])
    print("Available actions:", info["available_actions"])

    total_reward = 0.0

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:
        action = rng.choice(info["available_actions"])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Selected: {action} | "
            f"Current: {info['current_city']} | "
            f"Reward: {reward:.4f}"
        )

    print("\nFinal result")
    print("------------")
    print("Tour:", info["tour"])
    print("Total cost:", info["total_cost"])
    print("Total reward:", total_reward)
    print("Terminated:", terminated)


if __name__ == "__main__":
    main()
