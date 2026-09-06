from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.env import TSPEnv
import tsp.visualization as visualization


def choose_instance():
    print("\nSelect cost model")
    print("-----------------")
    print("1. Euclidean cost")
    print("2. Custom cost matrix")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == "1":
            return TSPInstance.from_json(
                PROJECT_ROOT / "instances/five_cities.json"
            )

        if choice == "2":
            return TSPInstance.from_json(
                PROJECT_ROOT / "instances/five_cities_custom_cost.json"
            )

        if choice == "3":
            sys.exit(0)

        print("Invalid choice.")


def main():
    instance = choose_instance()
    env = TSPEnv(instance)

    _, info = env.reset()
    rng = random.Random()

    print("\nProject root:", PROJECT_ROOT)
    print("Instance:", instance.name)
    print("Number of cities:", instance.num_cities)

    print("\nCoordinates:")
    print(instance.coordinates)

    print("\nCost matrix:")
    print(instance.cost_matrix)

    print("\nInitial state")
    print("-------------")
    print("Start city:", info["start_city"])
    print("Current city:", info["current_city"])
    print("Visited:", info["tour"])
    print("Available actions:", info["available_actions"])

    total_reward = 0.0

    print("\nAction sequence")
    print("---------------")

    while info["available_actions"]:
        action = rng.choice(info["available_actions"])
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Current city: {info['current_city']} | "
            f"Available actions: {info['available_actions']} | "
            f"Selected action: {action}"
        )

    simulator = env.simulator
    closed_tour = simulator.tour + [simulator.start_city]

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", closed_tour)
    print("Closed:", simulator.done)
    print("Total cost:", simulator.total_cost)
    print("Total reward:", total_reward)

    visualization.save_simulation(simulator, PROJECT_ROOT)

    print("Saved episode:", PROJECT_ROOT / ".simulation.json")


if __name__ == "__main__":
    main()
