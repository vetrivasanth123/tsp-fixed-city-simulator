import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance
from tsp.env import TSPEnv
import tsp.visualization as visualization


def main():
    parser = argparse.ArgumentParser(description="Run a flexible-city TSP simulation.")
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--n-cities", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    coordinates = CityLocationGenerator(
        args.width, args.height, args.n_cities, args.seed
    ).generate()

    instance = TSPInstance(coordinates, name="generated_tsp")
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

    rewards = []

    print("\nAction sequence")
    print("---------------")

    while info["available_actions"]:
        action = rng.choice(info["available_actions"])
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        city = info["current_city"]
        print(
            f"Current city: {city} {instance.coordinates[city]} | "
            f"Available actions: {info['available_actions']} | "
            f"Selected action: {action}"
        )

    _, reward, terminated, truncated, info = env.step(env.close_action)
    rewards.append(reward)

    city = info["current_city"]
    print(
        f"Current city: {city} {instance.coordinates[city]} | "
        f"Available actions: {info['available_actions']} | "
        f"Selected action: close"
    )

    simulator = env.simulator
    closed_tour = simulator.tour + [simulator.start_city]

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", closed_tour)
    print("Closed:", simulator.done)
    print("Total cost:", simulator.total_cost)
    print("Total reward:", sum(rewards))

    visualization.save_simulation(
        simulator,
        PROJECT_ROOT,
        rewards,
    )

    print("\nSimulation saved:", PROJECT_ROOT / ".simulation.json")


if __name__ == "__main__":
    main()
