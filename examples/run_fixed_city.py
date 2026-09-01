"""Run one fixed-city TSP simulation and record its trajectory."""

from pathlib import Path
import json
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator


def main() -> None:
    instance = TSPInstance.from_json(
        PROJECT_ROOT / "instances" / "five_cities.json"
    )

    simulator = TSPSimulator(instance)
    actions = []

    print("Project root:", PROJECT_ROOT)
    print("Instance:", instance.name)
    print("Number of cities:", instance.num_cities)
    print("\nCoordinates:")
    print(instance.coordinates)

    state = simulator.state()

    print("\nInitial state")
    print("-------------")
    print("Start city:", state["start_city"])
    print("Current city:", state["current_city"])
    print("Visited:", state["visited"])
    print("Available actions:", state["available_actions"])

    print("\nAction sequence")
    print("---------------")

    while simulator.available_actions():
        state = simulator.state()
        available = state["available_actions"]

        # Temporary random policy.
        action = random.choice(available)
        actions.append(action)

        print(
            f"Current city: {state['current_city']} | "
            f"Available actions: {available} | "
            f"Selected action: {action}"
        )

        simulator.step(action)

    simulator.close_tour()

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", simulator.tour)
    print("Closed:", simulator.done)
    print("Total distance:", simulator.total_distance)

    # Save only the exact simulation trajectory, not video frames.
    record = {
        "instance": instance.name,
        "start_city": simulator.start_city,
        "actions": actions,
        "tour": simulator.tour,
        "total_distance": simulator.total_distance,
    }

    with open(PROJECT_ROOT / ".tsp_last_simulation.json", "w") as f:
        json.dump(record, f, indent=2)

    print("\nSimulation saved for visualization.")


if __name__ == "__main__":
    main()
