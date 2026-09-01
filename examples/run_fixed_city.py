
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import save_simulation


def main():
    instance = TSPInstance.from_json(
        PROJECT_ROOT / "instances" / "five_cities.json"
    )

    simulator = TSPSimulator(instance)

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
        actions = state["available_actions"]

        action = simulator._rng.choice(actions)

        print(
            f"Current city: {state['current_city']} | "
            f"Available actions: {actions} | "
            f"Selected action: {action}"
        )

        simulator.step(action)

    simulator.close_tour()

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", simulator.tour + [simulator.start_city])
    print("Closed:", simulator.done)
    print("Total distance:", simulator.total_distance)

    save_simulation(simulator, PROJECT_ROOT)

    print("\nSimulation saved for visualization.")


if __name__ == "__main__":
    main()

