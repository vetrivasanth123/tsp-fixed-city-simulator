from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator


def main() -> None:

    instance_path = (
        PROJECT_ROOT
        / "instances"
        / "five_cities.json"
    )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "fixed_city_run.json"

    instance = TSPInstance.from_json(instance_path)

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

    actions = []

    print("\nAction sequence")
    print("---------------")

    while simulator.available_actions():

        state = simulator.state()
        available = state["available_actions"]

        # Temporary action policy.
        # Later replaced by the RL agent.
        action = simulator._rng.choice(available)

        print(
            f"Current city: {state['current_city']} | "
            f"Available actions: {available} | "
            f"Selected action: {action}"
        )

        actions.append(action)
        simulator.step(action)

    simulator.close_tour()

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", simulator.tour)
    print("Closed:", simulator.done)
    print("Total distance:", simulator.total_distance)

    # Save EXACT simulation trajectory.
    data = {
        "instance": instance.name,
        "start_city": simulator.start_city,
        "actions": actions,
        "tour": simulator.tour,
        "total_distance": simulator.total_distance,
    }

    output_path.write_text(
        json.dumps(data, indent=2)
    )

    print("\nSaved exact simulation to:")
    print(output_path)


if __name__ == "__main__":
    main()
