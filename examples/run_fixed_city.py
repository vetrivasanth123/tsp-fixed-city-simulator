from pathlib import Path
import sys

from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import animate_simulation


def main():

    instance = TSPInstance.from_json(
        PROJECT_ROOT / "instances" / "five_cities.json"
    )

    simulator = TSPSimulator(instance)

    state = simulator.state()

    print("Project root:", PROJECT_ROOT)
    print("Instance:", instance.name)
    print("Number of cities:", instance.num_cities)
    print("\nCoordinates:")
    print(instance.coordinates)

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

        # Temporary action policy.
        # Later this becomes the RL agent's action.
        action = simulator.available_actions()[
            simulator._rng.randrange(
                len(simulator.available_actions())
            )
        ]

        print(
            f"Current city: {state['current_city']} | "
            f"Available actions: {state['available_actions']} | "
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

    # Animation uses the SAME actions produced above.
    fig, animation = animate_simulation(
        instance,
        actions,
        simulator.start_city,
    )

    display(HTML(animation.to_jshtml()))


if __name__ == "__main__":
    main()
