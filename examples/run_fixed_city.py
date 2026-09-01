"""
Demonstration of the fixed-city TSP simulator.

The example:

1. Automatically finds the project root.
2. Automatically loads the fixed five-city instance.
3. Creates the simulator with a random starting city.
4. Displays the initial state and available actions.
5. Selects valid actions until all cities are visited.
6. Closes the tour.
7. Reports the final distance.
8. Visualizes the resulting tour.

This is a simulator demonstration only.
No optimization or RL agent is used yet.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

# --------------------------------------------------
# Automatically locate project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import plot_tour


def main() -> None:

    # --------------------------------------------------
    # 1. Automatically locate the instance
    # --------------------------------------------------

    instance_path = (
        PROJECT_ROOT
        / "instances"
        / "five_cities.json"
    )

    if not instance_path.exists():
        raise FileNotFoundError(
            f"Could not find TSP instance:\n{instance_path}"
        )

    # --------------------------------------------------
    # 2. Load fixed-city instance
    # --------------------------------------------------

    instance = TSPInstance.from_json(instance_path)

    print("Project root:", PROJECT_ROOT)
    print("Instance:", instance.name)
    print("Number of cities:", instance.num_cities)

    print("\nCoordinates:")
    print(instance.coordinates)

    # --------------------------------------------------
    # 3. Create simulator
    # --------------------------------------------------

    simulator = TSPSimulator(instance)

    # --------------------------------------------------
    # 4. Initial state
    # --------------------------------------------------

    state = simulator.state()

    print("\nInitial state")
    print("-------------")
    print("Start city:", state["start_city"])
    print("Current city:", state["current_city"])
    print("Visited:", state["visited"])
    print("Available actions:", state["available_actions"])

    # --------------------------------------------------
    # 5. Demonstrate sequential action selection
    # --------------------------------------------------
    #
    # For now there is NO RL agent.
    #
    # We simply choose the first valid action to demonstrate
    # how an agent will eventually interact with the simulator.
    #
    # Later this section will become:
    #
    #     action = agent.select_action(state)
    #
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        available_actions = state["available_actions"]

        # If all cities have been visited, stop selecting cities.
        if not available_actions:
            break

        current_city = state["current_city"]

        # Temporary deterministic action selection.
        # This will later be replaced by an RL agent.
        selected_action = available_actions[0]

        print(
            f"Current city: {current_city} | "
            f"Available actions: {available_actions} | "
            f"Selected action: {selected_action}"
        )

        simulator.step(selected_action)

    # --------------------------------------------------
    # 6. Close the tour
    # --------------------------------------------------

    simulator.close_tour()

    # --------------------------------------------------
    # 7. Final result
    # --------------------------------------------------

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", simulator.tour)
    print("Closed:", simulator.done)
    print("Total distance:", simulator.total_distance)

    # --------------------------------------------------
    # 8. Visualize
    # --------------------------------------------------

    plot_tour(
        instance,
        simulator.tour,
        title="Fixed-City TSP Tour",
    )

    plt.show()


if __name__ == "__main__":
    main()
