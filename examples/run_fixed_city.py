"""
Demonstration of the fixed-city TSP simulator.

The example:

1. Automatically finds the project root.
2. Loads the fixed five-city instance.
3. Creates the simulator with a random starting city.
4. Displays the initial state.
5. Selects valid actions sequentially.
6. Displays every state transition.
7. Closes the tour.
8. Reports the final result.
9. Displays the final static tour.

No RL agent is used yet.
"""

from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt

# --------------------------------------------------
# Locate project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import plot_tour


def main() -> None:

    # --------------------------------------------------
    # 1. Locate fixed instance
    # --------------------------------------------------

    instance_path = (
        PROJECT_ROOT
        / "instances"
        / "five_cities.json"
    )

    if not instance_path.exists():
        raise FileNotFoundError(
            f"Could not find TSP instance:\n"
            f"{instance_path}"
        )

    # --------------------------------------------------
    # 2. Load instance
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
    #
    # No seed is supplied.
    #
    # Therefore each new simulator instance can choose
    # a different random starting city.
    #
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
    # 5. Sequential action selection
    # --------------------------------------------------
    #
    # IMPORTANT:
    #
    # The simulator chooses the RANDOM START CITY.
    #
    # The next-city selection below is only a temporary
    # demonstration policy.
    #
    # It randomly selects one of the valid actions.
    #
    # Later this will become the RL agent's action:
    #
    #     action = agent.select_action(state)
    #
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        available_actions = state["available_actions"]

        if not available_actions:
            break

        current_city = state["current_city"]

        # Temporary random policy.
        selected_action = random.choice(
            available_actions
        )

        print(
            f"Current city: {current_city} | "
            f"Available actions: {available_actions} | "
            f"Selected action: {selected_action}"
        )

        simulator.step(selected_action)

    # --------------------------------------------------
    # 6. Close tour
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
    # 8. Static visualization
    # --------------------------------------------------

    plot_tour(
        instance,
        simulator.tour,
        title="Fixed-City TSP Tour",
    )

    plt.show()


if __name__ == "__main__":
    main()
