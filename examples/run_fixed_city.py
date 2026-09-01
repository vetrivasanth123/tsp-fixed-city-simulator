```python
"""
Demonstration of the fixed-city TSP simulator.

The simulator:
1. Loads a fixed set of cities.
2. Randomly selects the starting city.
3. Exposes available cities as possible actions.
4. Executes a simple action-selection policy.
5. Completes the tour.
6. Reports the final distance.
7. Visualizes the resulting tour.

This is NOT an RL agent.
The action-selection logic is deliberately simple so that
the simulator interface can be verified independently before
adding an RL agent.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import plot_tour


def main() -> None:
    # --------------------------------------------------
    # 1. Locate and load the fixed five-city instance
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]
    instance_path = project_root / "instances" / "five_cities.json"

    instance = TSPInstance.from_json(instance_path)

    print("TSP instance:", instance.name)
    print("Number of cities:", instance.num_cities)

    print("\nCoordinates:")
    print(instance.coordinates)

    # --------------------------------------------------
    # 2. Create simulator
    # --------------------------------------------------

    # The simulator randomly selects the starting city.
    simulator = TSPSimulator(instance, seed=42)

    state = simulator.reset()

    print("\nInitial state")
    print("-------------")
    print("Start city:", state["start_city"])
    print("Current city:", state["current_city"])
    print("Visited:", state["visited"])
    print("Available actions:", state["available_actions"])

    # --------------------------------------------------
    # 3. Construct a tour using available actions
    # --------------------------------------------------

    print("\nAction sequence")
    print("--------------")

    while not simulator.done:

        available_actions = simulator.available_actions()

        # If every city has been visited, close the tour.
        if not available_actions:
            break

        # --------------------------------------------------
        # Temporary demonstration policy
        # --------------------------------------------------
        # Select the first available city.
        #
        # IMPORTANT:
        # This is where an RL agent can later be connected.
        # For example:
        #
        # action = agent.select_action(state)
        #
        # For now we deliberately use a simple deterministic
        # policy so the simulator can be tested independently.
        # --------------------------------------------------

        action = available_actions[0]

        print(
            f"Current city: {simulator.current_city} "
            f"| Available actions: {available_actions} "
            f"| Selected action: {action}"
        )

        state = simulator.step(action)

    # --------------------------------------------------
    # 4. Close the tour
    # --------------------------------------------------

    simulator.close_tour()

    # --------------------------------------------------
    # 5. Report final result
    # --------------------------------------------------

    print("\nFinal result")
    print("------------")
    print("Start city:", simulator.start_city)
    print("Tour:", simulator.tour)
    print("Closed:", simulator.done)
    print("Total distance:", simulator.total_distance)

    # --------------------------------------------------
    # 6. Visualize the final tour
    # --------------------------------------------------

    plot_tour(
        instance,
        simulator.tour,
        title="Fixed-City TSP Tour",
    )

    plt.show()


if __name__ == "__main__":
    main()
```
