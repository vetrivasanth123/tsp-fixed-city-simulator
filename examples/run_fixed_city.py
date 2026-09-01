"""
Demonstration of the fixed-city TSP simulator.

The demonstration:

1. Loads the fixed five-city instance.
2. Creates the simulator.
3. Randomly selects a starting city.
4. Displays the initial state.
5. Selects valid actions sequentially.
6. Records every simulator transition.
7. Closes the tour.
8. Displays the final result.
9. Animates the exact simulator trajectory.

No optimization or RL agent is used yet.

The action-selection section is deliberately separated from
the simulator so that a future RL agent can replace it directly.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

# --------------------------------------------------
# Locate project root
# --------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# --------------------------------------------------
# Project imports
# --------------------------------------------------

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import animate_simulation


def main() -> None:

    # --------------------------------------------------
    # 1. Locate instance
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
    # 2. Load fixed cities
    # --------------------------------------------------

    instance = TSPInstance.from_json(
        instance_path
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Instance:",
        instance.name,
    )

    print(
        "Number of cities:",
        instance.num_cities,
    )

    print("\nCoordinates:")
    print(instance.coordinates)

    # --------------------------------------------------
    # 3. Create simulator
    # --------------------------------------------------

    simulator = TSPSimulator(
        instance,
        seed=42,
    )

    # --------------------------------------------------
    # 4. Initial state
    # --------------------------------------------------

    state = simulator.state()

    print("\nInitial state")
    print("-------------")

    print(
        "Start city:",
        state["start_city"],
    )

    print(
        "Current city:",
        state["current_city"],
    )

    print(
        "Visited:",
        state["visited"],
    )

    print(
        "Available actions:",
        state["available_actions"],
    )

    # --------------------------------------------------
    # 5. Sequential action selection
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        available_actions = (
            state["available_actions"]
        )

        # ----------------------------------------------
        # All cities have been visited.
        # ----------------------------------------------

        if not available_actions:
            break

        current_city = (
            state["current_city"]
        )

        # --------------------------------------------------
        # TEMPORARY ACTION SELECTION
        # --------------------------------------------------
        #
        # This is NOT an optimization algorithm.
        #
        # It is only a placeholder demonstrating how
        # an agent will interact with the simulator.
        #
        # Later:
        #
        #     selected_action = agent.select_action(state)
        #
        # --------------------------------------------------

        selected_action = (
            available_actions[0]
        )

        print(
            f"Current city: {current_city} | "
            f"Available actions: "
            f"{available_actions} | "
            f"Selected action: "
            f"{selected_action}"
        )

        simulator.step(
            selected_action
        )

    # --------------------------------------------------
    # 6. Close tour
    # --------------------------------------------------

    simulator.close_tour()

    # --------------------------------------------------
    # 7. Final result
    # --------------------------------------------------

    print("\nFinal result")
    print("------------")

    print(
        "Start city:",
        simulator.start_city,
    )

    print(
        "Tour:",
        simulator.tour,
    )

    print(
        "Closed:",
        simulator.done,
    )

    print(
        "Total distance:",
        simulator.total_distance,
    )

    # --------------------------------------------------
    # 8. Animate exact simulator trajectory
    # --------------------------------------------------

    animation = animate_simulation(
        simulator,
        interval=1200,
        title="Fixed-City TSP Simulation",
    )

    # Keep a reference to the animation until plt.show().
    _ = animation

    plt.show()


if __name__ == "__main__":
    main()
