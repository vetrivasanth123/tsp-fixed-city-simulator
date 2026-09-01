"""
Demonstration of the fixed-city TSP simulator.

This example:

1. Loads the fixed five-city instance.
2. Creates the simulator.
3. Randomly selects a starting city.
4. Displays the initial state.
5. Randomly selects a valid next city.
6. Records every simulator transition.
7. Closes the tour.
8. Displays the final result.
9. Animates the exact simulator trajectory.

No optimization or RL agent is used yet.

The random action policy is only a temporary demonstration
of how an eventual RL agent will interact with the simulator.
"""

from pathlib import Path
import sys
import random

import matplotlib.pyplot as plt

# --------------------------------------------------
# Locate project root
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# --------------------------------------------------
# Imports
# --------------------------------------------------

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import animate_simulation


def main():

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
    # 2. Load fixed instance
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
    #
    # No seed is supplied here.
    #
    # Therefore the starting city is randomly selected
    # every time this script is executed.
    #
    # --------------------------------------------------

    simulator = TSPSimulator(
        instance
    )

    # Separate random generator for the temporary
    # demonstration policy.
    #
    # IMPORTANT:
    # This is NOT an RL agent.
    #
    action_rng = random.Random()

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
    # 5. Execute simulator actions
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        available_actions = (
            state["available_actions"]
        )

        # No unvisited cities remain.
        if not available_actions:
            break

        current_city = (
            state["current_city"]
        )

        # --------------------------------------------------
        # TEMPORARY ACTION POLICY
        # --------------------------------------------------
        #
        # Randomly select one valid city.
        #
        # Later this becomes:
        #
        # selected_action = agent.select_action(state)
        #
        # The simulator itself does not care whether
        # the action came from random selection, a
        # heuristic, or an RL agent.
        # --------------------------------------------------

        selected_action = (
            action_rng.choice(
                available_actions
            )
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
    # 8. Create animation
    # --------------------------------------------------
    #
    # IMPORTANT:
    #
    # animate_simulation DOES NOT make new decisions.
    #
    # It reads simulator.trajectory(), which contains
    # exactly the actions that were already executed.
    #
    # Therefore the animation and printed simulation
    # are guaranteed to describe the same trajectory.
    # --------------------------------------------------

    animation = animate_simulation(
        simulator,
        interval=1500,
        title="Fixed-City TSP Simulation",
    )

    # --------------------------------------------------
    # 9. Display animation in Jupyter / Colab
    # --------------------------------------------------

    try:

        from IPython.display import HTML, display

        display(
            HTML(
                animation.to_jshtml()
            )
        )

    except ImportError:

        plt.show()


if __name__ == "__main__":
    main()
