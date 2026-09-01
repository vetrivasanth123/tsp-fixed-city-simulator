"""
Demonstration of the fixed-city TSP simulator.

The simulator runs first and records every action.

The recorded trajectory is then rendered immediately as an
animation showing exactly what happened.

No optimization or RL agent is used yet.
"""

from pathlib import Path
import sys
import random

from IPython.display import HTML, display


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


from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import animate_simulation


def main():

    # --------------------------------------------------
    # 1. Load fixed-city instance
    # --------------------------------------------------

    instance_path = (
        PROJECT_ROOT
        / "instances"
        / "five_cities.json"
    )

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
    # 2. Create simulator
    # --------------------------------------------------
    #
    # No seed here.
    #
    # Therefore the starting city is random every run.
    # --------------------------------------------------

    simulator = TSPSimulator(instance)

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
    # 3. Run the simulator
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        available_actions = (
            state["available_actions"]
        )

        if not available_actions:
            break

        current_city = (
            state["current_city"]
        )

        # --------------------------------------------------
        # TEMPORARY ACTION SELECTION
        #
        # This is NOT RL.
        #
        # It simply chooses randomly from the valid cities.
        #
        # Later this exact line becomes something like:
        #
        # selected_action = agent.select_action(state)
        # --------------------------------------------------

        selected_action = random.choice(
            available_actions
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
    # 4. Close tour
    # --------------------------------------------------

    simulator.close_tour()

    # --------------------------------------------------
    # 5. Final result
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
    # 6. Animate THIS EXACT simulation
    # --------------------------------------------------

    print(
        "\nRendering simulation animation..."
    )

    animation = animate_simulation(
        simulator,
        interval=1200,
        title="Fixed-City TSP Simulation",
    )

    # --------------------------------------------------
    # 7. Display immediately in Colab/Jupyter
    # --------------------------------------------------

    display(
        HTML(
            animation.to_jshtml()
        )
    )

    return animation


if __name__ == "__main__":
    main()
