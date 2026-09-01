"""
Run the fixed-city TSP simulator and immediately visualize
the exact simulator trajectory.

No RL or optimization is used yet.
"""

from pathlib import Path
import sys

from IPython.display import HTML, display

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Imports
# --------------------------------------------------

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import animate_simulation


def main():

    # --------------------------------------------------
    # 1. Load fixed instance
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
    #
    # No seed -> different starting city can be selected
    # on different executions.
    # --------------------------------------------------

    simulator = TSPSimulator(instance)

    # --------------------------------------------------
    # 3. Initial state
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
    # 4. Execute simulator
    #
    # This is the temporary action policy.
    #
    # Later:
    #
    #     selected_action = agent.select_action(state)
    #
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        actions = state["available_actions"]

        if not actions:
            break

        current_city = state["current_city"]

        # Temporary deterministic policy.
        selected_action = actions[0]

        print(
            f"Current city: {current_city} | "
            f"Available actions: {actions} | "
            f"Selected action: {selected_action}"
        )

        simulator.step(selected_action)

    # --------------------------------------------------
    # 5. Close tour
    # --------------------------------------------------

    simulator.close_tour()

    # --------------------------------------------------
    # 6. Final result
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
    # 7. Animate EXACT trajectory
    # --------------------------------------------------

    print("\nRendering simulation animation...")

    animation = animate_simulation(
        simulator,
        interval=1200,
        title="Fixed-City TSP Simulation",
    )

    # --------------------------------------------------
    # 8. Display immediately in Colab/Jupyter
    # --------------------------------------------------

    display(
        HTML(
            animation.to_jshtml()
        )
    )

    return animation


if __name__ == "__main__":
    main()
