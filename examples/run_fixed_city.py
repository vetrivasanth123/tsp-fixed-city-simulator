"""
Demonstration of the fixed-city TSP simulator.

The simulator:
1. Loads fixed city coordinates.
2. Randomly selects a starting city.
3. Exposes available actions.
4. Selects actions sequentially.
5. Records every transition.
6. Closes the tour.
7. Displays the exact simulator trajectory as an animation.

No RL or optimization is used yet.
"""

from pathlib import Path
import sys

from IPython.display import HTML, display

# --------------------------------------------------
# Locate project root
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


def main() -> None:

    # --------------------------------------------------
    # 1. Locate fixed-city instance
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

    instance = TSPInstance.from_json(
        instance_path
    )

    print(
        "Project root:",
        PROJECT_ROOT
    )

    print(
        "Instance:",
        instance.name
    )

    print(
        "Number of cities:",
        instance.num_cities
    )

    print("\nCoordinates:")
    print(instance.coordinates)

    # --------------------------------------------------
    # 3. Create simulator
    # --------------------------------------------------

    simulator = TSPSimulator(
        instance,
        seed=42
    )

    # --------------------------------------------------
    # 4. Initial state
    # --------------------------------------------------

    state = simulator.state()

    print("\nInitial state")
    print("-------------")

    print(
        "Start city:",
        state["start_city"]
    )

    print(
        "Current city:",
        state["current_city"]
    )

    print(
        "Visited:",
        state["visited"]
    )

    print(
        "Available actions:",
        state["available_actions"]
    )

    # --------------------------------------------------
    # 5. Execute simulator
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
        # Temporary action selection
        #
        # This will later be replaced by:
        #
        # action = agent.select_action(state)
        # --------------------------------------------------

        selected_action = (
            available_actions[0]
        )

        print(
            f"Current city: {current_city} | "
            f"Available actions: {available_actions} | "
            f"Selected action: {selected_action}"
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
        simulator.start_city
    )

    print(
        "Tour:",
        simulator.tour
    )

    print(
        "Closed:",
        simulator.done
    )

    print(
        "Total distance:",
        simulator.total_distance
    )

    # --------------------------------------------------
    # 8. Create animation
    # --------------------------------------------------

    animation = animate_simulation(
        simulator,
        interval=1200,
        title="Fixed-City TSP Simulation"
    )

    # --------------------------------------------------
    # 9. Display animation in Colab/Jupyter
    # --------------------------------------------------

    display(
        HTML(
            animation.to_jshtml()
        )
    )


if __name__ == "__main__":
    main()
