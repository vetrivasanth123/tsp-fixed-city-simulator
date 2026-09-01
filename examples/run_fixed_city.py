"""
Demonstration of the fixed-city TSP simulator.

The demonstration:

1. Loads the fixed five-city instance.
2. Creates the simulator.
3. Randomly selects a starting city.
4. Displays the initial simulator state.
5. Selects one valid action at a time.
6. Executes each action inside the animation.
7. Visually shows each selected city and connecting edge.
8. Automatically closes the tour.
9. Reports the final tour and distance.

There is no RL agent yet.

The action-selection function is deliberately separated so that
it can later be replaced by an RL policy without changing the
simulator or visualization architecture.
"""

from pathlib import Path
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
from tsp.visualization import animate_simulation


def select_action(state: dict) -> int:
    """
    Temporary action-selection policy.

    For now we simply select the first available city.

    Later this function will be replaced by something such as:

        action = agent.select_action(state)

    or an RL policy.
    """

    available_actions = state["available_actions"]

    if not available_actions:
        raise RuntimeError(
            "No available actions remain."
        )

    return available_actions[0]


def main() -> None:

    # --------------------------------------------------
    # 1. Load fixed five-city instance
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

    # Use a seed if you want reproducible demonstrations.
    #
    # Remove/change the seed later when random starts are desired.

    simulator = TSPSimulator(
        instance,
        seed=42,
    )

    # --------------------------------------------------
    # 3. Display initial state
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
    # 4. Create animated simulation
    # --------------------------------------------------

    print("\nStarting animated simulation...")

    animation = animate_simulation(
        simulator=simulator,
        action_selector=select_action,
        interval=1200,
        title="Fixed-City TSP Simulation",
    )

    # Keep the animation alive.
    # In a normal Python environment this opens the Matplotlib
    # animation window.

    plt.show()

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


if __name__ == "__main__":
    main()
