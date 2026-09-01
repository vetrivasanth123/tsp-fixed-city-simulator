
"""
Demonstration of the fixed-city TSP simulator.

The simulator performs the actions first and records every transition.
The recorded trajectory is then replayed as an animation.

The visualization shows:

1. Fixed city coordinates.
2. Random starting city.
3. Current city.
4. Available next-city actions.
5. Selected action.
6. Route after the action.
7. Distance added.
8. Final return to the starting city.
9. Complete tour and total distance.

No optimization or RL agent is used yet.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

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
    # 1. Locate instance
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

    # No fixed seed is supplied.
    #
    # Therefore every fresh simulator instance can select
    # a different starting city.

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
    # 5. Execute simulator actions
    # --------------------------------------------------

    print("\nAction sequence")
    print("---------------")

    while not simulator.done:

        state = simulator.state()

        available_actions = state["available_actions"]

        # All cities have been visited.
        if not available_actions:
            break

        current_city = state["current_city"]

        # --------------------------------------------------
        # Temporary action-selection policy.
        #
        # This is NOT RL.
        #
        # We simply select the first valid action so that
        # the simulator interface can be demonstrated.
        #
        # Later this becomes something like:
        #
        # selected_action = agent.select_action(state)
        #
        # --------------------------------------------------

        selected_action = available_actions[0]

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
    # 8. Create animation from the EXACT trajectory
    # --------------------------------------------------

    animation = animate_simulation(
        simulator,
        interval=1500,
        title="Fixed-City TSP Simulation",
    )

    # --------------------------------------------------
    # 9. Render animation in Jupyter / Colab
    # --------------------------------------------------

    print("\nRendering simulation animation...")

    html = animation.to_jshtml()

    display(HTML(html))

    # Keep a reference alive until rendering is complete.
    return animation


if __name__ == "__main__":
    main()

