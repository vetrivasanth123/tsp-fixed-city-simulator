
"""
Demonstration of the fixed-city TSP simulator.

The complete workflow is:

    fixed city instance
          ↓
    simulator reset
          ↓
    random starting city
          ↓
    random valid action
          ↓
    simulator.step(action)
          ↓
    record transition
          ↓
    repeat
          ↓
    close tour
          ↓
    replay exact trajectory as an animation

No optimization or RL agent is used yet.

The random action policy is only a temporary placeholder.
Later it will be replaced by an RL agent.
"""

from pathlib import Path
import sys
import random

from IPython.display import HTML, display

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
    # 2. Load fixed-city instance
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

    # No seed is supplied.
    #
    # Therefore the starting city is random.

    simulator = TSPSimulator(
        instance
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
    # 5. Execute actions
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
        # TEMPORARY ACTION POLICY
        # --------------------------------------------------
        #
        # This is NOT RL.
        #
        # Choose one of the currently valid cities randomly.
        #
        # Later this becomes:
        #
        # selected_action = agent.select_action(state)
        #
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

        # --------------------------------------------------
        # Actual simulator transition
        # --------------------------------------------------

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
    # 8. Create animation from exact trajectory
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
    # 9. Render as an actual HTML5 video
    # --------------------------------------------------

    video_html = animation.to_html5_video()

    display(
        HTML(video_html)
    )

    return animation


if __name__ == "__main__":
    main()

