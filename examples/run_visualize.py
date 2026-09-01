from pathlib import Path
import sys

from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.simulator import get_last_simulator
from tsp.visualization import animate_simulation


def main():

    simulator = get_last_simulator()

    if simulator is None:
        raise RuntimeError(
            "No simulation exists in memory. "
            "Run run_fixed_city.py first."
        )

    print("Visualizing the last simulation")
    print("--------------------------------")
    print("Start city:", simulator.start_city)
    print("Tour:", simulator.tour)
    print("Distance:", simulator.total_distance)

    animation = animate_simulation(
        simulator,
        interval=1000,
    )

    display(
        HTML(animation.to_jshtml())
    )


if __name__ == "__main__":
    main()
