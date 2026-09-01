
from pathlib import Path
import sys

from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.visualization import load_saved_simulation, animate_simulation


def main():
    instance = TSPInstance.from_json(
        PROJECT_ROOT / "instances" / "five_cities.json"
    )

    saved = load_saved_simulation(PROJECT_ROOT)

    if saved is None:
        raise RuntimeError(
            "No simulation exists. Run run_fixed_city.py first."
        )

    print("\nVisualizing saved simulation:")
    print("Start city:", saved["start_city"])
    print("Actions:", saved["actions"])
    print("Tour:", saved["tour"])
    print("Distance:", saved["total_distance"])

    fig, animation = animate_simulation(
        instance,
        saved["actions"],
        saved["start_city"],
    )

    display(HTML(animation.to_jshtml()))
    print("\n✓ Simulation video displayed.")


if __name__ == "__main__":
    main()

