"""Visualize the exact most recently completed TSP simulation."""

from pathlib import Path
import json
import sys

from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
from tsp.visualization import animate_simulation


def main() -> None:
    record_path = PROJECT_ROOT / ".tsp_last_simulation.json"

    if not record_path.exists():
        raise RuntimeError(
            "No simulation found. Run run_fixed_city.py first."
        )

    with open(record_path) as f:
        record = json.load(f)

    instance = TSPInstance.from_json(
        PROJECT_ROOT / "instances" / "five_cities.json"
    )

    print("Visualizing saved simulation:")
    print("Start city:", record["start_city"])
    print("Actions:", record["actions"])
    print("Tour:", record["tour"])
    print("Distance:", record["total_distance"])

    fig, animation = animate_simulation(
        instance=instance,
        actions=record["actions"],
        start_city=record["start_city"],
    )

    display(HTML(animation.to_jshtml()))


if __name__ == "__main__":
    main()
