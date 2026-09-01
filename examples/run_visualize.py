
from pathlib import Path
import sys
import importlib

from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsp.instance import TSPInstance
import tsp.visualization as visualization

importlib.reload(visualization)


def main():
    saved = visualization.load_saved_simulation(PROJECT_ROOT)

    if saved is None:
        raise RuntimeError(
            "No simulation exists. Run run_fixed_city.py first."
        )

    instance = TSPInstance.from_json(
        PROJECT_ROOT / "instances" / f"{saved['instance']}.json"
    )

    print("Visualizing saved simulation:")
    print("Saved instance:", saved["instance"])
    print("Start city:", saved["start_city"])
    print("Actions:", saved["actions"])
    print("Tour:", saved["tour"])
    print("Cost:", saved["total_cost"])

    print("\nVisualization cost matrix:")
    print(instance.cost_matrix)

    _, animation = visualization.animate_simulation(
        instance,
        saved["actions"],
        saved["start_city"],
    )

    display(HTML(animation.to_html5_video()))


if __name__ == "__main__":
    main()

