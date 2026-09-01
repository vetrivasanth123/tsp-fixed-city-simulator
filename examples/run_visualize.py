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

    instance_path = (
        PROJECT_ROOT
        / "instances"
        / "five_cities.json"
    )

    run_path = (
        PROJECT_ROOT
        / "outputs"
        / "fixed_city_run.json"
    )

    if not run_path.exists():
        raise FileNotFoundError(
            "No recorded simulation found. "
            "Run examples/run_fixed_city.py first."
        )

    instance = TSPInstance.from_json(instance_path)

    data = json.loads(
        run_path.read_text()
    )

    start_city = data["start_city"]
    actions = data["actions"]

    print("Recorded simulation")
    print("-------------------")
    print("Start city:", start_city)
    print("Actions:", actions)
    print("Tour:", data["tour"])
    print("Distance:", data["total_distance"])

    fig, animation = animate_simulation(
        instance=instance,
        start_city=start_city,
        actions=actions,
        interval=1000,
    )

    display(
        HTML(animation.to_jshtml())
    )


if __name__ == "__main__":
    main()
