"""
Basic demonstration of the fixed-city TSP simulator.

This example:
1. Loads the five-city instance.
2. Creates the simulator.
3. Executes a sample tour.
4. Reports the tour distance.
5. Visualizes the resulting tour.

This is a demonstration only; it does not perform optimization.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from tsp.instance import TSPInstance
from tsp.simulator import TSPSimulator
from tsp.visualization import plot_tour


def main():
    # Locate the fixed five-city instance.
    project_root = Path(__file__).resolve().parents[1]
    instance_path = project_root / "instances" / "five_cities.json"

    # Load the TSP instance.
    instance = TSPInstance.from_json(instance_path)

    print("TSP instance:", instance.name)
    print("Number of cities:", instance.num_cities)
    print("Distance matrix:")
    print(instance.distance_matrix)

    # Create the fixed-city simulator.
    simulator = TSPSimulator(instance)

    # Example tour.
    # The simulator will later become the interface used by RL.
    tour = [0, 1, 2, 3, 4]

    # Execute the tour.
    for city in tour:
        simulator.step(city)

    # Complete the tour by returning to the starting city.
    simulator.close_tour()

    print("\nTour:", simulator.tour)
    print("Total distance:", simulator.total_distance)

    # Visualize.
    plot_tour(
        instance,
        simulator.tour,
        title="Five-City TSP Example",
    )

    plt.show()


if __name__ == "__main__":
    main()
