
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsp.city_generator import CityLocationGenerator
from tsp.instance import TSPInstance


def main():
    parser = argparse.ArgumentParser(description="Generate a TSP instance.")
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--n-cities", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    coordinates = CityLocationGenerator(
        args.width, args.height, args.n_cities, args.seed
    ).generate_and_display()

    instance = TSPInstance(coordinates, name="generated_tsp")

    print("\nEuclidean distance matrix")
    print("----------------------")
    print(instance.cost_matrix)


if __name__ == "__main__":
    main()

