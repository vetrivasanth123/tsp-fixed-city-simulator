
import argparse
from tsp.city_generator import CityLocationGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate TSP city locations.")
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--n-cities", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generator = CityLocationGenerator(
        args.width, args.height, args.n_cities, args.seed
    )
    generator.generate_and_display()


if __name__ == "__main__":
    main()

