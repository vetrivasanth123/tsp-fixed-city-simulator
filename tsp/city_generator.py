import random


class CityLocationGenerator:
    """Generate unique city coordinates sequentially."""

    def __init__(self, width, height, n_cities, seed=None):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if n_cities < 2:
            raise ValueError("n_cities must be at least 2.")

        self.width = float(width)
        self.height = float(height)
        self.n_cities = int(n_cities)
        self.seed = seed
        self._rng = random.Random(seed)
        self.coordinates = []

    def generate(self):
        """Generate and record all city coordinates sequentially."""
        self.coordinates = []

        while len(self.coordinates) < self.n_cities:
            coordinate = (
                self._rng.uniform(0.0, self.width),
                self._rng.uniform(0.0, self.height),
            )

            if coordinate in self.coordinates:
                continue

            self.coordinates.append(coordinate)

        return [list(c) for c in self.coordinates]

    def generate_and_display(self):
        """Generate cities and display the construction history."""
        coordinates = self.generate()

        print("\nCity generation")
        print("----------------")

        for i, coordinate in enumerate(coordinates):
            print(
                f"City {i}: "
                f"({coordinate[0]:.4f}, {coordinate[1]:.4f}) "
                f"| Recorded cities: {i + 1}/{self.n_cities}"
            )

        print("\nFinal city locations")
        print("--------------------")

        for i, coordinate in enumerate(coordinates):
            print(f"City {i}: {coordinate}")

        print(f"\nGeneration complete: {len(coordinates)} cities")

        return coordinates

    def get_coordinates(self):
        return [list(c) for c in self.coordinates]
