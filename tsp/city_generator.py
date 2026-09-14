
import random


class CityLocationGenerator:
    """Sequentially generate unique city coordinates."""

    def __init__(self, width, height, n_cities, seed=None):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if n_cities < 2:
            raise ValueError("n_cities must be at least 2.")

        self.width = float(width)
        self.height = float(height)
        self.n_cities = int(n_cities)
        self.rng = random.Random(seed)
        self.coordinates = []

    def generate(self):
        self.coordinates = []

        while len(self.coordinates) < self.n_cities:
            point = (
                self.rng.uniform(0, self.width),
                self.rng.uniform(0, self.height),
            )
            if point not in self.coordinates:
                self.coordinates.append(point)

        return [list(p) for p in self.coordinates]

    def generate_and_display(self):
        coordinates = self.generate()

        print("\nCity generation")
        print("----------------")
        for i, point in enumerate(coordinates, 1):
            print(
                f"City {i-1}: ({point[0]:.4f}, {point[1]:.4f}) "
                f"| Recorded: {i}/{self.n_cities}"
            )

        print("\nFinal city locations")
        print("--------------------")
        for i, point in enumerate(coordinates):
            print(f"City {i}: {point}")

        print(f"\nGeneration complete: {self.n_cities} cities")
        return coordinates

    def get_coordinates(self):
        return [list(p) for p in self.coordinates]

