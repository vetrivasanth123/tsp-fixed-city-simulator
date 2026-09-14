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
        self.rng = random.Random(seed)
        self.coordinates = []

    def generate(self):
        """Generate and record all city coordinates."""
        self.coordinates = []

        while len(self.coordinates) < self.n_cities:
            point = (
                self.rng.uniform(0, self.width),
                self.rng.uniform(0, self.height),
            )

            if point not in self.coordinates:
                self.coordinates.append(point)

        return [list(point) for point in self.coordinates]

    def get_coordinates(self):
        """Return the currently generated coordinates."""
        return [list(point) for point in self.coordinates]
