import random


class CityLocationGenerator:
    """Generate unique city coordinates within a rectangular region."""

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

            if coordinate not in self.coordinates:
                self.coordinates.append(coordinate)

        return [list(c) for c in self.coordinates]

    def get_coordinates(self):
        """Return the generated coordinates."""
        return [list(c) for c in self.coordinates]
