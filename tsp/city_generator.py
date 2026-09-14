import random


class CityLocationGenerator:
    """
    Generates a reproducible set of city coordinates sequentially.

    The generator constructs the complete location design before the
    coordinates are passed to TSPInstance.
    """

    def __init__(self, width, height, n_cities, seed=None):
        if width <= 0:
            raise ValueError("width must be positive.")

        if height <= 0:
            raise ValueError("height must be positive.")

        if n_cities < 2:
            raise ValueError("n_cities must be at least 2.")

        self.width = float(width)
        self.height = float(height)
        self.n_cities = int(n_cities)
        self.seed = seed

        self.rng = random.Random(seed)
        self.coordinates = []

    def generate(self):
        """
        Generate all city coordinates sequentially.

        Returns:
            list[list[float]]: Generated city coordinates.
        """
        self.coordinates = []

        for _ in range(self.n_cities):
            x = self.rng.uniform(0.0, self.width)
            y = self.rng.uniform(0.0, self.height)

            coordinate = [x, y]

            # Record the newly generated city as part of the
            # accumulated design.
            self.coordinates.append(coordinate)

        return self.coordinates.copy()

    def get_coordinates(self):
        """
        Return the currently recorded city coordinates.
        """
        return self.coordinates.copy()
