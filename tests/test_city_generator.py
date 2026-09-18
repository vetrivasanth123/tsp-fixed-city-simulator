"""Tests for city location generation."""

import numpy as np
import pytest

from tsp.city_generator import CityLocationGenerator

def test_invalid_dimensions():
with pytest.raises(ValueError):
CityLocationGenerator(0, 4, 5)

with pytest.raises(ValueError):
    CityLocationGenerator(4, 0, 5)

def test_invalid_city_count():
with pytest.raises(ValueError):
CityLocationGenerator(4, 4, 1)

def test_generates_valid_unique_coordinates():
width = 8
height = 8
n_cities = 5
city_radius = 1.0

generator = CityLocationGenerator(
    width,
    height,
    n_cities,
    seed=42,
    city_shape="circle",
    city_radius=city_radius,
)

coordinates = generator.generate()

assert len(coordinates) == n_cities
assert len(generator.coordinates) == n_cities
assert len({tuple(c) for c in coordinates}) == n_cities

coordinates = np.asarray(coordinates)

assert np.all(coordinates[:, 0] >= 0)
assert np.all(coordinates[:, 0] <= width)
assert np.all(coordinates[:, 1] >= 0)
assert np.all(coordinates[:, 1] <= height)

def test_reproducible_with_seed():
width = 8
height = 8
n_cities = 5
city_radius = 1.0

a = CityLocationGenerator(
    width,
    height,
    n_cities,
    seed=42,
    city_shape="circle",
    city_radius=city_radius,
).generate()

b = CityLocationGenerator(
    width,
    height,
    n_cities,
    seed=42,
    city_shape="circle",
    city_radius=city_radius,
).generate()

assert a == b

def test_different_seeds_generate_different_locations():
width = 8
height = 8
n_cities = 5
city_radius = 1.0

a = CityLocationGenerator(
    width,
    height,
    n_cities,
    seed=42,
    city_shape="circle",
    city_radius=city_radius,
).generate()

b = CityLocationGenerator(
    width,
    height,
    n_cities,
    seed=43,
    city_shape="circle",
    city_radius=city_radius,
).generate()

assert a != b

def test_get_coordinates():
width = 8
height = 8
n_cities = 5
city_radius = 1.0

generator = CityLocationGenerator(
    width,
    height,
    n_cities,
    seed=42,
    city_shape="circle",
    city_radius=city_radius,
)

coordinates = generator.generate()

assert generator.get_coordinates() == coordinates
