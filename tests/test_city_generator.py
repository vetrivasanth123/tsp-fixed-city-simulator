
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
    generator = CityLocationGenerator(4, 4, 5, seed=42)
    coordinates = generator.generate()

    assert len(coordinates) == 5
    assert len(generator.coordinates) == 5
    assert len({tuple(c) for c in coordinates}) == 5

    coordinates = np.asarray(coordinates)
    assert np.all(coordinates[:, 0] >= 0)
    assert np.all(coordinates[:, 0] <= 4)
    assert np.all(coordinates[:, 1] >= 0)
    assert np.all(coordinates[:, 1] <= 4)


def test_reproducible_with_seed():
    a = CityLocationGenerator(4, 4, 5, seed=42).generate()
    b = CityLocationGenerator(4, 4, 5, seed=42).generate()

    assert a == b


def test_different_seeds_generate_different_locations():
    a = CityLocationGenerator(4, 4, 5, seed=42).generate()
    b = CityLocationGenerator(4, 4, 5, seed=43).generate()

    assert a != b


def test_get_coordinates():
    generator = CityLocationGenerator(4, 4, 5, seed=42)
    coordinates = generator.generate()

    assert generator.get_coordinates() == coordinates
