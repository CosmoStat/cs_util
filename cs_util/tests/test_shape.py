"""UNIT TESTS FOR SHAPE SUBPACKAGE.

This module contains unit tests for the shape subpackage.

"""

import numpy as np
from numpy import testing as npt

from unittest import TestCase

from cs_util import shape


class ShapeTestCase(TestCase):
    """Test case for the ``shape`` module."""

    def setUp(self):
        """Set test parameter values.

        Axis ratio q = 1/3 gives the clean pair
        |e| = (1 - q^2) / (1 + q^2) = 0.8,
        |g| = (1 - q)   / (1 + q)   = 0.5.
        """
        self._e = 0.8
        self._g = 0.5

    def tearDown(self):
        """Unset test parameter values."""
        self._e = None
        self._g = None

    def test_known_pair(self):
        """Test the q = 1/3 distortion/shear pair, both directions."""
        g1, g2 = shape.e_to_g(self._e, 0.0)
        npt.assert_almost_equal(g1, self._g)
        npt.assert_almost_equal(g2, 0.0)
        e1, e2 = shape.g_to_e(self._g, 0.0)
        npt.assert_almost_equal(e1, self._e)
        npt.assert_almost_equal(e2, 0.0)

    def test_axis_ratio_magnitudes(self):
        """Test |e| and |g| against their axis-ratio closed forms."""
        q = np.array([0.1, 0.3, 0.5, 0.9])
        e_mag = (1 - q ** 2) / (1 + q ** 2)
        g_mag = (1 - q) / (1 + q)
        g1, g2 = shape.e_to_g(e_mag, np.zeros_like(e_mag))
        npt.assert_allclose(g1, g_mag)
        e1, e2 = shape.g_to_e(g_mag, np.zeros_like(g_mag))
        npt.assert_allclose(e1, e_mag)

    def test_round_trips(self):
        """Test that the converters compose to the identity, component-wise."""
        g1 = np.array([-0.4, -0.05, 0.0, 0.1, 0.45])
        g2 = np.array([0.2, 0.3, 0.0, -0.25, 0.1])
        e1, e2 = shape.g_to_e(g1, g2)
        g1b, g2b = shape.e_to_g(e1, e2)
        npt.assert_allclose(g1b, g1)
        npt.assert_allclose(g2b, g2)

        e1 = np.array([-0.6, 0.0, 0.3, 0.7])
        e2 = np.array([0.1, 0.0, -0.4, 0.2])
        g1, g2 = shape.e_to_g(e1, e2)
        e1b, e2b = shape.g_to_e(g1, g2)
        npt.assert_allclose(e1b, e1)
        npt.assert_allclose(e2b, e2)

    def test_orientation_preserved(self):
        """Test that the conversion scales both components by one factor."""
        e1, e2 = 0.36, 0.48  # |e| = 0.6, position angle fixed
        g1, g2 = shape.e_to_g(e1, e2)
        # g1 / g2 == e1 / e2: orientation unchanged by the magnitude rescale.
        npt.assert_almost_equal(g1 / g2, e1 / e2)

    def test_zero_maps_to_zero(self):
        """Test that a round object stays round under both converters."""
        npt.assert_array_equal(shape.e_to_g(0.0, 0.0), (0.0, 0.0))
        npt.assert_array_equal(shape.g_to_e(0.0, 0.0), (0.0, 0.0))

    def test_small_ellipticity_limit(self):
        """Test ``e -> 2 g`` and ``g -> e / 2`` in the round-object limit."""
        g = np.array([1e-4, 5e-4, 1e-3])
        e1, _ = shape.g_to_e(g, np.zeros_like(g))
        npt.assert_allclose(e1, 2 * g, rtol=1e-5)
        e = np.array([1e-4, 5e-4, 1e-3])
        g1, _ = shape.e_to_g(e, np.zeros_like(e))
        npt.assert_allclose(g1, e / 2, rtol=1e-5)

    def test_array_input(self):
        """Test that array input returns arrays of the same shape."""
        e1 = np.linspace(-0.7, 0.7, 11)
        e2 = np.linspace(0.0, 0.5, 11)
        g1, g2 = shape.e_to_g(e1, e2)
        self.assertEqual(g1.shape, e1.shape)
        self.assertEqual(g2.shape, e2.shape)
        # |g| < |e| < 1 for every input ellipticity.
        self.assertTrue(np.all(np.hypot(g1, g2) < np.hypot(e1, e2) + 1e-12))
