"""UNIT TESTS FOR SIZE SUBPACKAGE.

This module contains unit tests for the size subpackage.

"""

import numpy as np
from numpy import testing as npt

from unittest import TestCase

from cs_util import size


class SizeTestCase(TestCase):
    """Test case for the ``size`` module."""

    def setUp(self):
        """Set test parameter values."""
        # Unit-sigma Gaussian: T = 2, r50 = 1.17741, FWHM = 2.35482
        self._sigma = 1.0
        self._T = 2.0
        self._r50 = 1.1774100226
        self._fwhm = 2.3548200450

    def tearDown(self):
        """Unset test parameter values."""
        self._sigma = None
        self._T = None
        self._r50 = None
        self._fwhm = None

    def test_constants(self):
        """Test the module constants against their closed forms."""
        npt.assert_almost_equal(size.SIGMA_TO_R50, np.sqrt(2 * np.log(2)))
        npt.assert_almost_equal(size.SIGMA_TO_FWHM, 2 * np.sqrt(2 * np.log(2)))
        npt.assert_almost_equal(size.SIGMA_TO_R50, 1.17741, decimal=5)
        npt.assert_almost_equal(size.SIGMA_TO_FWHM, 2.35482, decimal=5)

    def test_unit_sigma_values(self):
        """Test all conversions on the unit-sigma Gaussian."""
        npt.assert_almost_equal(size.T_to_sigma(self._T), self._sigma)
        npt.assert_almost_equal(size.sigma_to_T(self._sigma), self._T)
        npt.assert_almost_equal(size.sigma_to_r50(self._sigma), self._r50)
        npt.assert_almost_equal(size.sigma_to_fwhm(self._sigma), self._fwhm)
        npt.assert_almost_equal(size.T_to_r50(self._T), self._r50)
        npt.assert_almost_equal(size.T_to_fwhm(self._T), self._fwhm)

    def test_T_to_r50_closed_form(self):
        """Test ``T_to_r50(T) == sqrt(ln 2 * T)``."""
        T = np.array([0.05, 0.18, 2.0, 10.0])
        npt.assert_allclose(size.T_to_r50(T), np.sqrt(np.log(2) * T))

    def test_inverse_direct_values(self):
        """Test each inverse converter directly on the unit-sigma case."""
        npt.assert_almost_equal(size.r50_to_sigma(self._r50), self._sigma)
        npt.assert_almost_equal(size.fwhm_to_sigma(self._fwhm), self._sigma)
        npt.assert_almost_equal(size.r50_to_T(self._r50), self._T)

    def test_nonpositive_T(self):
        """Test that T = 0 maps to size 0 and negative T to NaN."""
        npt.assert_equal(size.T_to_sigma(0.0), 0.0)
        npt.assert_equal(size.T_to_r50(0.0), 0.0)
        npt.assert_equal(size.T_to_fwhm(0.0), 0.0)
        with np.errstate(invalid="ignore"):
            self.assertTrue(np.isnan(size.T_to_r50(-1.0)))

    def test_round_trips(self):
        """Test that inverse pairs compose to the identity."""
        values = np.array([0.01, 0.1, 1.0, 7.5])
        npt.assert_allclose(size.r50_to_T(size.T_to_r50(values)), values)
        npt.assert_allclose(size.T_to_r50(size.r50_to_T(values)), values)
        npt.assert_allclose(
            size.sigma_to_T(size.T_to_sigma(values)), values
        )
        npt.assert_allclose(
            size.r50_to_sigma(size.sigma_to_r50(values)), values
        )
        npt.assert_allclose(
            size.fwhm_to_sigma(size.sigma_to_fwhm(values)), values
        )

    def test_fwhm_is_twice_r50(self):
        """Test ``FWHM = 2 r50`` for a Gaussian, via both paths."""
        T = np.array([0.05, 0.18, 2.0])
        npt.assert_allclose(size.T_to_fwhm(T), 2 * size.T_to_r50(T))

    def test_array_input(self):
        """Test that array input returns arrays of the same shape."""
        T = np.linspace(0.01, 5.0, 11)
        r50 = size.T_to_r50(T)
        self.assertEqual(r50.shape, T.shape)
        self.assertTrue(np.all(np.diff(r50) > 0))
