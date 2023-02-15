# -*- coding: utf-8 -*-

"""UNIT TESTS FOR COSMO SUBPACKAGE.

This module contains unit tests for the cosmo subpackage.

"""

import os

import numpy as np
import pyccl as ccl

from astropy import units
from astropy.cosmology import FlatLambdaCDM

from numpy import testing as npt

from unittest import TestCase

from cs_util import cosmo


class CosmoTestCase(TestCase):
    """Test case for the ``cosmo` module."""

    def setUp(self):
        """Set test parameter values."""
        self._z_source = 0.8
        self._z_lens = 0.5
        self._z_source_arr = [0.4, 0.6, 0.8, 0.9]
        self._nz_source_arr = [0.5, 0.6, 2.2, 1.6]
        self._cosmo = ccl.core.CosmologyVanillaLCDM()
        self._sigma_crit_value = 3920.1478
        # Value verified with package dsigma as 3919.700

        self._sigma_crit_value_eff = 3917.2681
        self._sigma_crit_value_eff_m1 = 0.0002267

        self._sigma_crit_unit = units.Msun / units.pc**2
        self._d_source = 1617.9195 * units.Mpc
        self._d_lens = 1315.3937 * units.Mpc

    def tearDown(self):
        """Unset test parameter values."""
        self._z_source = None
        self._z_lens = None
        self._cosmo = None
        self._sigma_crit_value = None
        self._sigma_crit_unit = None
        self._d_source = None
        self._d_lens = None
        self._ds_cosmo = None

    def test_sigma_crit(self):
        """Test ``cs_util.cosmo.sigma_crit`` method.

        """
        sigma_crit = cosmo.sigma_crit(
            self._z_lens,
            self._z_source,
            self._cosmo,
        )
        # Test return value
        npt.assert_almost_equal(
            sigma_crit.value,
            self._sigma_crit_value,
            decimal=4,
        )
        # Test return unit
        npt.assert_equal(sigma_crit.unit, self._sigma_crit_unit)

        # Test with lens behind source
        sigma_crit = cosmo.sigma_crit(
            self._z_lens,
            self._z_lens / 2,
            self._cosmo
        )
        npt.assert_equal(sigma_crit, 0 * self._sigma_crit_unit)

        # Test without default arguments
        sigma_crit = cosmo.sigma_crit(
            self._z_lens,
            self._z_source,
            self._cosmo,
            d_source=self._d_source,
        )
        npt.assert_almost_equal(
            sigma_crit.value,
            self._sigma_crit_value,
            decimal=2,
        )

        sigma_crit = cosmo.sigma_crit(
            self._z_lens,
            self._z_source,
            self._cosmo,
            d_lens=self._d_lens,
        )
        npt.assert_almost_equal(
            sigma_crit.value,
            self._sigma_crit_value,
            decimal=2,
        )

        sigma_crit = cosmo.sigma_crit(
            self._z_lens,
            self._z_source,
            self._cosmo,
            d_lens=self._d_lens,
            d_source=self._d_source,
        )
        npt.assert_almost_equal(
            sigma_crit.value,
            self._sigma_crit_value,
            decimal=2,
        )

    def test_sigma_crit_eff(self):
        """Test ``cs_util.cosmo.sigma_crit_eff`` method.

        """
        sigma_crit_eff = cosmo.sigma_crit_eff(
            self._z_lens,
            self._z_source_arr,
            self._nz_source_arr,
            self._cosmo,
        )
        # Test return value
        npt.assert_almost_equal(
            sigma_crit_eff.value,
            self._sigma_crit_value_eff,
            decimal=4,
        )

        # Test return unit
        npt.assert_equal(sigma_crit_eff.unit, self._sigma_crit_unit)

        # Test exception
        self.assertRaises(                                                      
            IndexError,                                                         
            cosmo.sigma_crit_eff,
            self._z_lens,
            self._z_source_arr[:-1],                                                 
            self._nz_source_arr,                                                
            self._cosmo,
        )

    def test_sigma_crit_m1_eff(self):
        """Test ``cs_util.cosmo.sigma_crit_m1_eff`` method.

        """
        sigma_crit_m1_eff = cosmo.sigma_crit_m1_eff(
            self._z_lens,
            self._z_source_arr,
            self._nz_source_arr,
            self._cosmo,
        )
        # Test return value
        npt.assert_almost_equal(
            sigma_crit_m1_eff.value,
            self._sigma_crit_value_eff_m1,
            decimal=4,
        )
        # Test return unit
        npt.assert_equal(
            sigma_crit_m1_eff.unit,
            (1 / self._sigma_crit_unit).unit)

        # Test exception
        self.assertRaises(                                                      
            IndexError,                                                         
            cosmo.sigma_crit_m1_eff,
            self._z_lens,
            self._z_source_arr[:-1],                                                 
            self._nz_source_arr,                                                
            self._cosmo,
        )