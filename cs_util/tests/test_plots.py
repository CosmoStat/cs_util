# -*- coding: utf-8 -*-

"""UNIT TESTS FOR PLOTS SUBPACKAGE.

This module contains unit tests for the plotsw subpackage.

"""

import os

import numpy as np
from numpy import testing as npt

from unittest import TestCase

from cs_util import calc, plots


class CalcTestCase(TestCase):
    """Test case for the ``calc`` module."""

    def setUp(self):
        """Set test parameter values."""
        self._x = np.array([1.0, 2.0, 1.5, 0.7])
        self._w = np.array([0.8, 1.2, 2.0, 1.5])
        self._mean_w = 725 / 550
        self._std_w = 0.4820754030386492
        self._std_w_corr = 0.5566527274281229


    def tearDown(self):
        """Unset test parameter values."""
        self._x = None
        self._w = None
        self._mean_w = None
        self._std_w = None
        self._std_w_corr = None

    def test_weighted_avg_and_std(self):
        """Test ``plots.weighted_avg_and_std`` method.

        """
        mean_w, std_w = calc.weighted_avg_and_std(self._x, self._w)
        npt.assert_almost_equal(mean_w, self._mean_w)
        npt.assert_almost_equal(std_w, self._std_w)

        mean_w_corr, std_w_corr = calc.weighted_avg_and_std(
            self._x,
            self._w,
            corrected=True,
        )
        npt.assert_almost_equal(mean_w_corr, self._mean_w)
        npt.assert_almost_equal(std_w_corr, self._std_w_corr)
