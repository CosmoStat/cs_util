# -*- coding: utf-8 -*-

"""UNIT TESTS FOR PLOTS SUBPACKAGE.

This module contains unit tests for the plots subpackage.

"""

import os

import numpy as np
from matplotlib.figure import Figure
from numpy import testing as npt

from unittest import TestCase

from cs_util import plots


class PlotsTestCase(TestCase):
    """Test case for the ``plots`` module."""

    def setUp(self):
        """Set test parameter values."""
        self._fig_size = [13, 7]


    def tearDown(self):
        """Unset test parameter values."""

        self._fig_size = None

    def test_figure(self):
        """Test ``cs_util.weighted_avg_and_std`` method.

        """
        fig = plots.figure(figsize=(self._fig_size_x, self._fig_size_y))

        # Check for return value
        self.assertIsNotNone(fig, msg='Incorrect return type')

        # Check image size
        size = fig.get_size_inches()
        for idx in (0, 1):
            npt.assert_almost_equal(size[idx], self._fig_size[idx])
