# -*- coding: utf-8 -*-

"""UNIT TESTS FOR CALC SUBPACKAGE.

This module contains unit tests for the calc subpackage.

"""

import os

import numpy as np
from numpy import testing as npt

from unittest import TestCase

from cs_util import calc


class CatTestCase(TestCase):
    """Test case for the ``cat`` module."""

    def setUp(self):
        """Set test parameter values."""
        self._x = np.array([1.0, 2.0, 1.5, 0.7])
        self._w = np.array([0.8, 1.2, 2.0, 1.5])


    def tearDown(self):
        """Unset test parameter values."""
        self._x = None
        self._w = None

    def test_weighted_avg_and_std(self):
        """Test ``cs_util.weighted_avg_and_std`` method.

        """
        mean, std = calc.weighted_avg_and_std(self._x, self._w)
