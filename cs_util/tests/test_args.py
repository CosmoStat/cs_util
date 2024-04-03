"""UNIT TESTS FOR ARGS SUBPACKAGE.

This module contains unit tests for the args subpackage.

"""

import os

import numpy as np
from numpy import testing as npt
import optparse
import pytest

from unittest import TestCase

from cs_util import args


class ArgsTestCase(TestCase):
    """Test case for the ``args`` module."""

    def setUp(self):
        """Set test parameter values."""
        self._params_def = {
            "p_int": 1,
            "p_float": 0.612,
            "p_str": "some_string",
            "p_bool": -1,
            "p_Bool": -1,
            "p_str_def": "second_string",
            "p_none": "",
        }
        self._types = {
            "p_int": "int",
            "p_float": "float",
            "p_bool": "bool",
            "p_Bool": "bool",
            "p_str": "str",
        }
        self._short_options = {
            "p_int": "-i",
            "p_float": "-f",
            "p_bool": "-b",
            "p_Bool": "-B",
            "p_str": "-s",
        }
        self._help_strings = {
            "p_int": "integer option, default={}",
            "p_float": "float option, default={}",
            "p_bool": "bool option, set to True if given",
            "p_Bool": "bool option, set to True if given",
            "p_str": "string option, default={}",
        }

    def tearDown(self):
        """Unset test parameter values."""
        self._params_def = None
        self._types = None

        self._help_strings = None

    def test_parse_options(self):
        """Test `cs_util.args.parse_options` method."""
        self._options = args.parse_options(
            self._params_def,
            self._short_options,
            self._types,
            self._help_strings,
            args=["-i", "2", "-s", "test", "-b"],
        )

        # Test updated options
        npt.assert_equal(self._options["p_int"], 2)
        npt.assert_equal(self._options["p_str"], "test")
        npt.assert_equal(self._options["p_bool"], True)

        # Test unchanged (default) options
        npt.assert_equal(self._options["p_float"], 0.612)
        npt.assert_equal(self._options["p_Bool"], False)
        npt.assert_equal(self._options["p_str_def"], "second_string")

        # Test exceptions TBD
