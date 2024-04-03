"""UNIT TESTS FOR ARGS SUBPACKAGE.

This module contains unit tests for the args subpackage.

"""

import os

import numpy as np
from numpy import testing as npt

from unittest import TestCase

from cs_util import args


class ArgsTestCase(TestCase):
    """Test case for the ``args`` module."""

    def setUp(self):
        """Set test parameter values."""
        self._params_def = {
            "p_int": 1,
            "p_float": 0.612,
            "p_bool": True,
            "p_str": "some_string",
            "p_str_def": "second_string",
            "p_none": "",
        }
        self._types = {
            "p_int": "int",
            "p_float": "float",
            "p_bool": "bool",
            "p_str": "str",
        }
        self._short_options = {
            "p_int": "-i",
            "p_float": "-f",
            "p_bool": "-b",
            "p_str": "-s",
        }
        self._help_strings = {
            "p_int": "integer option, default={}",
            "p_float": "float option, default={}",
            "p_bool": "bool option, default={}",
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
            args=["-i", "2", "-s", "test"],
        )
        print(self._options)
