"""UNIT TESTS FOR PLOTS SUBPACKAGE.

This module contains unit tests for the plots subpackage.

"""

import os

import numpy as np
from matplotlib.figure import Figure
from numpy import testing as npt

from inspect import signature
from unittest import TestCase, skipIf

from cs_util import plots


class PlotsTestCase(TestCase):
    """Test case for the ``plots`` module."""

    def setUp(self):
        """Set test parameter values."""
        self._fig_size = [13, 7]

        self._x = [1, 1.5, 2, 2, 3, 5]
        self._n_bin = 4
        self._x_range = [1, 5]
        self._img_path = "test.png"
        self._n_arr = np.array([2.0, 2.0, 1.0, 1.0])
        self._bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def tearDown(self):
        """Unset test parameter values."""
        self._fig_size = None
        self._x = None
        self._n_bin = None
        self._x_range = None

        if os.path.exists(self._img_path):
            os.remove(self._img_path)
        self._img_path = None

        self._n_arr = None
        self._bins = None

    def test_figure(self):
        """Test ``cs_util.weighted_avg_and_std`` method."""
        fig = plots.figure(figsize=(self._fig_size[0], self._fig_size[1]))

        # Check for return value
        self.assertIsNotNone(fig, msg="Incorrect return type")

        # Check image size
        size = fig.get_size_inches()
        for idx in (0, 1):
            npt.assert_almost_equal(size[idx], self._fig_size[idx])

    def test_plot_histograms(self):
        """Test ``cs_util.plot_histograms`` method."""
        vline_x_arr = [None, [1.2]]
        vline_lab_arr = [None, ["vlab"]]
        for vline_x, vline_lab in zip(vline_x_arr, vline_lab_arr):
            n_arr, bins = plots.plot_histograms(
                [self._x],
                ["hist 1"],
                "title",
                "$x$",
                "freq",
                self._x_range,
                self._n_bin,
                self._img_path,
                density=False,
                vline_x=vline_x,
                vline_lab=vline_lab,
            )

            # Check return histogram data
            npt.assert_almost_equal(n_arr[0], self._n_arr)
            npt.assert_almost_equal(bins[0], self._bins)

            # Check output plot file
            self.assertTrue(os.path.exists(self._img_path))


class FootprintPlotterTestCase(TestCase):
    """Test case for the ``FootprintPlotter`` class."""

    def setUp(self):
        """Set test parameter values."""
        self._img_path = "test_footprint.png"

    def tearDown(self):
        """Unset test parameter values."""
        if os.path.exists(self._img_path):
            os.remove(self._img_path)

    def test_plot_area_exists(self):
        """Test ``FootprintPlotter.plot_area`` exists with expected signature."""
        self.assertTrue(
            hasattr(plots.FootprintPlotter, "plot_area"),
            msg="FootprintPlotter.plot_area is missing",
        )

        params = signature(plots.FootprintPlotter.plot_area).parameters
        for name in (
            "hsp_map",
            "ra_0",
            "extend",
            "vmin",
            "vmax",
            "projection",
            "outpath",
            "title",
            "colorbar",
            "colorbar_label",
        ):
            self.assertIn(name, params, msg=f"plot_area lost parameter '{name}'")

    def test_plot_region_forwards_to_plot_area(self):
        """Test ``FootprintPlotter.plot_region`` forwards to ``plot_area``."""
        plotter = plots.FootprintPlotter()
        region = plots.FootprintPlotter._regions["NGC"]

        calls = []

        def record(hsp_map, *args, **kwargs):
            calls.append((hsp_map, args, kwargs))
            return "projection", "ax"

        plotter.plot_area = record
        result = plotter.plot_region(None, region, title="title")

        self.assertEqual(result, ("projection", "ax"))
        self.assertEqual(len(calls), 1)

        _, args, kwargs = calls[0]
        self.assertEqual(
            args,
            (region["ra_0"], region["extend"], region["vmin"], region["vmax"]),
        )
        self.assertEqual(kwargs["title"], "title")

    def test_create_hsp_map(self):
        """Test ``FootprintPlotter.create_hsp_map`` method."""
        plotter = plots.FootprintPlotter(nside_coverage=32, nside_map=64)

        ra = np.array([180.0, 180.0, 190.0])
        dec = np.array([40.0, 40.0, 45.0])
        hsp_map = plotter.create_hsp_map(ra, dec)

        # Two objects fall in the same pixel, one in another.  The map
        # sentinel is NaN, so select the pixels that actually hold counts.
        values = hsp_map[hsp_map.valid_pixels]
        counts = values[np.isfinite(values)]
        npt.assert_almost_equal(sorted(counts), [1.0, 2.0])

    @skipIf(plots.skyproj is None, "skyproj is not installed")
    def test_plot_area(self):
        """Test ``FootprintPlotter.plot_area`` method."""
        plotter = plots.FootprintPlotter(nside_coverage=32, nside_map=64)

        ra = np.array([180.0, 185.0, 190.0])
        dec = np.array([40.0, 42.0, 45.0])
        hsp_map = plotter.create_hsp_map(ra, dec)

        projection, ax = plotter.plot_area(
            hsp_map,
            ra_0=180,
            extend=[120, 270, 20, 70],
            vmin=0,
            vmax=60,
            outpath=self._img_path,
            title="title",
        )

        self.assertIsNotNone(projection, msg="Incorrect return type")
        self.assertTrue(os.path.exists(self._img_path))
