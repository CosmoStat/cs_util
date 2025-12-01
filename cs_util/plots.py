"""PLOTS. 

:Name: plots.py

:Description: This file contains methods for plotting.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np


def figure(figsize=(30, 30)):
    """Figure.

    Create figure

    Parameters
    ----------
    figsize : tuple, optional
        figure size, default is (30, 30)

    Returns
    -------
    matplotlib.figure.Figure
        figure object

    """
    fig = plt.figure(figsize=figsize, facecolor="none")

    return fig


def savefig(fname, close_fig=True):
    """Save Figure.

    Save figure to file.

    Parameters
    ----------
    fname : str
        output file name
    close_fig : bool, optional
        closes figure if ``True`` (default); chose ``False``
        to display figure in a jupyter notebook

    """
    plt.savefig(fname, facecolor="w", bbox_inches="tight")
    if close_fig:
        plt.close()


def show():
    backend = matplotlib.get_backend()
    if 'inline' in backend.lower() or 'nbagg' in backend.lower():
        plt.show()  # Works in notebooks
    plt.close()


def get_x_dx(x_arr, shift_x, idx, log=True):

    if shift_x:
        if log:
            this_x = x_arr[idx] * dx(idx, len(x_arr), log=log)
        else:
            raise ValueError("shift_x without log not implemented yet")
    else:
        this_x = x_arr[idx]

    return this_x


def dx(idx, nx=3, fx=1.03, log=True):
    """Dx.

    Return small shift useful to diplace points along the the x-axis
    for a more readable plot.

    Parameters
    ----------
    idx : int
        dataset index
    nx : int, optional
        total number of datasets to plot; default is 3
    fx : float, optional
        shift, default is 1.025
    log : bool, optional
        if True (False), shift is logarithmic (linear); default is ``True``

    """
    if log:
        return fx ** (idx - (nx - 1) / 2)
    else:
        return fx * (idx - (nx - 1) / 2)


def plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path=None,
    weights=None,
    colors=None,
    linestyles=None,
    vline_x=None,
    vline_lab=None,
    density=True,
    close_fig=True,
):
    """Plot Histograms.

    Plot one or more 1D distributions.

    Parameters
    ----------
    xs : array of float
        array of values, each of which to plot the distribution
    labels : array of string
        plot labels
    title : string
        plot title
    x_label, y_label : string
        x-/y-axis label
    n_bin : int
        number of histogram bins
    out_path : string, optional
        output file path, default is ``None``
    weights : array of float, optional, default=None
        weights
    colors : array of string, optional, default=None
        plot colors
    linestyles : array of string, optional, default=None
        line styles
    vline_x : array of float, optional, default=None
        x-values of vertical lines if not None
    vline_lab : array of string, optional, default=None
        labels of vertical lines if not None
    density : bool, optional, default=True
        (normalised) density histogram if True
    close_fig : bool, optional
        closes figure if True (default)

    Returns
    -------
    list
        values, bins for each histogram call

    """
    if weights is None:
        weights = [np.ones_like(x) for x in xs]
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    if linestyles is None:
        linestyles = ["-"] * len(labels)

    figure(figsize=(15, 10))

    # Return lists
    n_arr = []
    bins_arr = []

    # Histograms
    for x, w, label, color, linestyle in zip(
        xs, weights, labels, colors, linestyles
    ):
        n, bins, _ = plt.hist(
            x,
            n_bin,
            weights=w,
            range=x_range,
            histtype="step",
            color=color,
            linestyle=linestyle,
            linewidth=1,
            density=density,
            label=label,
        )
        n_arr.append(n)
        bins_arr.append(bins)

    # Horizontal lines
    if vline_x:
        ylim = plt.ylim()
        for x, lab in zip(vline_x, vline_lab):
            plt.vlines(
                x=x, ymax=ylim[1], ymin=ylim[0], linestyles="--", colors="k"
            )
            plt.text(x * 1.5, ylim[1] * 0.95, lab)
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    if out_path:
        savefig(out_path, close_fig=close_fig)

    return n_arr, bins_arr


def plot_data_1d(
    x,
    y,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path=None,
    ax=None,
    xlog=False,
    ylog=False,
    log=False,
    labels=None,
    colors=None,
    linestyles=None,
    eb_linestyles=None,
    linewidths=None,
    markers=None,
    xlim=None,
    ylim=None,
    shift_x=False,
    neg_dash=False,
    close_fig=True,
    second_x_axis=None,
    second_x_label=None,
    second_x_every=1,
):
    """Plot Data 1D.

    Plot one-dimensional data points with errorbars.

    Parameters
    ----------
    x, y, yerr : array of array of float
        data
    title, xlabel, ylabel : string
        title and labels
    out_path : string, optional
        output file path, default is ``None``
    ax : matplotlib.axes, optional
        use this axis object if given; it not (default) create a new figure
    xlog, ylog : bool, optional, default is ``False``
        logscale on x, y if True
    labels : list, optional, default is ``None``
        plot labels, no labels if None
    color : list, optional, default is ``None``
        line colors, matplotlib default colors if ``None``
    linestyle : list, optional, default is ``None``
        linestyle indicators, '-' if ``None``
    linewidths : list
        line widths, default is `2`
    markers : list
        marker types, default is `o`
    eb_linestyles : array of string, optional, default is ``None``
        errorbar linestyle indicators, '-' if ``None``
    xlim : array(float, 2), optional, default=None
        x-axis limits, automatic if ``None``
    ylim : array(float, 2), optional, default is ``None``
        y-axis limits, automatic if ``None``
    shift_x : bool, optional
        shift datasets by small amount along x if ``True``; default is ``False``
    neg_dash: bool, optional
        if ylog is True, add negative points with dashed lines
    close_fig : bool, optional
        closes figure if True (default)
    second_x_axis : array of float, optional, default is ``None``
        values for second x-axis on top, if not None
    second_x_label : string, optional, default is ``None``
        label for second x-axis on top
    second_x_every: int, optional
        plot only one in every `every` point on second x-axis; default is 1

    """
    # Add negative points with dashed lines
    if neg_dash:
        if not ylog:
            raise ValueError("neg_dash only valid if ylog is True")

        n = len(x)

        # Duplicate the following lists
        x = x * 2
        yerr = yerr * 2
        colors = colors * 2
        labels = labels + [""] * n

        # Add negative y-values
        y = y + [-arr for arr in y]

        # Add dashed lines
        linestyles = linestyles + ["--"] * n

    if labels is None:
        labels = [""] * len(x)
        do_legend = False
    else:
        do_legend = True
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    if linestyles is None:
        linestyles = ["-"] * len(x)
    if eb_linestyles is None:
        eb_linestyles = ["-"] * len(x)
    if linewidths is None:
        linewidths = [2] * len(x)
    if markers is None:
        markers = ["o"] * len(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    for idx in range(len(x)):
        this_x = get_x_dx(x, shift_x, idx, log=xlog)

        if np.isnan(yerr[idx]).all():
            eb = ax.plot(
                this_x,
                y[idx],
                label=labels[idx],
                color=colors[idx],
                linestyle=linestyles[idx],
            )
        else:
            eb = ax.errorbar(
                this_x,
                y[idx],
                yerr=yerr[idx],
                label=labels[idx],
                color=colors[idx],
                linestyle=linestyles[idx],
                marker=markers[idx],
                markerfacecolor="none",
                capsize=4,
            )
            eb[-1][0].set_linestyle(eb_linestyles[idx])

    ax.axhline(color="k", linestyle="dashed", linewidth=linewidths[0] / 2)

    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1,2,5), numticks=15))
        ax.xaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=False))

    if ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if do_legend:
        ax.legend()

    # Add second x-axis on top if requested
    if second_x_axis is not None:
        ax2 = ax.twiny()
        
        # Set second x-axis with provided values
        ax2.set_xlim(ax.get_xlim())
        if xlog:
            ax2.set_xscale('log')
        
        if second_x_label is not None:
            ax2.set_xlabel(second_x_label)
        
        # Create tick positions that correspond to the main x-axis values
        # Map from main x values to second x-axis values
        main_x_values = x[0] if len(x) > 0 else []
        if len(main_x_values) > 0 and len(second_x_axis) == len(main_x_values):
            # Find values within the plot range
            xlim_current = ax.get_xlim()
            tick_positions = []
            tick_labels = []
            
            for i, main_x_val in enumerate(main_x_values):
                if (
                    xlim_current[0] <= main_x_val <= xlim_current[1]
                ):
                    tick_positions.append(main_x_val)
                    if i % second_x_every == 0:
                        my_label = f'{second_x_axis[i]:.2g}'
                    else:
                        my_label = ""
                    tick_labels.append(my_label)
            
            if tick_positions:
                ax2.set_xticks(tick_positions)
                ax2.set_xticklabels(tick_labels)
                ax2.tick_params(axis='x', labelrotation=45)

    if out_path:
        savefig(out_path, close_fig=close_fig)


def log_ticks(x):

    x = np.asarray(x)
    xmin, xmax = np.nanmin(x), np.nanmax(x)

    # figure out the exponent range
    exp_min = int(np.floor(np.log10(xmin)))
    exp_max = int(np.ceil(np.log10(xmax)))

    ticks = []
    for exp in range(exp_min, exp_max + 1):
        for base in [1, 2, 5]:
            val = base * 10**exp
            if xmin <= val <= xmax:
                ticks.append(val)

    return np.array(ticks)
