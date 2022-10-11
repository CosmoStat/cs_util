"""PLOTS.

:Name: plots.py

:Description: This file contains methods for plotting.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""


import numpy as np
from matplotlib import pylab as plt


def figure(figsize=(30, 30)):                                                   
    """Figure.                                                                  
                                                                                
    Create figure                                                               
                                                                                
    Parameters                                                                  
    ----------                                                                  
    figsize : tuple, optional                                                   
        figure size, default is (30, 30)                                        
                                                                                
    """                                                                         
    plt.figure(figsize=figsize, facecolor='none')                               
                                                                                
                                                                                
def savefig(fname):                                                             
    """Save Figure.                                                             
                                                                                
    Save figure to file.                                                        
                                                                                
    Parameters                                                                  
    ----------                                                                  
    fname : str                                                                 
        output file name                                                        
                                                                                
    """                                                                         
    plt.savefig(fname, facecolor='w', bbox_inches='tight')


def plot_histograms(
    xs,
    labels,
    title,
    x_label,
    y_label,
    x_range,
    n_bin,
    out_path,
    weights=None,
    colors=None,
    linestyles=None,
    vline_x=None,
    vline_lab=None,
    density=True,
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
    out_path : string
        output file path
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
    """
    if weights is None:
        weights = [np.ones_like(x) for x in xs]
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    if linestyles is None:
        linestyles = ['-'] * len(labels)

    figure(figsize=(15, 10))

    # Histogramsh
    for x, w, label, color, linestyle in zip(
            xs, weights, labels, colors, linestyles
    ):
        plt.hist(x, n_bin, weights=w, range=x_range, histtype='step',
                 color=color, linestyle=linestyle,
                 linewidth=1, density=density, label=label)

    # Horizontal lines
    if vline_x:
        ylim = plt.ylim()
        for x, lab in zip(vline_x, vline_lab):
            plt.vlines(
                x=x,
                ymax=ylim[1],
                ymin=ylim[0],
                linestyles='--',
                colors='k'
            )
            plt.text(x * 1.5, ylim[1] * 0.95, lab)
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    savefig(out_path)



def plot_data_1d(                                                               
    x,                                                                          
    y,                                                                          
    yerr,                                                                       
    title,                                                                      
    xlabel,                                                                     
    ylabel,                                                                     
    out_path,                                                                   
    xlog=False,                                                                 
    ylog=False,                                                                 
    log=False,                                                                  
    labels=None,                                                                
    colors=None,                                                                
    linestyles=None,                                                            
    eb_linestyles=None,                                                         
    linewidths=None,                                                            
    xlim=None,                                                                  
    ylim=None                                                                   
):                                                                              
    """Plot Data 1D.                                                            
                                                                                
    Plot one-dimensional data points with errorbars.                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    x, y, yerr : array of array of float                                        
        data                                                                    
    title, xlabel, ylabel : string                                              
        title and labels                                                        
    out_path : string                                                           
        output file path                                                        
    xlog, ylog : bool, optional, default=False                                  
        logscale on x, y if True                                                
    labels : list, optional, default=None                                       
        plot labels, no labels if None                                          
    color : list, optional, default=None                                        
        line colors, matplotlib default colors if None                          
    linestyle : list, optional, default=None                                    
        linestyle indicators, '-' if None                                       
    linewidths : list                                                           
        line widths, default is `2`                                             
    eb_linestyle : array of string, optional, default=None                      
        errorbar linestyle indicators, '-' if None                              
    xlim : array(float, 2), optional, default=None                              
        x-axis limits, automatic if None                                        
    ylim : array(float, 2), optional, default=None                              
        y-axis limits, automatic if None                                        
    """                                                                         
    if labels is None:                                                          
        labels = [''] * len(x)                                                  
        do_legend = False                                                       
    else:                                                                       
        do_legend = True                                                        
    if colors is None:                                                          
        prop_cycle = plt.rcParams['axes.prop_cycle']                            
        colors = prop_cycle.by_key()['color']                                   
    if linestyles is None:                                                      
        linestyles = ['-'] * len(x)                                             
    if eb_linestyles is None:                                                   
        eb_linestyles = ['-'] * len(x)                                          
    if linewidths is None:                                                      
        linewidths = [2] * len(x)

    figure(figsize=(15, 10))                                                    
                                                                                
    for i in range(len(x)):                                                     
        if np.isnan(yerr[i]).all():                                             
            eb = plt.plot(                                                      
                x[i],                                                           
                y[i],                                                           
                label=labels[i],                                                
                color=colors[i],                                                
                linestyle=linestyles[i],                                        
            )                                                                   
        else:                                                                   
            eb = plt.errorbar(                                                  
                x[i],                                                           
                y[i],                                                           
                yerr=yerr[i],                                                   
                label=labels[i],                                                
                color=colors[i],                                                
                linestyle=linestyles[i],                                        
                marker='o',                                                     
                markerfacecolor='none',                                         
                capsize=4,                                                      
            )                                                                   
            eb[-1][0].set_linestyle(eb_linestyles[i])                           
                                                                                
    plt.hlines(                                                                 
        y=0,                                                                    
        xmin=plt.xlim()[0],                                                     
        xmax=plt.xlim()[1],                                                     
        linestyles='dashed',                                                    
    )                                                                           
                                                                                
    if xlog:                                                                    
        plt.xscale('log')                                                       
        plt.xticks(                                                             
            [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],                
            labels=[                                                            
                '0.1',                                                          
                '0.2',                                                          
                '0.5',                                                          
                '1',                                                            
                '2',                                                            
                '5',                                                            
                '10',                                                           
                '20',                                                           
                '50',                                                           
                '100',                                                          
                '200',                                                          
                '500',                                                          
            ]                                                                   
        )                                                                       
    if ylog:                                                                    
        plt.yscale('log')                                                       
                                                                                
    if xlim:                                                                    
        plt.xlim(xlim)                                                          
    if ylim:                                                                    
        plt.ylim(ylim)                                                          
                                                                                
    plt.hlines(                                                                 
        y=0,                                                                    
        xmin=plt.xlim()[0],
        xmax=plt.xlim()[1],                                                     
        linestyles='dashed'                                                     
    )                                                                           
                                                                                
    plt.title(title)                                                            
    plt.xlabel(xlabel)                                                          
    plt.ylabel(ylabel)                                                          
    if do_legend:                                                               
        plt.legend()                                                            
                                                                                
    savefig(out_path)