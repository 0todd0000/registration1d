'''
Plotting utilities.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from matplotlib import pyplot as plt
from . import warp as _warp



def _gca(ax):
    return plt.gca() if ax is None else ax


def plot_curves(y, group=None, ax=None, colors=None, x=None, legend=True, labels=None, **kwargs):
    '''
    Plot a set of observations, optionally coloured by group.

    *y*     : (J,Q) array, or a sequence of 1D arrays of different lengths
    *group* : (J,) array of group labels
    *x*     : abscissa: None (frame numbers, 0..Q-1), 'percent' (0-100) or 'unit' (0-1)
    '''
    ax     = _gca(ax)
    J      = len(y)
    group  = np.zeros(J, dtype=int) if group is None else np.asarray(group)
    ug     = np.unique(group)
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'orange'] if colors is None else colors
    h      = []
    for k, u in enumerate(ug):
        c   = colors[k % len(colors)]
        for i in np.where(group == u)[0]:
            yy  = np.asarray(y[i], dtype=float)
            Q   = yy.size
            xx  = np.arange(Q) if x is None else (np.linspace(0, 100, Q) if x == 'percent' else np.linspace(0, 1, Q))
            hh  = ax.plot(xx, yy, color=c, **kwargs)[0]
        h.append(hh)
    if legend and (labels is not None or ug.size > 1):
        labels = [f'Group {u}'  for u in ug] if labels is None else labels
        ax.legend(h, labels)
    return ax


def plot_warps(w, group=None, ax=None, colors=None, identity=True, **kwargs):
    '''Plot warp functions (values on [0,1] against t on [0,1]).'''
    w  = np.atleast_2d(np.asarray(w, dtype=float))
    ax = plot_curves(w, group=group, ax=ax, colors=colors, x='unit', **kwargs)
    if identity:
        ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('Time (normalised)')
    ax.set_ylabel('Warped time')
    return ax


def plot_displacement_fields(w, group=None, ax=None, colors=None, **kwargs):
    '''Plot displacement fields (deviations from linear time on the original time axis).'''
    d  = _warp.displacement_field(np.atleast_2d(np.asarray(w, dtype=float)))
    ax = plot_curves(d, group=group, ax=ax, colors=colors, x='unit', **kwargs)
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.set_xlabel('Time (normalised)')
    ax.set_ylabel('Displacement')
    return ax


def plot_registration(result, group=None, colors=None, figsize=(12, 3.6), titles=None):
    '''
    Three-panel summary of a registration result: original (linearly
    registered) data, registered data, and warps.

    *result* : a RegistrationResult (or any object with .y0, .y and .warps)
    '''
    fig, AX = plt.subplots(1, 3, figsize=figsize)
    titles  = ('Before', 'After', 'Warps') if titles is None else titles
    plot_curves(result.y0, group=group, ax=AX[0], colors=colors, x='percent')
    plot_curves(result.y,  group=group, ax=AX[1], colors=colors, x='percent', legend=False)
    plot_warps(result.warps.asarray(), group=group, ax=AX[2], colors=colors, legend=False)
    for ax, s in zip(AX, titles):
        ax.set_title(s)
    AX[0].set_xlabel('Time (%)')
    AX[1].set_xlabel('Time (%)')
    plt.tight_layout()
    return fig, AX
