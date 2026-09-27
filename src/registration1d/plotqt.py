'''
PyQtGraph plotting backend (optional).

Mirrors the Matplotlib helpers in registration1d.plot so that the same
registration results can be shown in a Qt application (PyQt6 / PySide6)
through PyQtGraph, which draws through Qt's scene graph and remains
interactive (pan, zoom, hover) at thousands of curves.

Install the optional dependency with  pip install registration1d[qt]
(pyqtgraph is MIT-licensed; a Qt binding -- PyQt6, PySide6, PyQt5 or
PySide2 -- must also be present; pyqtgraph picks whichever is installed).

Design: every function accepts an existing pyqtgraph PlotItem (the
`plot` argument) so that it can draw into widgets owned by an application;
when `plot` is None a stand-alone PlotWidget is created and returned. The
three-panel `plot_registration` creates (or fills) a GraphicsLayoutWidget.
Nothing here starts a Qt event loop: call `app().exec()` (or let the host
application run its loop) to show the windows interactively, or use
`to_image` to render a widget to a PNG file off-screen. Stand-alone widgets
use a white background by default (`set_theme('light')`); an application
that manages its own theme should call `set_theme('app')` first.

>>> import registration1d as reg1d
>>> from registration1d import plotqt
>>> result = reg1d.register_srsf(y)
>>> win    = plotqt.plot_registration(result, group=group)
>>> win.show(); plotqt.app().exec()

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from . import warp as _warp


_DEFAULT_COLORS = ['k', 'b', 'g', 'r', 'c', 'm', 'y', (255, 165, 0)]
_MPL_TO_RGB     = dict(k=(0, 0, 0), b=(0, 0, 255), g=(0, 128, 0), r=(255, 0, 0), c=(0, 191, 191),
                       m=(191, 0, 191), y=(191, 191, 0), w=(255, 255, 255), orange=(255, 165, 0))


def _pg():
    '''Import pyqtgraph lazily with an informative error.'''
    try:
        import pyqtgraph as pg
    except ImportError as err:
        raise ImportError('registration1d.plotqt requires pyqtgraph and a Qt binding: '
                          'pip install "registration1d[qt]" PyQt6') from err
    return pg


def app():
    '''Return the running QApplication, creating one if necessary.'''
    return _pg().mkQApp()


_THEME = {'name': None}


def set_theme(name='light'):
    '''
    Set pyqtgraph's global background / foreground colours.

    'light' : white background, black foreground (Matplotlib-like; the
              default applied when this module creates its first widget)
    'dark'  : pyqtgraph's native dark theme
    'app'   : leave pyqtgraph's global options untouched (for embedding in
              an application that manages its own theme)
    '''
    pg = _pg()
    if name == 'light':
        pg.setConfigOptions(background='w', foreground='k')
    elif name == 'dark':
        pg.setConfigOptions(background='k', foreground='d')
    elif name != 'app':
        raise ValueError("theme must be 'light', 'dark' or 'app'")
    _THEME['name'] = name


def _ensure_theme():
    if _THEME['name'] is None:
        set_theme('light')


def _color(c):
    '''Accept Matplotlib one-letter names, named colours, RGB tuples or QColor.'''
    if isinstance(c, str):
        if c in _MPL_TO_RGB:
            return _MPL_TO_RGB[c]
        try:
            from matplotlib import colors as mcolors
            return tuple(int(255*v)  for v in mcolors.to_rgb(c))
        except Exception:
            return c
    if isinstance(c, (float, int)):          # Matplotlib grey level '0.5' as number
        v = int(255*float(c)); return (v, v, v)
    return c


def _pen(color, width=1.5, style=None):
    pg = _pg()
    from pyqtgraph.Qt import QtCore
    styles = {None: QtCore.Qt.PenStyle.SolidLine, ':': QtCore.Qt.PenStyle.DotLine,
              '--': QtCore.Qt.PenStyle.DashLine}
    return pg.mkPen(color=_color(color), width=width, style=styles.get(style, QtCore.Qt.PenStyle.SolidLine))


def _target(plot, title=None):
    '''Return (PlotItem, top-level widget) for a given plot or a new PlotWidget.'''
    pg = _pg()
    if plot is None:
        app()
        _ensure_theme()
        widget = pg.PlotWidget(title=title)
        return widget.getPlotItem(), widget
    if hasattr(plot, 'getPlotItem'):           # PlotWidget
        return plot.getPlotItem(), plot
    return plot, plot                          # PlotItem


def _xaxis(Q, x):
    if x is None:
        return np.arange(Q)
    return np.linspace(0, 100, Q) if x == 'percent' else np.linspace(0, 1, Q)



def plot_curves(y, group=None, plot=None, colors=None, x=None, legend=True, labels=None,
    width=1.5, title=None, xlabel=None, ylabel=None):
    '''
    Plot a set of observations, optionally coloured by group (PyQtGraph).

    *y*      : (J,Q) array, or a sequence of 1D arrays of different lengths
    *group*  : (J,) array of group labels
    *plot*   : an existing pyqtgraph PlotItem or PlotWidget to draw into, or None
    *colors* : one colour per group (Matplotlib letters, names, RGB tuples)
    *x*      : abscissa: None (frame numbers), 'percent' (0-100) or 'unit' (0-1)

    Returns the widget that holds the plot (a new PlotWidget if plot was None,
    otherwise the object passed in).
    '''
    item, widget = _target(plot, title)
    J      = len(y)
    group  = np.zeros(J, dtype=int) if group is None else np.asarray(group)
    ug     = np.unique(group)
    colors = _DEFAULT_COLORS if colors is None else colors
    labels = [f'Group {u}'  for u in ug] if labels is None else labels
    leg    = item.addLegend() if (legend and (ug.size > 1 or labels is not None)) else None
    for k, u in enumerate(ug):
        pen   = _pen(colors[k % len(colors)], width)
        first = True
        for i in np.where(group == u)[0]:
            yy = np.asarray(y[i], dtype=float)
            xx = _xaxis(yy.size, x)
            curve = item.plot(xx, yy, pen=pen, name=(labels[k] if (first and leg is not None) else None))
            first = False
    if title is not None:
        item.setTitle(title)
    if xlabel is not None:
        item.setLabel('bottom', xlabel)
    if ylabel is not None:
        item.setLabel('left', ylabel)
    return widget


def plot_warps(w, group=None, plot=None, colors=None, identity=True, **kwargs):
    '''Plot warp functions (values on [0,1] against t on [0,1]) with PyQtGraph.'''
    w      = np.atleast_2d(np.asarray(w, dtype=float))
    kwargs.setdefault('xlabel', 'Time (normalised)')
    kwargs.setdefault('ylabel', 'Warped time')
    widget = plot_curves(w, group=group, plot=plot, colors=colors, x='unit', **kwargs)
    item, _ = _target(widget)
    if identity:
        item.plot([0, 1], [0, 1], pen=_pen('k', 1, ':'))
    return widget


def plot_displacement_fields(w, group=None, plot=None, colors=None, **kwargs):
    '''Plot displacement fields (deviations from linear time on the original time axis).'''
    d      = _warp.displacement_field(np.atleast_2d(np.asarray(w, dtype=float)))
    kwargs.setdefault('xlabel', 'Time (normalised)')
    kwargs.setdefault('ylabel', 'Displacement')
    widget = plot_curves(d, group=group, plot=plot, colors=colors, x='unit', **kwargs)
    item, _ = _target(widget)
    item.addLine(y=0, pen=_pen('k', 1, ':'))
    return widget


def plot_registration(result, group=None, colors=None, win=None, titles=None, size=(1200, 360)):
    '''
    Three-panel summary of a registration result (before / after / warps)
    in a pyqtgraph GraphicsLayoutWidget.

    *win* : an existing GraphicsLayoutWidget to fill (three plots are added
            in one row), or None to create one

    Returns the GraphicsLayoutWidget; its three PlotItems are available as
    win.plots.
    '''
    pg = _pg()
    app()
    titles = ('Before', 'After', 'Warps') if titles is None else titles
    if win is None:
        _ensure_theme()
        win = pg.GraphicsLayoutWidget(title=f'Registration ({result.method})')
        win.resize(*size)
    p0 = win.addPlot(title=titles[0])
    p1 = win.addPlot(title=titles[1])
    p2 = win.addPlot(title=titles[2])
    plot_curves(result.y0, group=group, plot=p0, colors=colors, x='percent', xlabel='Time (%)')
    plot_curves(result.y,  group=group, plot=p1, colors=colors, x='percent', legend=False, xlabel='Time (%)')
    plot_warps(result.warps.asarray(), group=group, plot=p2, colors=colors, legend=False)
    p1.setXLink(p0); p1.setYLink(p0)
    win.plots = (p0, p1, p2)
    return win


def to_image(widget, path=None, size=None):
    '''
    Render a widget to a QImage (and optionally save it to *path*), which
    works off-screen (QT_QPA_PLATFORM=offscreen) and is what the notebooks
    and tests use. Returns the QImage.
    '''
    a = app()
    if size is not None:
        widget.resize(*size)
    widget.show()
    a.processEvents()
    img = widget.grab().toImage()
    if path is not None:
        img.save(str(path))
    return img
