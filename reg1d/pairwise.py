'''
Pairwise synchronisation.

Implemented from

    Tang R, Müller H-G (2008). Pairwise curve synchronization for
        functional data. Biometrika 95: 875-889.

Instead of aligning every observation to a template, every pair (i,j) of
observations is aligned, giving pairwise warps gamma_ij with
y_i( gamma_ij(t) ) ~ y_j(t). Under the assumption that the individual
warps average to the identity, the global warp of observation i is
estimated as the (Karcher) mean over j of gamma_ij. Tang & Müller used
penalised least-squares pairwise warps; here the pairwise engine is
selectable ('srsf' dynamic programming, 'dtw' or 'continuous').

The method is template-free, which removes the dependence on template
initialisation, at the cost of J(J-1)/2 pairwise alignments.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from . import warp as _warp
from . import srsf as _srsf
from . import dtw as _dtw
from . import continuous as _continuous



def align_group(y, engine='srsf', center=True, anchor=None, **kwargs):
    '''
    Pairwise synchronisation of a (J,Q) array.

    *engine* : 'srsf' | 'dtw' | 'continuous'  (keyword arguments are passed on)
    *center* : 'karcher' (default, = True) | 'pointwise' | 'anchor' | 'none' (= False).
               Pairwise synchronisation is template-free, so its warps are already
               centred in the sense of Tang & Müller (the mean of the pairwise warps
               is close to the identity); Karcher centring makes this exact.

    Returns a dict with keys 'y', 'warps', 'template' (cross-sectional mean of
    the synchronised observations), 'pairwise' ((J,J,Q) array of pairwise warps).
    '''
    y     = np.atleast_2d(np.asarray(y, dtype=float))
    J, Q  = y.shape
    t     = _warp.grid(Q)
    if engine == 'srsf':
        pair = lambda a, b: _srsf.align_pair(a, b, **kwargs)[1]
    elif engine == 'dtw':
        pair = lambda a, b: _dtw.align_pair(a, b, **kwargs)[1]
    elif engine == 'continuous':
        pair = lambda a, b: _continuous.align_pair(a, b, **kwargs)[1]
    else:
        raise ValueError("engine must be 'srsf', 'dtw' or 'continuous'")
    G = np.tile(t, (J, J, 1))
    for i in range(J):
        for j in range(J):
            if i == j:
                continue
            G[i, j] = pair(y[j], y[i])          # warp taking observation i onto observation j
    gam = np.array([_warp.karcher_mean_warp(G[i])  for i in range(J)])
    center = _warp.resolve_center(center, 'karcher')
    if center != 'none' and J > 1:
        gam, _ = _warp.center_warps(gam, method=center, anchor=anchor)
    yr  = np.array([_warp.apply_warp(y[i], gam[i])  for i in range(J)])
    return dict(y=yr, warps=gam, template=yr.mean(axis=0), pairwise=G, center=center)
