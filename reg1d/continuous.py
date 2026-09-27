'''
Continuous (parametric, penalised least-squares) registration.

Implemented directly from

    Ramsay JO, Li X (1998). Curve registration. Journal of the Royal
        Statistical Society B 60: 351-363.
    Ramsay JO, Silverman BW (2005). Functional Data Analysis, 2nd ed.
        Springer.  (Section 7.4, "Continuous registration")

The warp is parameterised through its log-derivative,

    gamma(t) = int_0^t exp( W(s) ) ds  /  int_0^1 exp( W(s) ) ds,
    W(s)     = sum_k c_k phi_k(s),

which guarantees that gamma is smooth and strictly increasing for any
coefficient vector c (the "smooth monotone transformation" of Ramsay
1998). Here phi_k are the first *n_basis* cosine functions
sqrt(2) cos(k pi s), k = 1..n_basis (a constant term would be absorbed by
the normalisation and is therefore omitted). The coefficients are found
by minimising

    F(c) = int ( y(gamma(t)) - target(t) )^2 dt  +  lam * int W'(s)^2 ds

with a quasi-Newton optimiser. The roughness penalty shrinks the warp
towards the identity. Group registration alternates between fitting the
warps and updating the target as the cross-sectional mean of the
registered observations (Procrustes iteration).

Compared with SRSF dynamic programming, this approach produces very
smooth warps with few degrees of freedom and is therefore well suited
to data whose timing differences are gradual; it is less able to align
sharp local features, and the optimisation is local (not guaranteed
global).

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import integrate, optimize
from . import warp as _warp



def _basis(t, n_basis):
    k = np.arange(1, n_basis+1)[:, None]
    return np.sqrt(2) * np.cos(k * np.pi * t[None, :])          # (n_basis, Q)


def _dbasis(t, n_basis):
    k = np.arange(1, n_basis+1)[:, None]
    return -np.sqrt(2) * k * np.pi * np.sin(k * np.pi * t[None, :])


def coef_to_warp(c, Q, n_basis=None):
    '''Warp gamma(t) = int_0^t exp(W) / int_0^1 exp(W),  W = sum c_k phi_k.'''
    c   = np.asarray(c, dtype=float)
    t   = _warp.grid(Q)
    W   = np.clip(c @ _basis(t, c.size), -20, 20)
    g   = integrate.cumulative_trapezoid(np.exp(W), t, initial=0)
    return g / g[-1]


def _objective(c, y, target, t, B, dB, lam):
    W    = np.clip(c @ B, -20, 20)
    g    = integrate.cumulative_trapezoid(np.exp(W), t, initial=0)
    g   /= g[-1]
    yw   = np.interp(g, t, y)
    sse  = integrate.trapezoid((yw - target)**2, t)
    pen  = integrate.trapezoid((c @ dB)**2, t) if lam > 0 else 0.0
    return sse + lam * pen


def align_pair(y_template, y, n_basis=4, lam=1e-2, c0=None, c_max=3.0):
    '''
    Register *y* to *y_template* with a smooth parametric warp.

    *n_basis* : number of sine basis functions for W = log gamma'
    *lam*     : roughness penalty (dimensionless, relative to the variance of the template; 0 = none)
    *c0*      : initial coefficients (None = identity warp, fitted coarse-to-fine)
    *c_max*   : bound on |c_k| (limits the local slope of the warp to about exp(sqrt(2) c_max))

    Returns (y_aligned, gamma, coef).
    '''
    y_template = np.asarray(y_template, dtype=float)
    y          = np.asarray(y, dtype=float)
    Q   = y.size
    t   = _warp.grid(Q)
    B   = _basis(t, n_basis)
    dB  = _dbasis(t, n_basis)
    # scale the penalty relative to the data so that lam is dimensionless
    scale = max(integrate.trapezoid((y_template - y_template.mean())**2, t), 1e-12)
    c     = np.zeros(n_basis) if c0 is None else np.asarray(c0, dtype=float).copy()
    # coarse-to-fine: optimise the first k coefficients for k = 1..n_basis, warm-starting
    # each stage from the previous one, which greatly reduces the risk of a poor local
    # minimum when the warp is far from the identity
    for k in range(1, n_basis+1):
        if c0 is not None and k < n_basis:
            continue
        Bk, dBk = B[:k], dB[:k]
        args    = (y, y_template, t, Bk, dBk, lam*scale)
        # the first two stages are multi-started on a small grid because the
        # objective is not convex (a feature of y can lock on to the wrong feature
        # of the template); later stages refine from the best solution so far
        if c0 is None and k <= 2:
            starts = [np.r_[c[:k-1], v]  for v in np.linspace(-1.5, 1.5, 7)]
        else:
            starts = [c[:k]]
        best = None
        for s0 in starts:
            res = optimize.minimize(_objective, s0, args=args, method='L-BFGS-B',
                bounds=[(-c_max, c_max)]*k)
            if best is None or res.fun < best.fun:
                best = res
        c[:k] = best.x
    gam = coef_to_warp(c, Q)
    return _warp.apply_warp(y, gam), gam, c


def align_group(y, template='mean', n_basis=4, lam=1e-2, max_iter=5, tol=1e-4,
    center=True, verbose=False):
    '''
    Register a set of observations with smooth parametric warps
    (Procrustes iteration on the cross-sectional mean).

    *y*        : (J,Q) array
    *template* : 'mean' (iteratively updated cross-sectional mean), an
                 integer index (fixed), or a (Q,) array (fixed)
    *n_basis*  : number of sine basis functions for W = log gamma'
    *lam*      : roughness penalty (dimensionless; 0 = none)
    *center*   : center the warps so that their Karcher mean is the identity

    Returns a dict with keys 'y', 'warps', 'template', 'coef', 'niter'.
    '''
    y     = np.atleast_2d(np.asarray(y, dtype=float))
    J, Q  = y.shape
    fixed = False
    if isinstance(template, str) and template == 'mean':
        tmpl = y.mean(axis=0)
    elif isinstance(template, (int, np.integer)):
        tmpl, fixed = y[int(template)].copy(), True
    else:
        tmpl, fixed = np.asarray(template, dtype=float), True
    gam   = np.zeros((J, Q))
    coef  = np.zeros((J, n_basis))
    niter = 0
    for it in range(1 if fixed else max_iter):
        niter += 1
        yr = np.empty_like(y)
        for i in range(J):
            yr[i], gam[i], coef[i] = align_pair(tmpl, y[i], n_basis, lam, c0=(None if it == 0 else coef[i]))
        sse = float(((yr - tmpl)**2).sum())
        if verbose:
            print(f'iteration {it+1}: sse = {sse:.6g}')
        if fixed:
            break
        new    = yr.mean(axis=0)
        change = np.linalg.norm(new - tmpl) / max(np.linalg.norm(tmpl), 1e-12)
        tmpl   = new
        if change < tol:
            break
    if center and J > 1:
        gam, _ = _warp.center_warps(gam)
    yr = np.array([_warp.apply_warp(y[i], gam[i])  for i in range(J)])
    if not fixed:
        tmpl = yr.mean(axis=0)
    return dict(y=yr, warps=gam, template=tmpl, coef=coef, niter=niter)
