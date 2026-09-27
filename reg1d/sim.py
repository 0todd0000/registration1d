'''
Self-modelling registration (shape-invariant model).

Implemented from

    Kneip A, Gasser T (1988). Convergence and consistency results for
        self-modeling nonlinear regression. Annals of Statistics 16: 82-112.
    Gervini D, Gasser T (2004). Self-modelling warping functions. Journal
        of the Royal Statistical Society B 66: 959-971.

Model:   y_i(t) = a_i * mu( gamma_i(t) ) + b_i + noise

where mu is a common shape function, (a_i, b_i) are amplitude scale and
offset, and gamma_i are smooth warps (here the parametric warps of
reg1d.continuous). The parameters are estimated by alternating

    (1) given mu:            fit gamma_i, a_i, b_i to each observation
    (2) given gamma_i,a_i,b_i: update mu as the cross-sectional mean of the
                              amplitude-normalised, registered observations

which is the self-modelling (Procrustes) iteration of Kneip & Gasser.
Because amplitude is modelled explicitly, observations whose amplitudes
differ strongly (e.g. running speeds) do not distort the warps as they can
when a plain least-squares criterion is used.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import integrate, optimize
from . import warp as _warp
from . import continuous as _continuous



def _fit_amplitude(mu_w, y, t):
    '''Least-squares (a, b) with y ~ a mu_w + b.'''
    X    = np.column_stack([mu_w, np.ones_like(mu_w)])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return a, b


def align_pair(mu, y, n_basis=4, lam=1e-2, c0=None):
    '''
    Fit gamma (parametric), a and b so that y ~ a mu(gamma) + b.
    Returns (y_registered_and_normalised, gamma, (a, b), coef) where the first
    output is (y(gamma) - b) / a, i.e. the observation mapped onto the shape function.
    '''
    mu  = np.asarray(mu, dtype=float)
    y   = np.asarray(y, dtype=float)
    Q   = y.size
    t   = _warp.grid(Q)
    B   = _continuous._basis(t, n_basis)
    dB  = _continuous._dbasis(t, n_basis)
    scale = max(integrate.trapezoid((mu - mu.mean())**2, t), 1e-12)
    def obj(c):
        g    = _continuous.coef_to_warp(c, Q)
        yw   = np.interp(g, t, y)                  # y(gamma)
        a, b = _fit_amplitude(mu, yw, t)           # yw ~ a mu + b
        r    = yw - (a*mu + b)
        pen  = integrate.trapezoid((c @ dB)**2, t) if lam > 0 else 0.0
        return integrate.trapezoid(r**2, t) / max(a*a, 1e-6) + lam*scale*pen
    c = np.zeros(n_basis) if c0 is None else np.asarray(c0, dtype=float).copy()
    for k in range(1, n_basis+1):
        if c0 is not None and k < n_basis:
            continue
        starts = [np.r_[c[:k-1], v]  for v in np.linspace(-1.5, 1.5, 7)] if (c0 is None and k <= 2) else [c[:k]]
        best = None
        for s0 in starts:
            res = optimize.minimize(lambda cc: obj(np.r_[cc, np.zeros(n_basis-k)]), s0, method='L-BFGS-B',
                bounds=[(-3, 3)]*k)
            if best is None or res.fun < best.fun:
                best = res
        c[:k] = best.x
    g    = _continuous.coef_to_warp(c, Q)
    yw   = np.interp(g, t, y)
    a, b = _fit_amplitude(mu, yw, t)
    return (yw - b) / a, g, (a, b), c


def align_group(y, n_basis=4, lam=1e-2, max_iter=5, tol=1e-4, center=True, verbose=False):
    '''
    Self-modelling registration of a (J,Q) array.

    Returns a dict with keys 'y' (registered observations y_i(gamma_i), amplitude
    NOT removed), 'warps', 'template' (shape function mu), 'amplitude' ((J,2)
    array of (a_i, b_i)), 'niter'.
    '''
    y     = np.atleast_2d(np.asarray(y, dtype=float))
    J, Q  = y.shape
    mu    = y.mean(axis=0)
    gam   = np.zeros((J, Q))
    coef  = np.zeros((J, n_basis))
    ab    = np.zeros((J, 2))
    niter = 0
    for it in range(max_iter):
        niter += 1
        yn = np.empty_like(y)
        for i in range(J):
            yn[i], gam[i], ab[i], coef[i] = align_pair(mu, y[i], n_basis, lam, c0=(None if it == 0 else coef[i]))
        new    = yn.mean(axis=0)
        change = np.linalg.norm(new - mu) / max(np.linalg.norm(mu), 1e-12)
        mu     = new
        if verbose:
            print(f'iteration {it+1}: relative change in shape function = {change:.3g}')
        if change < tol:
            break
    if center and J > 1:
        gam, _ = _warp.center_warps(gam)
    yr = np.array([_warp.apply_warp(y[i], gam[i])  for i in range(J)])
    return dict(y=yr, warps=gam, template=mu, amplitude=ab, coef=coef, niter=niter)
