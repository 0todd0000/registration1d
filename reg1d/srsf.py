'''
Elastic (square-root slope function, SRSF) registration.

Implemented directly from the mathematics of the square-root velocity /
square-root slope framework:

    Srivastava A, Wu W, Kurtek S, Klassen E, Marron JS (2011).
        Registration of functional data using Fisher-Rao metric.
        arXiv:1103.3817
    Tucker JD, Wu W, Srivastava A (2013). Generative models for functional
        data using phase and amplitude separation. Computational Statistics
        and Data Analysis 61: 50-66.
    Srivastava A, Klassen EP (2016). Functional and Shape Data Analysis.
        Springer.  (Chapter 4 and Chapter 8)

The SRSF of an absolutely continuous function f is

    q(t) = sign( f'(t) ) * sqrt( | f'(t) | )

Under time warping f -> f o gamma the SRSF transforms as

    q -> (q o gamma) * sqrt( gamma' )

which is an isometry of L2, so the elastic distance between two
functions can be found by minimising the ordinary L2 distance between q1
and the warped q2 over all warps gamma. The minimisation is carried out
by dynamic programming over a discrete set of admissible slopes; group
registration iterates between (i) aligning every observation to a
template and (ii) updating the template to the mean of the aligned SRSFs
(a Karcher mean under the Fisher-Rao metric).

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import integrate
from math import gcd
from . import warp as _warp



# ---------------------------------------------------------------------
# SRSF transform
# ---------------------------------------------------------------------

def srsf(y, smooth=0):
    """
    Square-root slope function of one or more observations.

    *y*      : (Q,) or (J,Q) array sampled on the uniform grid over [0,1];
               for multivariate observations a (Q,D) or (J,Q,D) array (time
               on the second-to-last axis, components on the last), in which
               case the vector SRSF  q = f' / sqrt(||f'||)  is returned
    *smooth* : 0 (none); an integer (number of passes of a 3-point moving
               average applied to y before differentiation); or 'spline'
               (derivative of a smoothing spline with the smoothing
               parameter chosen by generalised cross-validation, as in
               fdasrsf's f_to_srsf with smooth=True)
    """
    y  = np.asarray(y, dtype=float)
    if _is_multivariate(y):
        return _srsf_mv(y, smooth)
    t  = _warp.grid(y.shape[-1])
    if isinstance(smooth, str) and smooth == 'spline':
        dy = _spline_derivative(y, t)
    else:
        for _ in range(int(smooth)):
            y  = _boxsmooth(y)
        dy = np.gradient(y, t, axis=-1)
    return np.sign(dy) * np.sqrt(np.abs(dy))


def _is_multivariate(y):
    """(Q,D) with D <= 3 is ambiguous with (J,Q); multivariate arrays are therefore
    recognised only as 3D arrays (J,Q,D), or through the explicit srsf_mv function."""
    return y.ndim == 3


def _spline_derivative(y, t):
    from scipy.interpolate import make_smoothing_spline
    if y.ndim == 1:
        return make_smoothing_spline(t, y).derivative()(t)
    return np.array([_spline_derivative(yy, t)  for yy in y])


def _srsf_mv(y, smooth=0):
    """Vector SRSF of (Q,D) or (J,Q,D) arrays:  q = f' / sqrt(||f'||)."""
    if y.ndim == 3:
        return np.array([_srsf_mv(yy, smooth)  for yy in y])
    t  = _warp.grid(y.shape[0])
    if isinstance(smooth, str) and smooth == 'spline':
        dy = _spline_derivative(y.T, t).T
    else:
        for _ in range(int(smooth)):
            y  = _boxsmooth(y.T).T
        dy = np.gradient(y, t, axis=0)
    n  = np.linalg.norm(dy, axis=1)
    return dy / np.sqrt(np.maximum(n, 1e-12))[:, None]


def srsf_mv(y, smooth=0):
    """Vector SRSF of a single (Q,D) multivariate observation (or (J,Q,D))."""
    return _srsf_mv(np.asarray(y, dtype=float), smooth)


def srsf_inverse(q, y0=0.0):
    '''
    Reconstruct f from its SRSF q and the initial value f(0)=y0:

        f(t) = y0 + int_0^t q(s) |q(s)| ds
    '''
    q  = np.asarray(q, dtype=float)
    t  = _warp.grid(q.shape[-1])
    f  = integrate.cumulative_trapezoid(q*np.abs(q), t, axis=-1, initial=0)
    return f + np.asarray(y0, dtype=float)[..., None] if np.ndim(y0) else f + y0


def warp_srsf(q, w):
    '''
    Group action of a warp on an SRSF:  (q o w) * sqrt( w' )

    *q* : (Q,) univariate SRSF, or (Q,D) vector SRSF (time on the first axis)
    '''
    q  = np.asarray(q, dtype=float)
    if q.ndim == 2:      # (Q,D): warp each component
        return _warp.apply_warp(q.T, w).T * np.sqrt(np.clip(_warp.derivative(w), 0, None))[:, None]
    qw = _warp.apply_warp(q, w)
    return qw * np.sqrt(np.clip(_warp.derivative(w), 0, None))


def _apply_warp_any(y, w):
    '''Apply a single warp to a (Q,) or (Q,D) observation.'''
    y = np.asarray(y, dtype=float)
    return _warp.apply_warp(y.T, w).T if y.ndim == 2 else _warp.apply_warp(y, w)


def _boxsmooth(y):
    yp = np.pad(y, [(0,0)]*(y.ndim-1) + [(1,1)], mode='edge')
    return (yp[..., :-2] + yp[..., 1:-1] + yp[..., 2:]) / 3.0



# ---------------------------------------------------------------------
# pairwise alignment by dynamic programming
# ---------------------------------------------------------------------

def _slope_set(max_step):
    '''All coprime (di,dj) steps with 1 <= di,dj <= max_step (strictly increasing paths only).'''
    steps = [(a, b)  for a in range(1, max_step+1)  for b in range(1, max_step+1)  if gcd(a, b) == 1]
    return sorted(steps)


def _interp_q(X, idx, q):
    """Linear interpolation of a (Q,) or (Q,D) SRSF at fractional indices X (any shape)."""
    if q.ndim == 1:
        return np.interp(X, idx, q)
    return np.stack([np.interp(X, idx, q[:, k])  for k in range(q.shape[1])], axis=-1)


def _segment_costs(q1, q2, di, dj, t, nsub=4):
    """
    Cost E[i,j] of the linear path segment from node (i,j) to (i+di, j+dj),
    for every start node, vectorised over the whole grid:

        E[i,j] = int_{t_i}^{t_{i+di}}  || q1(t) - sqrt(s) q2( t_j + s (t - t_i) ) ||^2 dt,   s = dj/di

    q1 and q2 ((Q,) or (Q,D)) are sampled by linear interpolation at nsub
    points per grid step and the integral is approximated by the
    trapezoidal rule.
    """
    Q     = t.size
    s     = dj / di
    n1    = Q - di               # number of admissible start rows
    n2    = Q - dj               # number of admissible start columns
    if n1 <= 0 or n2 <= 0:
        return None
    m     = di * nsub + 1
    u     = np.linspace(0, di, m)                   # offset along t (grid units)
    dt    = (t[1] - t[0]) * (u[1] - u[0])
    I     = np.arange(n1)[:, None] + u[None, :]     # (n1, m) fractional row indices
    Jx    = np.arange(n2)[:, None] + s*u[None, :]   # (n2, m) fractional column indices
    idx   = np.arange(Q)
    Q1    = _interp_q(I, idx, q1)                   # (n1, m[, D])
    Q2    = _interp_q(Jx, idx, q2) * np.sqrt(s)     # (n2, m[, D])
    D     = Q1[:, None, ...] - Q2[None, :, ...]     # (n1, n2, m[, D])
    D2    = D**2 if q1.ndim == 1 else (D**2).sum(axis=-1)
    E     = dt * (D2[..., 1:] + D2[..., :-1]).sum(axis=-1) / 2.0
    return E


def align_srsf_pair(q1, q2, max_step=6, nsub=4, lam=0.0, band=None):
    """
    Optimal warp aligning SRSF q2 to SRSF q1 by dynamic programming:

        gamma* = argmin_gamma  || q1 - (q2 o gamma) sqrt(gamma') ||^2  +  lam * R(gamma)

    where R penalises departure from the identity (int (sqrt(gamma') - 1)^2 dt,
    as used in the fdasrsf "lam" argument).

    *q1*, *q2*  : (Q,) univariate SRSFs, or (Q,D) vector SRSFs
    *max_step*  : the admissible path slopes are all coprime (di,dj) pairs with
                  1 <= di, dj <= max_step (6 -> local slopes 1/6 ... 6); the path
                  is strictly increasing so the warp is a valid diffeomorphism
    *nsub*      : sub-samples per grid step in the segment-cost integrals
    *band*      : Sakoe-Chiba band: |gamma(t) - t| <= band (normalised time);
                  None = unconstrained

    Returns the warp gamma sampled on the uniform grid (Q,).
    """
    q1    = np.asarray(q1, dtype=float)
    q2    = np.asarray(q2, dtype=float)
    Q     = q1.shape[0]
    t     = _warp.grid(Q)
    steps = _slope_set(max_step)
    E     = {}
    for (di, dj) in steps:
        e   = _segment_costs(q1, q2, di, dj, t, nsub=nsub)
        if e is not None and lam > 0:
            s   = dj / di
            e   = e + lam * (np.sqrt(s) - 1)**2 * di * (t[1]-t[0])
        E[(di, dj)] = e
    INF   = np.inf
    D     = np.full((Q, Q), INF)
    P     = np.full((Q, Q, 2), -1, dtype=int)     # predecessor node
    D[0, 0] = 0.0
    if band is not None:
        jj    = np.arange(Q)
        bw    = int(np.ceil(band * (Q - 1)))
    for i in range(1, Q):
        for (di, dj) in steps:
            e   = E[(di, dj)]
            i0  = i - di
            if i0 < 0 or e is None:
                continue
            jmax   = Q - dj                      # start columns 0..jmax-1  ->  end columns dj..Q-1
            cand   = D[i0, :jmax] + e[i0, :]
            cur    = D[i, dj:]
            better = cand < cur
            if np.any(better):
                D[i, dj:][better]    = cand[better]
                P[i, dj:][better, 0] = i0
                P[i, dj:][better, 1] = np.arange(jmax)[better]
        if band is not None:                     # forbid nodes outside the band
            out = np.abs(jj - i) > bw
            D[i, out] = INF
    # backtrack from (Q-1, Q-1)
    i, j  = Q-1, Q-1
    if not np.isfinite(D[i, j]):
        raise RuntimeError('dynamic programming failed to reach the end node; increase max_step or band')
    path  = [(i, j)]
    while (i, j) != (0, 0):
        i, j = P[i, j]
        path.append((i, j))
    path  = np.array(path[::-1], dtype=float)
    gam   = np.interp(np.arange(Q), path[:, 0], path[:, 1]) / (Q - 1)
    return _warp.normalize_warp(gam)


def refine_warp(q1, q2, gam, n_basis=6, lam=0.0):
    """
    Gradient-based refinement of a dynamic-programming warp.

    The DP warp is piecewise linear with slopes drawn from a finite set. It
    is refined by composing it with a smooth correction  gamma = gam o eta,
    eta = int exp(W) / int_0^1 exp(W),  W in a small cosine basis, and
    minimising the SRSF distance to q1 with L-BFGS-B (a local, smooth
    optimisation started at the global DP solution; comparable in purpose
    to fdasrsf's omethod="RBFGS", though not the same algorithm).
    """
    from scipy import optimize
    Q  = q1.shape[0]
    t  = _warp.grid(Q)
    k  = np.arange(1, n_basis+1)[:, None]
    B  = np.sqrt(2) * np.cos(k * np.pi * t[None, :])
    def eta_of(c):
        g = integrate.cumulative_trapezoid(np.exp(np.clip(c @ B, -10, 10)), t, initial=0)
        return g / g[-1]
    def obj(c):
        g  = _warp.compose(gam, eta_of(c))
        d  = q1 - warp_srsf(q2, g)
        d2 = d**2 if d.ndim == 1 else (d**2).sum(axis=1)
        pen = lam * integrate.trapezoid((_warp.warp_to_psi(g) - 1)**2, t) if lam > 0 else 0.0
        return integrate.trapezoid(d2, t) + pen
    res = optimize.minimize(obj, np.zeros(n_basis), method='L-BFGS-B', bounds=[(-1.5, 1.5)]*n_basis)
    g   = _warp.compose(gam, eta_of(res.x))
    return g if obj(res.x) <= obj(np.zeros(n_basis)) else gam


def align_pair(y_template, y, max_step=6, nsub=4, lam=0.0, band=None, smooth=0, refine=False):
    '''
    Elastically align observation *y* to *y_template* (both (Q,) arrays, or
    both (Q,D) multivariate arrays).

    Returns (y_aligned, gamma) where y_aligned = y( gamma(t) ).
    '''
    y_template = np.asarray(y_template, dtype=float)
    y          = np.asarray(y, dtype=float)
    f   = srsf_mv if y.ndim == 2 else srsf
    q1  = f(y_template, smooth=smooth)
    q2  = f(y, smooth=smooth)
    gam = align_srsf_pair(q1, q2, max_step=max_step, nsub=nsub, lam=lam, band=band)
    if refine:
        gam = refine_warp(q1, q2, gam, lam=lam)
    return _apply_warp_any(y, gam), gam


def _align_one(args):
    '''Worker for parallel group alignment (module-level so that it can be pickled).'''
    mq, qi, max_step, nsub, lam, band, refine = args
    g = align_srsf_pair(mq, qi, max_step=max_step, nsub=nsub, lam=lam, band=band)
    if refine:
        g = refine_warp(mq, qi, g, lam=lam)
    return g



# ---------------------------------------------------------------------
# group alignment (Karcher mean template)
# ---------------------------------------------------------------------

def align_group(y, template='karcher', method='mean', max_iter=20, tol=1e-3, center=True,
    max_step=6, nsub=4, lam=0.0, band=None, smooth=0, refine=False, parallel=False, verbose=False):
    """
    Elastically register a set of observations to a common template.

    *y*         : (J,Q) array, or (J,Q,D) array of D-variate observations
    *template*  : 'karcher' (iteratively updated mean / median of the aligned
                  SRSFs, initialised at the observation closest to the mean SRSF),
                  'first' (align all to the first observation, no iteration),
                  an integer (align all to that observation, no iteration),
                  or a (Q,) / (Q,D) array (a fixed user-supplied template)
    *method*    : 'mean' (Karcher mean) or 'median' (Karcher median, i.e. the
                  minimiser of the sum of (unsquared) SRSF distances, found by
                  Weiszfeld iteration as in fdasrsf's srsf_align(method="median"))
    *max_iter*  : maximum number of template updates (karcher only)
    *tol*       : stop when the relative change in the template is below tol
    *center*    : if True, the warps are centered so that their Karcher mean
                  is the identity, and the registered observations and the
                  template are recomputed accordingly (matches the default
                  behaviour of fdasrsf's srsf_align with center=True)
    *max_step*, *nsub*, *lam*, *band* : see align_srsf_pair
    *smooth*    : see srsf
    *refine*    : see refine_warp
    *parallel*  : False, True (all cores) or an integer number of worker processes

    Returns a dict with keys:
        'y'        : (J,Q[,D]) registered observations
        'warps'    : (J,Q) warps such that y_registered[i] = y[i]( warps[i](t) )
        'template' : (Q[,D]) the final (function-space) template
        'q'        : (J,Q[,D]) SRSFs of the registered observations
        'niter'    : number of iterations
        'cost'     : sum of squared SRSF distances to the template at each iteration
    """
    y     = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    mv    = y.ndim == 3
    J, Q  = y.shape[:2]
    t     = _warp.grid(Q)
    q     = srsf(y, smooth=smooth)
    sumax = tuple(range(1, q.ndim))          # axes to sum over for per-observation norms
    f0    = lambda arr: arr[..., 0, :] if mv else arr[..., 0]    # value(s) at t=0
    # --- initial template
    fixed = True
    if isinstance(template, str) and template == 'karcher':
        mq    = q.mean(axis=0)
        d     = ((q - mq)**2).sum(axis=sumax)
        ind   = int(np.argmin(d))
        mq    = q[ind].copy()
        mf0   = f0(y[ind])
        fixed = False
    elif isinstance(template, str) and template == 'first':
        mq, mf0 = q[0].copy(), f0(y[0])
    elif isinstance(template, (int, np.integer)):
        mq, mf0 = q[int(template)].copy(), f0(y[int(template)])
    else:
        tmpl    = np.asarray(template, dtype=float)
        mq, mf0 = (srsf_mv(tmpl, smooth) if mv else srsf(tmpl, smooth=smooth)), f0(tmpl)
    # --- alignment of all observations to the current template
    pool = None
    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=None if parallel is True else int(parallel))
    def align_all(mq):
        args = [(mq, q[i], max_step, nsub, lam, band, refine)  for i in range(J)]
        if pool is None:
            return np.array([_align_one(a)  for a in args])
        return np.array(list(pool.map(_align_one, args)))
    # --- iterate
    gam   = np.tile(t, (J, 1))
    costs = []
    niter = 0
    try:
        for it in range(1 if fixed else max_iter):
            niter += 1
            gam   = align_all(mq)
            qn    = np.array([warp_srsf(q[i], gam[i])  for i in range(J)])
            cost  = float(((qn - mq)**2).sum() * (t[1]-t[0]))
            costs.append(cost)
            if verbose:
                print(f'iteration {it+1}: cost = {cost:.6g}')
            if fixed:
                break
            if method == 'median':
                dist   = np.sqrt(((qn - mq)**2).sum(axis=sumax) * (t[1]-t[0]))
                wts    = 1.0 / np.maximum(dist, 1e-8)
                mq_new = np.tensordot(wts, qn, axes=(0, 0)) / wts.sum()
            else:
                mq_new = qn.mean(axis=0)
            change = np.linalg.norm(mq_new - mq) / max(np.linalg.norm(mq), 1e-12)
            mq     = mq_new
            if change < tol:
                break
    finally:
        if pool is not None:
            pool.shutdown()
    # --- center the warps
    if center and J > 1:
        gam, gmean = _warp.center_warps(gam)
        mq         = warp_srsf(mq, _warp.invert(gmean))
    yr    = np.array([_apply_warp_any(y[i], gam[i])  for i in range(J)])
    qn    = np.array([warp_srsf(q[i], gam[i])  for i in range(J)])
    if not fixed:
        mf0 = f0(yr).mean(axis=0)
    mf    = srsf_inverse_mv(mq, mf0) if mv else srsf_inverse(mq, mf0)
    return dict(y=yr, warps=gam, template=mf, q=qn, niter=niter, cost=np.array(costs))


def srsf_inverse_mv(q, y0=0.0):
    """Reconstruct a (Q,D) multivariate observation from its vector SRSF:  f = y0 + int q ||q||."""
    q  = np.asarray(q, dtype=float)
    t  = _warp.grid(q.shape[0])
    n  = np.linalg.norm(q, axis=1)[:, None]
    return integrate.cumulative_trapezoid(q*n, t, axis=0, initial=0) + np.asarray(y0, dtype=float)



# ---------------------------------------------------------------------
# elastic distances
# ---------------------------------------------------------------------

def amplitude_distance(y1, y2, **kwargs):
    '''
    Elastic amplitude distance:  min_gamma || q1 - (q2 o gamma) sqrt(gamma') ||
    '''
    q1  = srsf(y1)
    q2  = srsf(y2)
    gam = align_srsf_pair(q1, q2, **kwargs)
    t   = _warp.grid(q1.size)
    return float(np.sqrt(integrate.trapezoid((q1 - warp_srsf(q2, gam))**2, t)))


def phase_distance(y1, y2, **kwargs):
    '''
    Elastic phase distance:  arccos( int sqrt(gamma*') dt ),  the geodesic
    distance on the sphere between the optimal warp and the identity.
    '''
    q1  = srsf(y1)
    q2  = srsf(y2)
    gam = align_srsf_pair(q1, q2, **kwargs)
    t   = _warp.grid(q1.size)
    c   = np.clip(integrate.trapezoid(_warp.warp_to_psi(gam), t), -1, 1)
    return float(np.arccos(c))
