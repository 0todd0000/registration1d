'''
Warping functions for one-dimensional data.

A warping function ("warp") gamma is a boundary-preserving, monotonically
increasing map of the unit interval onto itself:

    gamma : [0,1] -> [0,1],   gamma(0)=0,  gamma(1)=1,  gamma'(t) > 0

Applying a warp to an observation f yields the time-warped observation
(f o gamma)(t) = f( gamma(t) ). Throughout reg1d, warps are stored as
(Q,) arrays sampled on the uniform grid t = linspace(0, 1, Q), and a set
of J warps is stored as a (J,Q) array.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import interpolate, integrate



# ---------------------------------------------------------------------
# elementary warp operations (all operate on (Q,) or (J,Q) arrays)
# ---------------------------------------------------------------------

def grid(Q):
    '''Uniform grid of Q points on [0,1].'''
    return np.linspace(0, 1, Q)


def apply_warp(y, w, fill_value='edge'):
    '''
    Warp an observation:  returns  y( w(t) )  sampled on the original grid.

    *y* : (Q,) or (J,Q) array
    *w* : (Q,) or (J,Q) array of warp function(s) with values in [0,1]
    '''
    y  = np.asarray(y, dtype=float)
    w  = np.asarray(w, dtype=float)
    if y.ndim == 2:
        if w.ndim == 1:
            w = np.tile(w, (y.shape[0], 1))
        return np.array([apply_warp(yy, ww, fill_value)  for yy,ww in zip(y, w)])
    t  = grid(y.size)
    if fill_value == 'edge':
        return np.interp(w, t, y)
    f  = interpolate.interp1d(t, y, 'linear', bounds_error=False, fill_value=fill_value)
    return f(w)


def compose(w1, w2):
    '''
    Composition of warps:  (w1 o w2)(t) = w1( w2(t) )

    Both arguments are (Q,) arrays on the uniform grid.
    '''
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    return np.interp(w2, grid(w1.size), w1)


def invert(w):
    '''
    Numerical inverse of a warp, sampled on the uniform grid.

    Because w is monotonic, w^{-1} is obtained by swapping the roles of
    abscissa and ordinate and re-interpolating onto the grid.
    '''
    w  = np.asarray(w, dtype=float)
    if w.ndim == 2:
        return np.array([invert(ww)  for ww in w])
    t  = grid(w.size)
    ww = _make_strictly_increasing(w)
    wi = np.interp(t, ww, t)
    wi[0], wi[-1] = 0.0, 1.0
    return wi


def derivative(w):
    '''Numerical derivative of a warp (or any function) on the uniform grid.'''
    w  = np.asarray(w, dtype=float)
    t  = grid(w.shape[-1])
    return np.gradient(w, t, axis=-1)


def displacement(w):
    '''
    Deviation from linear time:  d(t) = w(t) - t

    Positive values mean that the warped observation at time t draws its
    value from a LATER point of the original observation.
    '''
    w  = np.asarray(w, dtype=float)
    return w - grid(w.shape[-1])


def displacement_field(w):
    '''
    Displacement field expressed on the ORIGINAL (unwarped) time axis:

        u(s) = s - w^{-1}(s)   i.e. the negative deviation of the inverse warp

    This is the quantity plotted in nlreg1d as "deviation from linear
    time": it reports, for each point of the original observation, how far
    (and in which direction) it was moved to reach its registered position.
    '''
    w  = np.asarray(w, dtype=float)
    return -displacement(invert(w))


def identity(Q):
    '''Identity warp (linear time) on a Q-point grid.'''
    return grid(Q)


def is_valid_warp(w, tol=1e-8):
    '''True if w is monotonically non-decreasing with w(0)=0 and w(1)=1.'''
    w  = np.asarray(w, dtype=float)
    ok = np.all(np.diff(w, axis=-1) >= -tol, axis=-1)
    ok = ok & np.isclose(w[...,0], 0, atol=tol) & np.isclose(w[...,-1], 1, atol=tol)
    return ok


def normalize_warp(w):
    '''
    Force a candidate warp onto [0,1] with fixed end points and monotonic
    increase (a cumulative maximum is used to remove any small decreases
    arising from numerical error).
    '''
    w  = np.asarray(w, dtype=float)
    if w.ndim == 2:
        return np.array([normalize_warp(ww)  for ww in w])
    w  = np.maximum.accumulate(w)
    w  = (w - w[0]) / (w[-1] - w[0])
    return w


def _make_strictly_increasing(w, eps=1e-10):
    '''Add a negligible ramp so that np.interp receives strictly increasing abscissae.'''
    t  = grid(w.size)
    ww = np.maximum.accumulate(w) + eps * t
    return (ww - ww[0]) / (ww[-1] - ww[0])



# ---------------------------------------------------------------------
# warps as points on the Hilbert sphere (used to average warps)
# ---------------------------------------------------------------------

def warp_to_psi(w):
    '''
    Square-root-slope representation of a warp:  psi = sqrt( gamma' )

    psi lies on the unit sphere of L2[0,1] because int psi^2 = gamma(1)-gamma(0) = 1.
    '''
    d  = derivative(w)
    return np.sqrt(np.clip(d, 0, None))


def psi_to_warp(psi):
    '''Inverse of warp_to_psi:  gamma(t) = int_0^t psi(s)^2 ds  (normalised to end at 1).'''
    psi = np.asarray(psi, dtype=float)
    t   = grid(psi.shape[-1])
    w   = integrate.cumulative_trapezoid(psi**2, t, axis=-1, initial=0)
    return normalize_warp(w)


def _sphere_inner(a, b, t):
    return integrate.trapezoid(a*b, t)


def _sphere_exp(mu, v, t):
    '''Exponential map on the L2 unit sphere at mu, in direction v.'''
    nv = np.sqrt(max(_sphere_inner(v, v, t), 0))
    if nv < 1e-12:
        return mu.copy()
    return np.cos(nv)*mu + np.sin(nv)*v/nv


def _sphere_log(mu, psi, t):
    '''Inverse exponential map on the L2 unit sphere: tangent vector at mu pointing to psi.'''
    c   = np.clip(_sphere_inner(mu, psi, t), -1, 1)
    th  = np.arccos(c)
    if th < 1e-12:
        return np.zeros_like(mu)
    return th/np.sin(th) * (psi - c*mu)


def karcher_mean_warp(w, max_iter=20, tol=1e-6, return_psi=False):
    '''
    Karcher (Frechet) mean of a set of warps under the Fisher-Rao metric.

    The warps are mapped to the unit sphere via psi = sqrt(gamma'), the
    intrinsic mean on the sphere is found by iterating exp/log maps, and
    the result is mapped back to a warp.

    *w* : (J,Q) array of warps
    '''
    w   = np.atleast_2d(np.asarray(w, dtype=float))
    Q   = w.shape[1]
    t   = grid(Q)
    psi = warp_to_psi(w)
    mu  = psi.mean(axis=0)
    mu /= np.sqrt(_sphere_inner(mu, mu, t))
    for _ in range(max_iter):
        v   = np.mean([_sphere_log(mu, p, t)  for p in psi], axis=0)
        nv  = np.sqrt(_sphere_inner(v, v, t))
        mu  = _sphere_exp(mu, v, t)
        if nv < tol:
            break
    wm  = psi_to_warp(mu)
    return (wm, mu) if return_psi else wm


def smooth_warp(w, sigma):
    """
    Smooth a warp while preserving monotonicity and the end points.

    The square-root slope psi = sqrt(gamma') is smoothed with a Gaussian
    kernel of width *sigma* (in normalised time; the kernel is reflected
    at the boundaries), squared and re-integrated. Because psi^2 >= 0 the
    result is always a valid, strictly increasing warp, and the total
    "amount" of warping is preserved. Useful for turning the piecewise
    (staircase-like) paths of dynamic time warping into physically
    plausible warps with smooth first derivatives.

    *w*     : (Q,) or (J,Q) array
    *sigma* : kernel width in normalised time (0 = no smoothing)
    """
    w = np.asarray(w, dtype=float)
    if sigma <= 0:
        return w.copy()
    if w.ndim == 2:
        return np.array([smooth_warp(ww, sigma)  for ww in w])
    from scipy import ndimage
    Q   = w.size
    psi = warp_to_psi(w)
    psi = ndimage.gaussian_filter1d(psi, sigma * (Q - 1), mode='reflect')
    return psi_to_warp(psi)


def center_warps(w):
    '''
    Center a set of warps so that their Karcher mean is the identity.

    Returns (w_centered, w_mean) where  w_centered[i] = w[i] o w_mean^{-1}.
    Applying w_centered to the raw observations yields registered
    observations whose average timing matches the average timing of the
    original data, rather than that of an arbitrary template.
    '''
    w    = np.atleast_2d(np.asarray(w, dtype=float))
    wm   = karcher_mean_warp(w)
    wmi  = invert(wm)
    wc   = np.array([compose(ww, wmi)  for ww in w])
    return wc, wm



# ---------------------------------------------------------------------
# random warps (for simulation)
# ---------------------------------------------------------------------

def random_warp(J=1, Q=101, sigma=0.5, n_basis=5, random_state=None):
    '''
    Random warps generated as random points on the unit sphere of the
    square-root-slope representation.

    A random tangent vector v = sum_k a_k sqrt(2) sin(k pi t), with
    a_k ~ N(0, sigma^2 / k^2), is mapped through the exponential map at
    the identity warp (psi = 1) and integrated back to a warp. Larger
    *sigma* gives stronger warps; larger *n_basis* gives rougher warps.

    Returns a (Q,) array if J==1 else a (J,Q) array.
    '''
    rng = np.random.default_rng(random_state)
    t   = grid(Q)
    one = np.ones(Q)
    ws  = []
    for _ in range(J):
        v   = np.zeros(Q)
        for k in range(1, n_basis+1):
            v += rng.normal(0, sigma/k) * np.sqrt(2) * np.sin(k*np.pi*t)
        v  -= _sphere_inner(v, one, t) * one     # project onto tangent space at identity
        psi = _sphere_exp(one, v, t)
        ws.append(psi_to_warp(psi))
    ws  = np.array(ws)
    return ws[0] if J == 1 else ws



# ---------------------------------------------------------------------
# object-oriented convenience wrappers
# ---------------------------------------------------------------------

class Warp1D(object):
    '''
    A single warp function sampled on the uniform grid.

    >>> w   = Warp1D( wf )          # wf : (Q,) array
    >>> yw  = w.apply( y )          # y( w(t) )
    >>> wi  = w.inverse()           # Warp1D
    >>> d   = w.displacement_field()
    '''
    def __init__(self, w):
        self.w  = normalize_warp(np.asarray(w, dtype=float))

    def __repr__(self):
        return f'Warp1D (Q={self.Q})'

    @property
    def Q(self):
        return self.w.size

    @property
    def t(self):
        return grid(self.Q)

    def apply(self, y):
        return apply_warp(y, self.w)

    def asarray(self):
        return self.w.copy()

    def compose(self, other):
        other = other.w if isinstance(other, Warp1D) else other
        return Warp1D( compose(self.w, other) )

    def derivative(self):
        return derivative(self.w)

    def displacement(self):
        return displacement(self.w)

    def displacement_field(self):
        return displacement_field(self.w)

    def inverse(self):
        return Warp1D( invert(self.w) )

    def smooth(self, sigma):
        return Warp1D( smooth_warp(self.w, sigma) )

    def plot(self, ax=None, **kwargs):
        from . import plot
        return plot.plot_warps(self.w, ax=ax, **kwargs)



class Warp1DList(list):
    '''
    A list of Warp1D objects with array-level conveniences.

    >>> wl  = Warp1DList( wf )      # wf : (J,Q) array
    >>> yw  = wl.apply( y )         # (J,Q) array of warped observations
    >>> wm  = wl.mean()             # Karcher mean warp (Warp1D)
    >>> wc  = wl.center()           # centered copy (Warp1DList)
    '''
    def __init__(self, w):
        w  = np.atleast_2d(np.asarray(w, dtype=float))
        super().__init__( [Warp1D(ww)  for ww in w] )

    def __repr__(self):
        return f'Warp1DList (J={self.J}, Q={self.Q})'

    @property
    def J(self):
        return len(self)

    @property
    def Q(self):
        return self[0].Q if self.J > 0 else 0

    @property
    def shape(self):
        return (self.J, self.Q)

    @property
    def t(self):
        return grid(self.Q)

    def apply(self, y):
        return apply_warp(y, self.asarray())

    def asarray(self):
        return np.array([w.w  for w in self])

    def center(self):
        wc,_ = center_warps(self.asarray())
        return Warp1DList(wc)

    def displacement(self):
        return displacement(self.asarray())

    def displacement_field(self):
        return displacement_field(self.asarray())

    def inverse(self):
        return Warp1DList( invert(self.asarray()) )

    def mean(self):
        return Warp1D( karcher_mean_warp(self.asarray()) )

    def smooth(self, sigma):
        return Warp1DList( smooth_warp(self.asarray(), sigma) )

    def plot(self, ax=None, **kwargs):
        from . import plot
        return plot.plot_warps(self.asarray(), ax=ax, **kwargs)
