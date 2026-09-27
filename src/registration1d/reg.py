'''
Public registration interface.

All registration functions accept a (J,Q) array of observations (or, for
register_linear, a sequence of observations of arbitrary lengths) and
return a RegistrationResult, which can be unpacked as

    (yr, wf) = register_xxx(y)

for compatibility with nlreg1d.

Time grids
----------
By default observations are assumed to be sampled on the uniform grid
t = linspace(0, 1, Q). An explicit grid may be passed with the keyword
argument *t* (a (Q,) array, strictly increasing, not necessarily uniform).
Internally all methods work on a uniform grid over [t[0], t[-1]]: if *t* is
non-uniform the observations are interpolated onto the uniform grid with the
same number of points before registration, and the results (registered
observations, template, warps) are reported on that uniform grid, which is
available as result.t. Warps are always stored in normalised time [0,1]
(result.warps); result.warps_t gives them in the units of *t*.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from . import warp as _warp
from . import linear as _linear
from . import srsf as _srsf
from . import dtw as _dtw
from . import landmark as _landmark
from . import continuous as _continuous
from . import bayes as _bayes
from . import pairwise as _pairwise
from . import sim as _sim
from . import realtime as _realtime



# ---------------------------------------------------------------------
# time grids
# ---------------------------------------------------------------------

def _prepare_grid(y, t):
    '''
    Return (y_uniform, t_uniform, t_original).

    If *t* is None the grid is linspace(0,1,Q). If *t* is uniform (to within
    numerical precision) it is used as is. Otherwise the observations are
    linearly interpolated onto a uniform grid with the same Q over the same
    span.
    '''
    y  = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    Q  = y.shape[1]
    if t is None:
        tu = np.linspace(0, 1, Q)
        return y, tu, tu
    t  = np.asarray(t, dtype=float)
    if t.shape != (Q,):
        raise ValueError(f't must have shape ({Q},), got {t.shape}')
    if np.any(np.diff(t) <= 0):
        raise ValueError('t must be strictly increasing')
    tu = np.linspace(t[0], t[-1], Q)
    if np.allclose(t, tu):
        return y, tu, t
    if y.ndim == 3:
        yu = np.stack([np.array([np.interp(tu, t, yy)  for yy in y[:, :, k]])  for k in range(y.shape[2])], axis=2)
    else:
        yu = np.array([np.interp(tu, t, yy)  for yy in y])
    return yu, tu, t



def _resolve_anchor(anchor, y):
    '''
    'max' / 'min' -> (J,) normalised times of each observation's extremum
    (first component for multivariate data); arrays are passed through.
    '''
    if anchor is None or not isinstance(anchor, str):
        return anchor
    out = []
    for yy in y:
        yy = np.asarray(yy, dtype=float)
        v  = yy[:, 0] if yy.ndim == 2 else yy
        k  = int(np.argmax(v)) if anchor == 'max' else int(np.argmin(v))
        out.append(k / (v.size - 1))
    return np.array(out)


def _realtime_result(r, method, extra_keys):
    '''Wrap a reg1d.realtime output dictionary in a NonlinearRegistrationResult.'''
    tau  = r['t']
    info = dict(realtime=True, durations=r['durations'], lengths=r['lengths'],
                warps_realtime=r['warps_realtime'], displacement_realtime=r['displacement_realtime'])
    for k in extra_keys:
        info[k] = r[k]
    return NonlinearRegistrationResult(r['y'], r['y0'], r['warps'], r['template'], method, info,
        t=tau, t_original=tau)



# ---------------------------------------------------------------------
# result objects
# ---------------------------------------------------------------------

class RegistrationResult(object):
    '''
    Container for the output of a registration.

    Attributes:
        y        : (J,Q) registered observations  ((J,Q,D) for multivariate data)
        y0       : (J,Q) input observations (on the uniform grid)
        t        : (Q,) time grid on which y, y0, template are sampled
        warps    : Warp1DList in normalised time  (y[i] = y0[i]( warps[i] ))
        template : (Q,) template to which the observations were aligned (or None)
        method   : name of the method
        islinear : True for linear methods (linear, shift, affine)
        info     : dict of method-specific extras (iterations, costs, landmarks, ...)

    Tuple unpacking returns (y, warps-as-array):

    >>> yr, wf = reg1d.register_srsf( y )

    Applying the same warps to another variable measured on the same time
    base (e.g. registering on vertical GRF and applying to joint angles):

    >>> result = reg1d.register_srsf( grf )
    >>> angles_registered = result.apply( angles )      # angles: (J,Q)

    Undoing the registration (e.g. mapping the template back to each
    observation's own time base):

    >>> y_back = result.unapply( result.y )             # approximately y0
    '''
    islinear = None

    def __init__(self, y, y0, warps, template=None, method=None, info=None, t=None, t_original=None):
        self.y          = np.asarray(y, dtype=float)
        self.y0         = np.asarray(y0, dtype=float)
        self.t          = np.linspace(0, 1, self.y.shape[1]) if t is None else np.asarray(t, dtype=float)
        self.t_original = self.t if t_original is None else np.asarray(t_original, dtype=float)
        self.warps      = warps if isinstance(warps, _warp.Warp1DList) else _warp.Warp1DList(warps)
        self.template   = None if template is None else np.asarray(template, dtype=float)
        self.method     = method
        self.info       = {} if info is None else info

    def __iter__(self):
        yield self.y
        yield self.warps.asarray()

    def __repr__(self):
        s  = f'{self.__class__.__name__} ({self.method})\n'
        s += f'    y        : {self.y.shape}\n'
        s += f'    warps    : {self.warps.shape}\n'
        s += f'    t        : [{self.t[0]:g}, {self.t[-1]:g}] ({self.t.size} points)\n'
        s += f'    template : {None if self.template is None else self.template.shape}\n'
        s += f'    islinear : {self.islinear}\n'
        s += f'    info     : {list(self.info.keys())}\n'
        return s

    # --- basic properties
    @property
    def J(self):
        return self.y.shape[0]

    @property
    def Q(self):
        return self.y.shape[1]

    @property
    def isnonlinear(self):
        return not self.islinear

    @property
    def displacement_fields(self):
        '''Displacement fields in normalised time (see warp.displacement_field).'''
        return self.warps.displacement_field()

    @property
    def displacement_fields_t(self):
        '''Displacement fields in the units of t.'''
        return self.displacement_fields * (self.t[-1] - self.t[0])

    @property
    def inverse_warps(self):
        '''Warp1DList of inverse warps  (y0[i] = y[i]( inverse_warps[i] ), approximately).'''
        return self.warps.inverse()

    @property
    def warps_t(self):
        '''Warps expressed in the units of t: (J,Q) array with values in [t0, t1].'''
        return self.t[0] + (self.t[-1] - self.t[0]) * self.warps.asarray()

    # --- applying and undoing warps
    def apply(self, z):
        '''
        Apply the registration warps to another set of observations *z*
        ((J,Q) array, or (J,Q,D) for multivariate data) sampled on the same
        time grid as the registered data:  z_registered[i] = z[i]( warps[i] ).
        '''
        if self.info.get('realtime', False) and _realtime.is_ragged(z):
            # real-time result: z_i sampled on observation i's own grid (same length as y_i)
            G = self.info['warps_realtime']
            out = []
            for i, zz in enumerate(z):
                zz = np.asarray(zz, dtype=float)
                if zz.shape[0] != self.info['lengths'][i]:
                    raise ValueError(f'observation {i}: expected {self.info["lengths"][i]} points, got {zz.shape[0]}')
                tt = np.linspace(0, self.info['durations'][i], zz.shape[0])
                out.append(_realtime._eval_on_reference(zz, tt, G[i]))
            return np.array(out)
        z = np.asarray(z, dtype=float)
        if z.ndim == 3:
            return np.stack([self.apply(z[:, :, k])  for k in range(z.shape[2])], axis=2)
        if z.shape[0] != self.J:
            raise ValueError(f'expected {self.J} observations, got {z.shape[0]}')
        return self.warps.apply(z)

    def unapply(self, z):
        '''
        Apply the INVERSE warps to *z* (a (J,Q) array in registered time),
        mapping it back to each observation's original time base. Passing a
        single (Q,) array (e.g. the template) broadcasts it to all J.
        '''
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = np.tile(z, (self.J, 1))
        if z.ndim == 3:
            return np.stack([self.unapply(z[:, :, k])  for k in range(z.shape[2])], axis=2)
        return self.inverse_warps.apply(z)

    # --- diagnostics
    def sse(self):
        '''Total squared deviation from the cross-sectional mean, before and after.'''
        f = lambda a: float(((a - a.mean(axis=0))**2).sum())
        return f(self.y0), f(self.y)

    def plot(self, group=None, backend='matplotlib', **kwargs):
        '''
        Three-panel summary (before / after / warps).

        *backend* : 'matplotlib' (returns (fig, axes)) or 'pyqtgraph' (returns a
                    GraphicsLayoutWidget; requires the optional [qt] extra)
        '''
        if backend == 'pyqtgraph':
            from . import plotqt
            return plotqt.plot_registration(self, group=group, **kwargs)
        from . import plot
        return plot.plot_registration(self, group=group, **kwargs)



class LinearRegistrationResult(RegistrationResult):
    '''
    Result of a linear registration (linear, shift, affine).

    The warps of shift and affine registration are affine maps that may
    leave the unit interval; they are stored unnormalised. Where an
    observation is evaluated outside its original domain the value is
    determined by the fill rule used by the method (see info['fill_value']).
    '''
    islinear = True



class NonlinearRegistrationResult(RegistrationResult):
    '''Result of a nonlinear registration (boundary-preserving warps).'''
    islinear = False



class _AffineWarp(_warp.Warp1D):
    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

class _AffineWarps(_warp.Warp1DList):
    '''Warp1DList whose elements are not normalised to [0,1] (affine maps may leave the unit interval).'''
    def __init__(self, w):
        list.__init__(self, [_AffineWarp(ww)  for ww in np.atleast_2d(w)])



# ---------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------

def register_linear(y, n=101, kind='linear', t=None):
    '''
    Linearly register (interpolate) one observation, or a sequence of
    observations of arbitrary lengths, to *n* equally spaced points.

    >>> result = reg1d.register_linear( ylist, n=101 )   # list of arrays
    >>> yi     = result.y                                # (J,n) array
    >>> yi, wf = reg1d.register_linear( ylist, n=101 )   # tuple unpacking

    For this method y0 is the input resampled to n points as well (the
    inputs generally have different lengths), and the warps are identities;
    the original lengths are in info['lengths'].

    *t* : optional (t0, t1) span for the output grid; defaults to (0, 1)
    '''
    single = isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype != object
    ylist  = [np.asarray(y, dtype=float)] if single else [np.asarray(yy, dtype=float)  for yy in y]
    yi     = _linear.resample(ylist, n=n, kind=kind)
    if t is None:
        tu = np.linspace(0, 1, n)
    else:
        t  = np.asarray(t, dtype=float)
        tu = np.linspace(t[0], t[-1], n)
    J      = len(ylist)
    w      = np.tile(np.linspace(0, 1, n), (J, 1))
    return LinearRegistrationResult(yi, yi, w, None, 'linear',
        dict(lengths=[yy.size  for yy in ylist], kind=kind), t=tu)


def register_shift(y, t=None, **kwargs):
    '''
    Shift registration:  gamma_i(t) = t + delta_i  (least-squares, Procrustes iteration).
    Keyword arguments: template, max_iter, tol, center, fill_value, max_shift, verbose.
    '''
    yu, tu, t0 = _prepare_grid(y, t)
    r = _linear.align_group(yu, method='shift', **kwargs)
    return LinearRegistrationResult(r['y'], yu, _AffineWarps(r['warps']), r['template'], 'shift',
        dict(shift=r['params'], niter=r['niter'], fill_value=r['fill_value']), t=tu, t_original=t0)


def register_affine(y, t=None, **kwargs):
    '''
    Affine registration:  gamma_i(t) = a_i t + b_i  (least-squares, Procrustes iteration).
    Keyword arguments: template, max_iter, tol, center, fill_value, max_shift,
    scale_range, cover, verbose  (see linear.affine_pair for *cover*).
    '''
    yu, tu, t0 = _prepare_grid(y, t)
    r = _linear.align_group(yu, method='affine', **kwargs)
    return LinearRegistrationResult(r['y'], yu, _AffineWarps(r['warps']), r['template'], 'affine',
        dict(params=r['params'], niter=r['niter'], fill_value=r['fill_value']), t=tu, t_original=t0)



# ---------------------------------------------------------------------
# nonlinear
# ---------------------------------------------------------------------

def register_srsf(y, t=None, template='karcher', method='mean', max_iter=20, tol=1e-3, center=True,
    anchor=None, max_step=6, nsub=4, lam='auto', band=None, smooth=0, refine=False, parallel=False,
    verbose=False, n_ref=101, T_ref=None):
    '''
    Elastic (SRSF / Fisher-Rao) registration by dynamic programming with
    an iteratively updated Karcher-mean (or median) template.

    >>> yr, wf = reg1d.register_srsf( y, max_iter=5 )

    *y*         : (J,Q) array, or (J,Q,D) array of D-variate observations
    *template*  : 'karcher' | 'first' | int | (Q,) array
    *method*    : 'mean' (Karcher mean template) or 'median' (Karcher median)
    *max_iter*  : maximum number of template updates
    *center*    : 'karcher' (default, = True) | 'pointwise' | 'anchor' | 'none' (= False):
                  choice of the common time axis of the registered data (see
                  warp.center_warps and the WarpCentering notebook)
    *anchor*    : (J,) event times in normalised original time for center='anchor',
                  or 'max' / 'min' to use the time of each observation's extremum
    *max_step*  : slope set for dynamic programming (6 -> local slopes 1/6 ... 6)
    *nsub*      : sub-samples per grid step in the segment-cost integrals
    *lam*       : elasticity penalty on departure from the identity warp:
                  'auto' (default; the median SRSF energy, i.e. median total
                  variation, of the observations, see srsf.auto_lam), or a
                  number (0 = none). The value used is reported in info['lam']
    *band*      : Sakoe-Chiba band half-width in normalised time (None = unconstrained)
    *smooth*    : 0 (none), an integer (moving-average passes) or 'spline'
                  (smoothing-spline derivative) for the SRSF computation
    *refine*    : gradient-based refinement of each dynamic-programming warp
    *parallel*  : align observations in parallel processes (bool or number of workers)

    Real-time registration: if *y* is a sequence of observations of different
    lengths (a list, or an object array), the observations are NOT resampled;
    each is aligned on its own time grid (given by *t*: None = frames, a
    scalar sampling interval, 'fs=<Hz>', or one time vector per observation)
    to a template on a reference axis of *n_ref* points over [0, *T_ref*]
    (default: the mean duration). See reg1d.realtime. The result's
    info['warps_realtime'] and info['displacement_realtime'] are in time units.
    '''
    if _realtime.is_ragged(y):
        anchor = _resolve_anchor(anchor, [np.asarray(yy, dtype=float)  for yy in y])
        r = _realtime.align_group_srsf(y, t=t, n_ref=n_ref, T_ref=T_ref, template=template, method=method,
            max_iter=max_iter, tol=tol, center=center, anchor=anchor, max_step=max_step, nsub=nsub, lam=lam,
            band=band, smooth=smooth, verbose=verbose)
        return _realtime_result(r, 'srsf', ('q', 'niter', 'cost', 'lam', 'center'))
    yu, tu, t0 = _prepare_grid(y, t)
    anchor = _resolve_anchor(anchor, yu)
    r = _srsf.align_group(yu, template=template, method=method, max_iter=max_iter, tol=tol,
        center=center, anchor=anchor, max_step=max_step, nsub=nsub, lam=lam, band=band, smooth=smooth,
        refine=refine, parallel=parallel, verbose=verbose)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], r['template'], 'srsf',
        dict(niter=r['niter'], cost=r['cost'], q=r['q'], lam=r['lam'], center=r['center']), t=tu, t_original=t0)


def register_dtw(y, t=None, template='mean', max_iter=10, step_pattern='symmetric2', window=None,
    p=2, derivative=False, smooth=0.0, center=True, anchor=None, verbose=False, n_ref=101, T_ref=None):
    '''
    Dynamic time warping registration with an iteratively refined template
    (DTW barycentre averaging).

    *template*     : 'mean' | 'medoid' | int | (Q,) array
    *step_pattern* : 'symmetric1' | 'symmetric2' | 'asymmetric' | 'strict'
    *window*       : Sakoe-Chiba band half-width (fraction of the domain), or None
    *p*            : exponent of the local distance |x_i - y_j|^p
    *derivative*   : if True, align on the estimated first derivatives
                     (derivative DTW, Keogh & Pazzani 2001) rather than the values
    *smooth*       : Gaussian smoothing width (normalised time; 0 = none) applied
                     to sqrt(gamma') of each DTW warp, which turns the piecewise
                     path into a smooth, strictly increasing warp
    '''
    if _realtime.is_ragged(y):
        anchor = _resolve_anchor(anchor, [np.asarray(yy, dtype=float)  for yy in y])
        r = _realtime.align_group_dtw(y, t=t, n_ref=n_ref, T_ref=T_ref, template=template, max_iter=max_iter,
            step_pattern=step_pattern, window=window, p=p, derivative=derivative, smooth=smooth,
            center=center, anchor=anchor, verbose=verbose)
        return _realtime_result(r, 'dtw', ('distance', 'niter', 'center'))
    yu, tu, t0 = _prepare_grid(y, t)
    anchor = _resolve_anchor(anchor, yu)
    r = _dtw.align_group(yu, template=template, max_iter=max_iter, step_pattern=step_pattern,
        window=window, p=p, derivative=derivative, smooth=smooth, center=center, anchor=anchor, verbose=verbose)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], r['template'], 'dtw',
        dict(niter=r['niter'], distance=r['distance'], center=r['center']), t=tu, t_original=t0)


def register_landmark(y, t=None, landmarks=None, targets='mean', kind='pchip', kinds=('min', 'zero', 'max'),
    center=False, anchor=None, n_ref=101, T_ref=None):
    '''
    Landmark registration.

    *landmarks* : (J,K) landmark times in normalised time (0,1) (or in the
                  units of *t* if *t* is given), or None for automatic
                  detection of the landmark kinds in *kinds*
    *targets*   : 'mean' | 'median' | (K,) array
    *kind*      : warp interpolation between landmarks: 'pchip' | 'linear'
    *center*    : 'none' (default, = False) | 'karcher' | 'pointwise' | 'anchor'. Mean
                  targets already anchor the registered data at the landmarks;
                  other centering methods move the landmarks off their targets.

    Ragged input (observations of different lengths) triggers real-time
    registration, with landmarks in the time units of *t*; see reg1d.realtime.
    '''
    if _realtime.is_ragged(y):
        anchor = _resolve_anchor(anchor, [np.asarray(yy, dtype=float)  for yy in y])
        r = _realtime.align_group_landmark(y, t=t, n_ref=n_ref, T_ref=T_ref, landmarks=landmarks,
            targets=targets, kind=kind, kinds=kinds, center=center, anchor=anchor)
        return _realtime_result(r, 'landmark', ('landmarks', 'targets', 'center'))
    yu, tu, t0 = _prepare_grid(y, t)
    if landmarks is not None and t is not None:
        landmarks = (np.asarray(landmarks, dtype=float) - tu[0]) / (tu[-1] - tu[0])
    if not isinstance(targets, str) and t is not None:
        targets = (np.asarray(targets, dtype=float) - tu[0]) / (tu[-1] - tu[0])
    anchor = _resolve_anchor(anchor, yu)
    r = _landmark.align_group(yu, landmarks=landmarks, targets=targets, kind=kind, kinds=kinds,
        center=center, anchor=anchor)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], None, 'landmark',
        dict(landmarks=r['landmarks'], targets=r['targets'], center=r['center']), t=tu, t_original=t0)


def register_continuous(y, t=None, template='mean', n_basis=4, lam=1e-2, max_iter=5, center=True, anchor=None,
    verbose=False):
    '''
    Continuous (parametric, penalised least-squares) registration with
    smooth monotone warps gamma = int exp(W), W in a cosine basis.

    *n_basis* : number of basis functions (degrees of freedom of each warp)
    *lam*     : roughness penalty (dimensionless, relative to the variance of the template; 0 = none)
    *center*  : 'karcher' (default, = True) | 'pointwise' | 'anchor' | 'none' (= False)
    '''
    yu, tu, t0 = _prepare_grid(y, t)
    anchor = _resolve_anchor(anchor, yu)
    r = _continuous.align_group(yu, template=template, n_basis=n_basis, lam=lam, max_iter=max_iter,
        center=center, anchor=anchor, verbose=verbose)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], r['template'], 'continuous',
        dict(niter=r['niter'], coef=r['coef'], center=r['center']), t=tu, t_original=t0)


def register_bayes(y, t=None, template='srsf', n_samples=1000, burn=500, thin=1, K=8, tau=0.3,
    beta=0.05, init='dp', n_eff=None, smooth=0, center=True, anchor=None, random_state=None, verbose=False,
    **srsf_kwargs):
    '''
    Bayesian registration (posterior sampling of each warp; see reg1d.bayes).

    The returned warps are the posterior-mean (Karcher mean) warps. Posterior
    samples are in info['samples'] ((J,S,Q)), pointwise 95 % credible
    intervals of the displacement fields in info['disp_ci'] ((J,2,Q)),
    acceptance rates in info['accept'].
    *center* : 'karcher' (default, = True) | 'pointwise' | 'anchor' | 'none' (= False),
    applied to the posterior-mean warps and to all samples.
    '''
    yu, tu, t0 = _prepare_grid(y, t)
    anchor = _resolve_anchor(anchor, yu)
    r = _bayes.align_group(yu, template=template, n_samples=n_samples, burn=burn, thin=thin, K=K,
        tau=tau, beta=beta, init=init, n_eff=n_eff, smooth=smooth, center=center, anchor=anchor,
        random_state=random_state, verbose=verbose, **srsf_kwargs)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], r['template'], 'bayes',
        dict(samples=r['samples'], sigma2=r['sigma2'], accept=r['accept'], disp_ci=r['disp_ci'], center=r['center']),
        t=tu, t_original=t0)


def register_pairwise(y, t=None, engine='srsf', center=True, anchor=None, **kwargs):
    '''
    Pairwise synchronisation (Tang & Müller 2008): template-free registration
    in which each warp is the Karcher mean of the warps to all other observations.
    *engine* : 'srsf' | 'dtw' | 'continuous' (keyword arguments passed on)
    *center* : 'karcher' (default, = True) | 'pointwise' | 'anchor' | 'none' (= False)
    '''
    yu, tu, t0 = _prepare_grid(y, t)
    anchor = _resolve_anchor(anchor, yu)
    r = _pairwise.align_group(yu, engine=engine, center=center, anchor=anchor, **kwargs)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], r['template'], 'pairwise',
        dict(pairwise=r['pairwise'], engine=engine, center=r['center']), t=tu, t_original=t0)


def register_sim(y, t=None, n_basis=4, lam=1e-2, max_iter=5, center=True, anchor=None, verbose=False):
    '''
    Self-modelling (shape-invariant model) registration:
    y_i = a_i mu(gamma_i) + b_i, with smooth parametric warps (see reg1d.sim).
    Amplitude parameters are in info['amplitude'] ((J,2) array of (a_i, b_i)).
    '''
    yu, tu, t0 = _prepare_grid(y, t)
    anchor = _resolve_anchor(anchor, yu)
    r = _sim.align_group(yu, n_basis=n_basis, lam=lam, max_iter=max_iter, center=center, anchor=anchor, verbose=verbose)
    return NonlinearRegistrationResult(r['y'], yu, r['warps'], r['template'], 'sim',
        dict(amplitude=r['amplitude'], coef=r['coef'], niter=r['niter'], center=r['center']), t=tu, t_original=t0)


METHODS = {
    'linear'     : register_linear,
    'shift'      : register_shift,
    'affine'     : register_affine,
    'srsf'       : register_srsf,
    'dtw'        : register_dtw,
    'landmark'   : register_landmark,
    'continuous' : register_continuous,
    'bayes'      : register_bayes,
    'pairwise'   : register_pairwise,
    'sim'        : register_sim,
}


def register(y, method='srsf', **kwargs):
    '''Dispatch to one of the register_* functions by name (see METHODS).'''
    return METHODS[method](y, **kwargs)
