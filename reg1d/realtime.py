'''
Real-time registration of observations of different lengths.

Ordinary (normalised-time) registration first interpolates every
observation onto a common grid over [0,1]. That rescales each observation's
time axis by its own duration, so first derivatives -- and therefore the
square-root slope functions on which elastic registration operates -- are
expressed in different physical time units for observations of different
durations. Real-time registration avoids this: every observation stays on
its own time grid t_i (seconds, or frames), the dynamic programming
compares slopes in physical time, and the warps map a common REFERENCE
time axis onto each observation's real time.

Definitions
-----------
Observation i is sampled at times t_i (length n_i, uniform spacing dt_i,
duration T_i = t_i[-1] - t_i[0]). The reference axis is uniform with n_ref
points over [0, T_ref], where T_ref is (by default) the mean duration.
The real-time warp of observation i is

    Gamma_i : [0, T_ref] -> [0, T_i],   monotone, Gamma_i(0)=0, Gamma_i(T_ref)=T_i

and the registered observation is  y_i( t_i[0] + Gamma_i(tau) )  on the
reference axis tau. The NORMALISED warp  gamma_i(s) = Gamma_i(s T_ref) / T_i
is an ordinary [0,1] -> [0,1] warp and is what RegistrationResult.warps
stores; the real-time warp and the real-time displacement field
Gamma_i(tau) - tau T_i/T_ref (deviation from the pure linear rescaling)
are available in the result's info dictionary.

Centering: the normalised warps are centered so that their Karcher mean is
the identity, i.e. the average timing of the registered data is the
average timing of the original data expressed in the reference duration.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import interpolate
from . import warp as _warp
from . import srsf as _srsf
from . import dtw as _dtw
from . import landmark as _landmark



# ---------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------

def is_ragged(y):
    '''True if *y* is a sequence of 1D arrays of (possibly) different lengths.'''
    if isinstance(y, np.ndarray) and y.dtype != object:
        return False
    try:
        lengths = {np.asarray(yy).shape[0]  for yy in y}
    except TypeError:
        return False
    return len(y) > 0 and all(np.asarray(yy).ndim in (1, 2)  for yy in y)


def prepare(y, t=None):
    '''
    Normalise ragged input to lists of arrays and time vectors.

    *y* : sequence of J arrays, each (n_i,) or (n_i, D)
    *t* : None (frames: t_i = arange(n_i)), a scalar sampling interval dt
          (t_i = dt * arange(n_i)), a scalar sampling frequency given as
          the string 'fs=<value>', or a sequence of J uniformly spaced time
          vectors

    Returns (ylist, tlist, dts, durations).
    '''
    ylist = [np.asarray(yy, dtype=float)  for yy in y]
    J     = len(ylist)
    if t is None:
        tlist = [np.arange(yy.shape[0], dtype=float)  for yy in ylist]
    elif np.isscalar(t) and not isinstance(t, str):
        tlist = [float(t) * np.arange(yy.shape[0])  for yy in ylist]
    elif isinstance(t, str) and t.startswith('fs='):
        fs    = float(t[3:])
        tlist = [np.arange(yy.shape[0]) / fs  for yy in ylist]
    else:
        tlist = [np.asarray(tt, dtype=float)  for tt in t]
        if len(tlist) != J:
            raise ValueError('t must contain one time vector per observation')
    dts = []
    for yy, tt in zip(ylist, tlist):
        if tt.shape[0] != yy.shape[0]:
            raise ValueError('each time vector must have the same length as its observation')
        d = np.diff(tt)
        if np.any(d <= 0) or not np.allclose(d, d[0], rtol=1e-6, atol=1e-12):
            raise ValueError('time vectors must be uniformly spaced and increasing')
        dts.append(float(d[0]))
    durations = np.array([tt[-1] - tt[0]  for tt in tlist])
    return ylist, tlist, np.array(dts), durations


def reference_axis(durations, n_ref=101, T_ref=None):
    '''Uniform reference time axis of n_ref points over [0, T_ref] (default T_ref = mean duration).'''
    T_ref = float(np.mean(durations)) if T_ref is None else float(T_ref)
    return np.linspace(0, T_ref, n_ref)


def _eval_on_reference(yy, tt, Gamma):
    '''y_i( t_i[0] + Gamma(tau) ) for a (n_i,) or (n_i,D) observation.'''
    x = tt[0] + Gamma
    if yy.ndim == 1:
        return np.interp(x, tt, yy)
    return np.column_stack([np.interp(x, tt, yy[:, k])  for k in range(yy.shape[1])])


def _to_realtime(gam_norm, T_i, tau):
    '''Normalised warp on the reference grid -> real-time warp Gamma_i(tau) in [0, T_i].'''
    return T_i * gam_norm


def _package(ylist, tlist, durations, tau, gam_norm, template, extra):
    '''Assemble the output dictionary common to all real-time methods.'''
    J     = len(ylist)
    T_ref = tau[-1]
    G     = np.array([durations[i] * gam_norm[i]  for i in range(J)])          # real-time warps (J, n_ref)
    yr    = np.array([_eval_on_reference(ylist[i], tlist[i], G[i])  for i in range(J)])
    lin   = tau[None, :] * (durations / T_ref)[:, None]                          # pure linear rescaling
    y0    = np.array([_eval_on_reference(ylist[i], tlist[i], lin[i])  for i in range(J)])  # linearly rescaled only
    out   = dict(y=yr, y0=y0, warps=gam_norm, template=template, t=tau, durations=durations,
                 warps_realtime=G, displacement_realtime=G - lin, lengths=[yy.shape[0]  for yy in ylist])
    out.update(extra)
    return out



# ---------------------------------------------------------------------
# SRSF in real time
# ---------------------------------------------------------------------

def _srsf_realtime(yy, dt, smooth):
    '''SRSF with the derivative taken in real time.'''
    if yy.ndim == 2:
        q = _srsf._srsf_mv(yy, smooth)                       # computed on the unit interval ...
        n = yy.shape[0]
        return q / np.sqrt((n - 1) * dt)                     # ... rescaled: f'_real = f'_unit / ((n-1) dt); q scales with sqrt
    q = _srsf.srsf(yy, smooth=smooth)
    n = yy.shape[0]
    return q / np.sqrt((n - 1) * dt)


def align_group_srsf(y, t=None, n_ref=101, T_ref=None, template='karcher', method='mean',
    max_iter=20, tol=1e-3, center=True, anchor=None, max_step=6, nsub=4, lam='auto', band=None, smooth=0,
    verbose=False):
    '''
    Real-time elastic registration of observations of different lengths.

    The template lives on the reference axis (n_ref points over [0, T_ref]);
    each observation is aligned to it on its OWN grid (Q1 = n_ref, Q2 = n_i,
    dt1 = T_ref/(n_ref-1), dt2 = dt_i) by dynamic programming, so slopes are
    compared in physical time and the observations are never resampled
    before alignment.

    Returns a dict with keys 'y' (J,n_ref[,D]) registered on the reference
    axis, 'warps' (J,n_ref) normalised warps, 'warps_realtime' (J,n_ref) in
    time units, 'displacement_realtime', 'template', 't' (reference axis),
    'durations', 'lengths', 'q', 'niter', 'cost'.
    '''
    ylist, tlist, dts, durations = prepare(y, t)
    J     = len(ylist)
    tau   = reference_axis(durations, n_ref, T_ref)
    dtr   = tau[1] - tau[0]
    Tr    = tau[-1]
    q     = [_srsf_realtime(yy, dt, smooth)  for yy, dt in zip(ylist, dts)]
    mv    = ylist[0].ndim == 2
    # --- initial template on the reference axis: the observation nearest the mean
    #     (all observations linearly rescaled to the reference axis for this choice only)
    qref  = np.array([_resample_q(qi, tau, Tr, durations[i])  for i, qi in enumerate(q)])
    fixed = True
    if isinstance(template, str) and template == 'karcher':
        d     = ((qref - qref.mean(axis=0))**2).sum(axis=tuple(range(1, qref.ndim)))
        ind   = int(np.argmin(d))
        mq    = qref[ind].copy()
        fixed = False
    elif isinstance(template, str) and template == 'first':
        mq = qref[0].copy()
    elif isinstance(template, (int, np.integer)):
        mq = qref[int(template)].copy()
    else:
        tmpl = np.asarray(template, dtype=float)
        if tmpl.shape[0] != n_ref:
            raise ValueError('an explicit template must be sampled on the reference axis (n_ref points)')
        mq = _srsf_realtime(tmpl, dtr, smooth)
    center = _warp.resolve_center(center, 'karcher')
    if isinstance(lam, str) and lam == 'auto':
        lam = _srsf.auto_lam(qref, mq, tau)     # energies of the linearly rescaled SRSFs (same scale as the DP cost)
    # --- iterate
    gam   = np.zeros((J, n_ref))
    costs = []
    niter = 0
    for it in range(1 if fixed else max_iter):
        niter += 1
        qn = []
        for i in range(J):
            jidx   = _srsf.align_srsf_pair(mq, q[i], max_step=max_step, nsub=nsub, lam=lam, band=band,
                                           dt1=dtr, dt2=dts[i], return_index=True)
            gam[i] = _warp.normalize_warp(jidx / (ylist[i].shape[0] - 1))
            qn.append(_warp_q_realtime(q[i], gam[i], durations[i], Tr))
        qn    = np.array(qn)
        cost  = float(((qn - mq)**2).sum() * dtr)
        costs.append(cost)
        if verbose:
            print(f'iteration {it+1}: cost = {cost:.6g}')
        if fixed:
            break
        if method == 'median':
            dist   = np.sqrt(((qn - mq)**2).sum(axis=tuple(range(1, qn.ndim))) * dtr)
            wts    = 1.0 / np.maximum(dist, 1e-8)
            mq_new = np.tensordot(wts, qn, axes=(0, 0)) / wts.sum()
        else:
            mq_new = qn.mean(axis=0)
        change = np.linalg.norm(mq_new - mq) / max(np.linalg.norm(mq), 1e-12)
        mq     = mq_new
        if change < tol:
            break
    if center != 'none' and J > 1:
        gam, gmean = _warp.center_warps(gam, method=center, anchor=anchor)
        mq         = _srsf.warp_srsf(mq, _warp.invert(gmean))
    qn   = np.array([_warp_q_realtime(q[i], gam[i], durations[i], Tr)  for i in range(J)])
    out  = _package(ylist, tlist, durations, tau, gam, None, dict(q=qn, niter=niter, cost=np.array(costs), lam=lam, center=center))
    f0   = out['y'][:, 0].mean(axis=0)
    out['template'] = _srsf_inverse_realtime(mq, f0, dtr)
    return out


def _resample_q(qi, tau, Tr, Ti):
    '''Linearly rescale an SRSF from its own duration to the reference axis (for template initialisation only).'''
    n  = qi.shape[0]
    s  = np.linspace(0, 1, n)
    x  = tau / Tr
    if qi.ndim == 1:
        return np.interp(x, s, qi) * np.sqrt(Ti / Tr)
    return np.column_stack([np.interp(x, s, qi[:, k])  for k in range(qi.shape[1])]) * np.sqrt(Ti / Tr)


def _warp_q_realtime(qi, gam_norm, Ti, Tr):
    '''(q_i o Gamma) sqrt(Gamma') on the reference axis, Gamma = Ti * gam_norm(tau/Tr).'''
    n   = qi.shape[0]
    s   = np.linspace(0, 1, n)
    dG  = np.gradient(Ti * gam_norm, Tr / (gam_norm.size - 1))      # dGamma/dtau
    if qi.ndim == 1:
        return np.interp(gam_norm, s, qi) * np.sqrt(np.clip(dG, 0, None))
    return np.column_stack([np.interp(gam_norm, s, qi[:, k])  for k in range(qi.shape[1])]) * np.sqrt(np.clip(dG, 0, None))[:, None]


def _srsf_inverse_realtime(q, f0, dt):
    from scipy import integrate
    if q.ndim == 1:
        return f0 + integrate.cumulative_trapezoid(q * np.abs(q), dx=dt, initial=0)
    n = np.linalg.norm(q, axis=1)[:, None]
    return f0 + integrate.cumulative_trapezoid(q * n, dx=dt, axis=0, initial=0)



# ---------------------------------------------------------------------
# DTW in real time
# ---------------------------------------------------------------------

def align_group_dtw(y, t=None, n_ref=101, T_ref=None, template='mean', max_iter=10, tol=1e-4,
    step_pattern='symmetric2', window=None, p=2, derivative=False, smooth=0.0, center=True, anchor=None,
    verbose=False):
    '''
    Real-time DTW registration: each observation (own grid, n_i points) is
    aligned to the template on the reference axis (n_ref points); with
    derivative=True the derivative estimates are divided by the sampling
    intervals so that they are compared in physical time.
    '''
    ylist, tlist, dts, durations = prepare(y, t)
    J     = len(ylist)
    tau   = reference_axis(durations, n_ref, T_ref)
    dtr   = tau[1] - tau[0]
    if any(yy.ndim == 2  for yy in ylist):
        raise ValueError('real-time DTW registration supports univariate observations only')
    # initial template: linear rescaling to the reference axis
    yref  = np.array([np.interp(tau / tau[-1], np.linspace(0, 1, yy.size), yy)  for yy in ylist])
    fixed = False
    if isinstance(template, str) and template in ('mean', 'dba', 'medoid'):
        tmpl = yref.mean(axis=0)
    elif isinstance(template, (int, np.integer)):
        tmpl, fixed = yref[int(template)].copy(), True
    else:
        tmpl, fixed = np.asarray(template, dtype=float), True
    gam   = np.zeros((J, n_ref))
    dist  = np.zeros(J)
    niter = 0
    for it in range(1 if fixed else max_iter):
        niter += 1
        for i in range(J):
            a, b = tmpl, ylist[i]
            if derivative:
                a = _dtw.derivative_estimate(tmpl) / dtr
                b = _dtw.derivative_estimate(ylist[i]) / dts[i]
            try:
                path, dist[i] = _dtw.dtw_path(a, b, step_pattern=step_pattern, window=window, p=p)
            except RuntimeError as err:
                raise RuntimeError(f'{err} (observation {i}: {ylist[i].size} points versus {n_ref} reference '
                    f'points; slope-constrained step patterns cannot bridge length ratios beyond their slope '
                    f'range -- use symmetric1/symmetric2, or choose n_ref close to the observation lengths)') from err
            g = _dtw.path_to_warp(path, n_ref, ylist[i].size)
            gam[i] = _warp.smooth_warp(g, smooth) if smooth > 0 else g
        yr = np.array([_eval_on_reference(ylist[i], tlist[i], durations[i]*gam[i])  for i in range(J)])
        if verbose:
            print(f'iteration {it+1}: total distance = {dist.sum():.6g}')
        if fixed:
            break
        new    = yr.mean(axis=0)
        change = np.linalg.norm(new - tmpl) / max(np.linalg.norm(tmpl), 1e-12)
        tmpl   = new
        if change < tol:
            break
    center = _warp.resolve_center(center, 'pointwise')
    if center != 'none' and J > 1:
        gam, gref = _warp.center_warps(gam, method=center, anchor=anchor)
        tmpl      = _warp.apply_warp(tmpl, _warp.invert(gref))
    return _package(ylist, tlist, durations, tau, gam, tmpl, dict(distance=dist, niter=niter, center=center))



# ---------------------------------------------------------------------
# landmarks in real time
# ---------------------------------------------------------------------

def align_group_landmark(y, t=None, n_ref=101, T_ref=None, landmarks=None, targets='mean',
    kind='pchip', kinds=('min', 'zero', 'max'), center=False, anchor=None):
    '''
    Real-time landmark registration: landmarks are given (or detected) in
    each observation's own time units; targets are their mean (or median)
    across observations on the reference axis; the warp Gamma_i is the
    monotone interpolant through (0,0), (target_k, landmark_ik), (T_ref, T_i).

    *landmarks* : (J,K) array in the time units of *t*, or None for
                  automatic detection (detect_landmarks on each observation)
    '''
    ylist, tlist, dts, durations = prepare(y, t)
    J     = len(ylist)
    tau   = reference_axis(durations, n_ref, T_ref)
    Tr    = tau[-1]
    if landmarks is None:
        landmarks = np.array([tlist[i][0] + durations[i] * _landmark.detect_landmarks(ylist[i], kinds)
                              for i in range(J)])
    landmarks = np.atleast_2d(np.asarray(landmarks, dtype=float))
    lm_rel    = landmarks - np.array([tt[0]  for tt in tlist])[:, None]        # relative to each start
    if isinstance(targets, str):
        targets = lm_rel.mean(axis=0) if targets == 'mean' else np.median(lm_rel, axis=0)
    targets   = np.asarray(targets, dtype=float)
    gam = np.zeros((J, n_ref))
    for i in range(J):
        x = np.concatenate([[0.0], targets, [Tr]])
        v = np.concatenate([[0.0], lm_rel[i], [durations[i]]])
        if np.any(np.diff(x) <= 0) or np.any(np.diff(v) <= 0):
            raise ValueError('landmarks and targets must be strictly increasing and inside the domain')
        G = np.interp(tau, x, v) if kind == 'linear' else interpolate.PchipInterpolator(x, v)(tau)
        gam[i] = _warp.normalize_warp(G / durations[i])
    center = _warp.resolve_center(center, 'none')
    if center != 'none' and J > 1:
        gam, _ = _warp.center_warps(gam, method=center, anchor=anchor)
    return _package(ylist, tlist, durations, tau, gam, None, dict(landmarks=landmarks, targets=targets, center=center))
