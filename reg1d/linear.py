'''
Linear registration.

"Linear" registration restricts the warp to an affine map of time,
gamma(t) = a t + b. Three special cases are provided:

    resample : a = (Q-1)/(n-1) in index units, b = 0.  Observations of
               different lengths are interpolated onto a common grid of n
               points (temporal normalisation, e.g. to 0-100 % stance).
               This is the "linear registration" of nlreg1d.
    shift    : a = 1, b = delta_i.  Each observation is translated in time
               to best match a template (Procrustes / least-squares shift
               registration; Ramsay & Silverman 2005, Section 7.2).
    affine   : a and b both free.  Each observation is translated and
               uniformly stretched to best match a template.

For shift and affine registration the observation must be evaluated
outside its original domain; values there are held at the boundary
value ('edge') or set to a constant (fill_value).

References:
    Ramsay JO, Silverman BW (2005). Functional Data Analysis, 2nd ed.
        Springer.  (Section 7.2, "Shift registration")

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import integrate, optimize
from . import warp as _warp



def resample(y, n=101, kind='linear'):
	'''
	Linearly register (interpolate) one observation, or a sequence of
	observations of arbitrary lengths, to *n* equally spaced points.

	*y*    : (Q,) array, or a list / object array of 1D arrays
	*kind* : 'linear' or 'cubic'

	Returns an (n,) array, or a (J,n) array for a sequence of observations.
	'''
	if isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype != object:
		Q   = y.size
		t0  = np.linspace(0, 1, Q)
		ti  = np.linspace(0, 1, n)
		if kind == 'linear':
			return np.interp(ti, t0, y.astype(float))
		from scipy import interpolate
		return interpolate.interp1d(t0, y.astype(float), kind=kind)(ti)
	return np.array([resample(np.asarray(yy, dtype=float), n, kind)  for yy in y])


def _affine_warp(a, b, Q):
	t = _warp.grid(Q)
	return a*t + b


def _eval(y, w, fill_value):
	t = _warp.grid(y.size)
	if fill_value == 'edge':
		return np.interp(w, t, y)
	return np.interp(w, t, y, left=fill_value, right=fill_value)


def _sse(y, target, w, fill_value):
	t = _warp.grid(y.size)
	return integrate.trapezoid((_eval(y, w, fill_value) - target)**2, t)


def shift_pair(y_template, y, max_shift=0.2, fill_value='edge'):
	'''
	Least-squares shift of *y* onto *y_template*:  delta* = argmin int (y(t+delta) - template(t))^2 dt

	Returns (y_aligned, gamma, delta) with gamma(t) = t + delta.
	'''
	y_template = np.asarray(y_template, dtype=float)
	y          = np.asarray(y, dtype=float)
	Q   = y.size
	f   = lambda d: _sse(y, y_template, _affine_warp(1.0, d, Q), fill_value)
	res = optimize.minimize_scalar(f, bounds=(-max_shift, max_shift), method='bounded')
	d   = float(res.x)
	w   = _affine_warp(1.0, d, Q)
	return _eval(y, w, fill_value), w, d


def affine_pair(y_template, y, max_shift=0.2, scale_range=(0.7, 1.4), fill_value='edge'):
	'''
	Least-squares affine time transformation of *y* onto *y_template*:

	    (a*, b*) = argmin int ( y(a t + b) - template(t) )^2 dt

	Returns (y_aligned, gamma, (a, b)) with gamma(t) = a t + b.
	'''
	y_template = np.asarray(y_template, dtype=float)
	y          = np.asarray(y, dtype=float)
	Q   = y.size
	f   = lambda p: _sse(y, y_template, _affine_warp(p[0], p[1], Q), fill_value)
	# coarse grid search followed by local refinement (the objective is not convex)
	A   = np.linspace(scale_range[0], scale_range[1], 15)
	Bg  = np.linspace(-max_shift, max_shift, 15)
	best, p0 = np.inf, (1.0, 0.0)
	for a in A:
		for b in Bg:
			v = f((a, b))
			if v < best:
				best, p0 = v, (a, b)
	res = optimize.minimize(f, np.array(p0), method='Nelder-Mead',
		options=dict(xatol=1e-5, fatol=1e-10))
	a, b = float(res.x[0]), float(res.x[1])
	w   = _affine_warp(a, b, Q)
	return _eval(y, w, fill_value), w, (a, b)


def align_group(y, method='shift', template='mean', max_iter=5, tol=1e-5, center=True,
	fill_value='edge', verbose=False, **kwargs):
	'''
	Shift or affine registration of a set of observations to a template,
	with Procrustes iteration on the cross-sectional mean.

	*y*        : (J,Q) array
	*method*   : 'shift' or 'affine'
	*template* : 'mean' (iteratively updated), an integer index, or a (Q,) array
	*center*   : if True the mean shift (and, for 'affine', the mean scale) is
	             removed so that the registered data keep the original average timing
	*kwargs*   : passed to shift_pair / affine_pair (max_shift, scale_range)

	Returns a dict with keys 'y', 'warps', 'template', 'params', 'niter'.
	Note that the warps are affine maps that generally leave [0,1]; they are
	NOT boundary-preserving warps in the sense of the nonlinear methods.
	'''
	y     = np.atleast_2d(np.asarray(y, dtype=float))
	J, Q  = y.shape
	pair  = shift_pair if method == 'shift' else affine_pair
	fixed = False
	if isinstance(template, str) and template == 'mean':
		tmpl = y.mean(axis=0)
	elif isinstance(template, (int, np.integer)):
		tmpl, fixed = y[int(template)].copy(), True
	else:
		tmpl, fixed = np.asarray(template, dtype=float), True
	gam    = np.zeros((J, Q))
	params = []
	niter  = 0
	for it in range(1 if fixed else max_iter):
		niter += 1
		yr, params = np.empty_like(y), []
		for i in range(J):
			yr[i], gam[i], p = pair(tmpl, y[i], fill_value=fill_value, **kwargs)
			params.append(p)
		if verbose:
			print(f'iteration {it+1}: sse = {((yr - tmpl)**2).sum():.6g}')
		if fixed:
			break
		new    = yr.mean(axis=0)
		change = np.linalg.norm(new - tmpl) / max(np.linalg.norm(tmpl), 1e-12)
		tmpl   = new
		if change < tol:
			break
	params = np.array(params, dtype=float)
	if center and J > 1 and not fixed:
		if method == 'shift':
			params = params - params.mean()
			gam    = np.array([_affine_warp(1.0, d, Q)  for d in params])
		else:
			# remove the mean affine map:  gamma_i <- gamma_i o gbar^{-1}
			abar, bbar = params[:, 0].mean(), params[:, 1].mean()
			params = np.array([(a/abar, b - a*bbar/abar)  for a, b in params])
			gam    = np.array([_affine_warp(a, b, Q)  for a, b in params])
	yr = np.array([_eval(y[i], gam[i], fill_value)  for i in range(J)])
	if not fixed:
		tmpl = yr.mean(axis=0)
	return dict(y=yr, warps=gam, template=tmpl, params=params, niter=niter)
