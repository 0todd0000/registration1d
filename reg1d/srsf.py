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
	'''
	Square-root slope function of one or more observations.

	*y*      : (Q,) or (J,Q) array sampled on the uniform grid over [0,1]
	*smooth* : number of passes of a 3-point moving average applied to y
	           before differentiation (0 = none)
	'''
	y  = np.asarray(y, dtype=float)
	t  = _warp.grid(y.shape[-1])
	for _ in range(int(smooth)):
		y  = _boxsmooth(y)
	dy = np.gradient(y, t, axis=-1)
	return np.sign(dy) * np.sqrt(np.abs(dy))


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
	'''Group action of a warp on an SRSF:  (q o w) * sqrt( w' )'''
	qw = _warp.apply_warp(q, w)
	return qw * np.sqrt(np.clip(_warp.derivative(w), 0, None))


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


def _segment_costs(q1, q2, di, dj, t, nsub=4):
	'''
	Cost E[i,j] of the linear path segment from node (i,j) to (i+di, j+dj),
	for every start node, vectorised over the whole grid:

	    E[i,j] = int_{t_i}^{t_{i+di}}  ( q1(t) - sqrt(s) q2( t_j + s (t - t_i) ) )^2 dt,   s = dj/di

	q1 and q2 are sampled by linear interpolation at nsub points per grid
	step and the integral is approximated by the trapezoidal rule.
	'''
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
	Q1    = np.interp(I, idx, q1)                   # (n1, m)
	Q2    = np.interp(Jx, idx, q2) * np.sqrt(s)     # (n2, m)
	D     = Q1[:, None, :] - Q2[None, :, :]         # (n1, n2, m)
	D2    = D**2
	E     = dt * (D2[..., 1:] + D2[..., :-1]).sum(axis=-1) / 2.0
	return E


def align_srsf_pair(q1, q2, max_step=6, nsub=4, lam=0.0):
	'''
	Optimal warp aligning SRSF q2 to SRSF q1 by dynamic programming:

	    gamma* = argmin_gamma  || q1 - (q2 o gamma) sqrt(gamma') ||^2  +  lam * R(gamma)

	where R penalises departure from the identity (roughness penalty
	int (sqrt(gamma') - 1)^2 dt, as used in the fdasrsf "lam" argument).

	The admissible path slopes are all coprime (di,dj) pairs with
	1 <= di, dj <= max_step; the path is restricted to be strictly increasing
	so the resulting warp is a valid diffeomorphism. Returns the warp gamma
	sampled on the uniform grid (Q,).

	*max_step* = 6 (default) admits local slopes from 1/6 to 6; *max_step* = 3 admits 1/3, 1/2, 2/3, 1, 3/2, 2, 3.
	'''
	q1    = np.asarray(q1, dtype=float)
	q2    = np.asarray(q2, dtype=float)
	Q     = q1.size
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
	# backtrack from (Q-1, Q-1)
	i, j  = Q-1, Q-1
	if not np.isfinite(D[i, j]):
		raise RuntimeError('dynamic programming failed to reach the end node; increase max_step')
	path  = [(i, j)]
	while (i, j) != (0, 0):
		i, j = P[i, j]
		path.append((i, j))
	path  = np.array(path[::-1], dtype=float)
	gam   = np.interp(np.arange(Q), path[:, 0], path[:, 1]) / (Q - 1)
	return _warp.normalize_warp(gam)


def align_pair(y_template, y, max_step=6, nsub=4, lam=0.0, smooth=0):
	'''
	Elastically align observation *y* to *y_template* (both (Q,) arrays).

	Returns (y_aligned, gamma) where y_aligned = y( gamma(t) ).
	'''
	q1  = srsf(y_template, smooth=smooth)
	q2  = srsf(y, smooth=smooth)
	gam = align_srsf_pair(q1, q2, max_step=max_step, nsub=nsub, lam=lam)
	return _warp.apply_warp(y, gam), gam



# ---------------------------------------------------------------------
# group alignment (Karcher mean template)
# ---------------------------------------------------------------------

def align_group(y, template='karcher', max_iter=20, tol=1e-3, center=True,
	max_step=6, nsub=4, lam=0.0, smooth=0, verbose=False):
	'''
	Elastically register a set of observations to a common template.

	*y*         : (J,Q) array
	*template*  : 'karcher' (iteratively updated mean of aligned SRSFs,
	              initialised at the observation closest to the mean SRSF),
	              'first' (align all to the first observation, no iteration),
	              an integer (align all to that observation, no iteration),
	              or a (Q,) array (a fixed user-supplied template)
	*max_iter*  : maximum number of template updates (karcher only)
	*tol*       : stop when the relative change in the template is below tol
	*center*    : if True, the warps are centered so that their Karcher mean
	              is the identity, and the registered observations and the
	              template are recomputed accordingly (matches the default
	              behaviour of fdasrsf's srsf_align with center=True)
	*max_step*, *nsub*, *lam*, *smooth* : see align_srsf_pair / srsf

	Returns a dict with keys:
	    'y'        : (J,Q) registered observations
	    'warps'    : (J,Q) warps such that y_registered[i] = y[i]( warps[i](t) )
	    'template' : (Q,) the final (function-space) template
	    'q'        : (J,Q) SRSFs of the registered observations
	    'niter'    : number of iterations
	    'cost'     : sum of squared SRSF distances to the template at each iteration
	'''
	y     = np.atleast_2d(np.asarray(y, dtype=float))
	J, Q  = y.shape
	t     = _warp.grid(Q)
	q     = srsf(y, smooth=smooth)
	# --- initial template
	fixed = True
	if isinstance(template, str) and template == 'karcher':
		mq    = q.mean(axis=0)
		d     = ((q - mq)**2).sum(axis=1)
		ind   = int(np.argmin(d))
		mq    = q[ind].copy()
		mf0   = y[ind, 0]
		fixed = False
	elif isinstance(template, str) and template == 'first':
		mq, mf0 = q[0].copy(), y[0, 0]
	elif isinstance(template, (int, np.integer)):
		mq, mf0 = q[int(template)].copy(), y[int(template), 0]
	else:
		tmpl    = np.asarray(template, dtype=float)
		mq, mf0 = srsf(tmpl, smooth=smooth), tmpl[0]
	# --- iterate
	gam   = np.tile(t, (J, 1))
	costs = []
	niter = 0
	for it in range(1 if fixed else max_iter):
		niter += 1
		qn    = np.empty_like(q)
		for i in range(J):
			gam[i] = align_srsf_pair(mq, q[i], max_step=max_step, nsub=nsub, lam=lam)
			qn[i]  = warp_srsf(q[i], gam[i])
		cost  = float(((qn - mq)**2).sum() * (t[1]-t[0]))
		costs.append(cost)
		if verbose:
			print(f'iteration {it+1}: cost = {cost:.6g}')
		if fixed:
			break
		mq_new = qn.mean(axis=0)
		change = np.linalg.norm(mq_new - mq) / max(np.linalg.norm(mq), 1e-12)
		mq     = mq_new
		mf0    = float(np.mean([_warp.apply_warp(y[i], gam[i])[0]  for i in range(J)]))
		if change < tol:
			break
	# --- center the warps
	if center and J > 1:
		gam, gmean = _warp.center_warps(gam)
		mq         = warp_srsf(mq, _warp.invert(gmean))
	yr    = np.array([_warp.apply_warp(y[i], gam[i])  for i in range(J)])
	qn    = np.array([warp_srsf(q[i], gam[i])  for i in range(J)])
	if not fixed:
		mf0 = float(yr[:, 0].mean())
	mf    = srsf_inverse(mq, mf0)
	return dict(y=yr, warps=gam, template=mf, q=qn, niter=niter, cost=np.array(costs))



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
