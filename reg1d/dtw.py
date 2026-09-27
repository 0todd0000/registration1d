'''
Dynamic time warping (DTW) registration.

Implemented directly from the classical algorithm:

    Sakoe H, Chiba S (1978). Dynamic programming algorithm optimization
        for spoken word recognition. IEEE Transactions on Acoustics,
        Speech, and Signal Processing 26: 43-49.
    Giorgino T (2009). Computing and visualizing dynamic time warping
        alignments in R: the dtw package. Journal of Statistical Software
        31(7): 1-24.   (step-pattern nomenclature)

DTW finds a monotone alignment path between two sequences x (reference)
and y (query) that minimises the accumulated local distance |x_i - y_j|^p
subject to a step pattern. Unlike the SRSF approach, DTW compares raw
amplitudes (not slopes), the path may contain horizontal and vertical
runs (many-to-one matches), and there is no penalty for extreme warping
unless a window or a slope-constrained step pattern is used. The path is
converted to a warping function gamma(t) so that y( gamma(t) ) ~ x(t).

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from . import warp as _warp



# step patterns:  list of (di, dj, weight)   (weights follow Giorgino 2009)
STEP_PATTERNS = {
	'symmetric1' : [(1, 1, 1.0), (1, 0, 1.0), (0, 1, 1.0)],
	'symmetric2' : [(1, 1, 2.0), (1, 0, 1.0), (0, 1, 1.0)],
	'asymmetric' : [(1, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0)],
	'strict'     : [(1, 1, 2.0), (1, 2, 3.0), (2, 1, 3.0)],   # no flat segments (slopes 1/2..2)
}


def local_cost(x, y, p=2):
	'''Local cost matrix  C[i,j] = |x_i - y_j|^p.'''
	x  = np.asarray(x, dtype=float)
	y  = np.asarray(y, dtype=float)
	return np.abs(x[:, None] - y[None, :])**p


def dtw_path(x, y, step_pattern='symmetric2', window=None, p=2):
	'''
	Dynamic time warping alignment of query *y* to reference *x*.

	*step_pattern* : key of STEP_PATTERNS
	*window*       : Sakoe-Chiba band half-width as a fraction of the domain
	                 (None = unconstrained)
	*p*            : exponent of the local distance

	Returns (path, distance) where path is an (L,2) integer array of
	(i,j) index pairs from (0,0) to (Q1-1, Q2-1), and distance is the
	accumulated cost at the end node.
	'''
	x     = np.asarray(x, dtype=float)
	y     = np.asarray(y, dtype=float)
	n, m  = x.size, y.size
	C     = local_cost(x, y, p)
	steps = STEP_PATTERNS[step_pattern]
	D     = np.full((n, m), np.inf)
	P     = np.full((n, m), -1, dtype=int)
	D[0, 0] = C[0, 0]
	if window is not None:
		ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')
		allowed = np.abs(ii / max(n-1, 1) - jj / max(m-1, 1)) <= window
	else:
		allowed = np.ones((n, m), dtype=bool)
	# strictly row-major recursion; steps with di==0 depend on the same row,
	# so within a row the columns are processed in increasing order
	for i in range(n):
		for j in range(m):
			if (i, j) == (0, 0) or not allowed[i, j]:
				continue
			best, bk = np.inf, -1
			for k, (di, dj, w) in enumerate(steps):
				i0, j0 = i - di, j - dj
				if i0 < 0 or j0 < 0:
					continue
				c = D[i0, j0] + w*C[i, j]
				if c < best:
					best, bk = c, k
			D[i, j] = best
			P[i, j] = bk
	if not np.isfinite(D[n-1, m-1]):
		raise RuntimeError('no admissible DTW path; widen the window or change the step pattern')
	i, j  = n-1, m-1
	path  = [(i, j)]
	while (i, j) != (0, 0):
		di, dj, _ = steps[P[i, j]]
		i, j = i - di, j - dj
		path.append((i, j))
	return np.array(path[::-1], dtype=int), float(D[n-1, m-1])


def path_to_warp(path, Q1, Q2):
	'''
	Convert a DTW path to a warping function on the reference grid.

	For reference index i the matched query indices j are averaged (a
	vertical run in the path maps one reference point to several query
	points), yielding a non-decreasing function j(i); it is rescaled to
	[0,1] on both axes. Horizontal runs produce flat segments, so the
	result is monotone but not necessarily strictly increasing.
	'''
	i, j  = path[:, 0], path[:, 1]
	ks    = np.unique(i)
	jm    = np.array([j[i == k].mean()  for k in ks])
	# rows skipped by multi-step patterns (e.g. the (2,1) step) are filled by interpolation
	jm    = np.interp(np.arange(Q1), ks, jm)
	jm[0], jm[-1] = 0, Q2 - 1                       # vertical runs at the ends keep the end points fixed
	gam   = jm / (Q2 - 1)
	return _warp.normalize_warp(gam)


def align_pair(y_template, y, step_pattern='symmetric2', window=None, p=2):
	'''
	Align observation *y* to *y_template* using DTW.

	Returns (y_aligned, gamma, distance) with y_aligned(t) = y( gamma(t) ).
	'''
	path, dist = dtw_path(y_template, y, step_pattern=step_pattern, window=window, p=p)
	gam        = path_to_warp(path, y_template.size, y.size)
	return _warp.apply_warp(y, gam), gam, dist


def align_group(y, template='mean', max_iter=10, tol=1e-4, step_pattern='symmetric2',
	window=None, p=2, verbose=False):
	'''
	Register a set of observations with DTW, iteratively refining the
	template as the cross-sectional mean of the aligned observations
	(the simplest form of DTW barycenter averaging).

	*y*         : (J,Q) array
	*template*  : 'mean' (start from the cross-sectional mean), 'medoid'
	              (start from the observation with the smallest total DTW
	              distance to the others), an integer index, or a (Q,) array
	*max_iter*  : number of template refinements ('mean'/'medoid' only)

	Returns a dict with keys 'y', 'warps', 'template', 'distance', 'niter'.
	'''
	y     = np.atleast_2d(np.asarray(y, dtype=float))
	J, Q  = y.shape
	fixed = False
	if isinstance(template, str) and template == 'mean':
		tmpl = y.mean(axis=0)
	elif isinstance(template, str) and template == 'medoid':
		Dm   = np.zeros((J, J))
		for a in range(J):
			for b in range(a+1, J):
				Dm[a, b] = Dm[b, a] = dtw_path(y[a], y[b], step_pattern, window, p)[1]
		tmpl = y[int(np.argmin(Dm.sum(axis=1)))].copy()
	elif isinstance(template, (int, np.integer)):
		tmpl, fixed = y[int(template)].copy(), True
	else:
		tmpl, fixed = np.asarray(template, dtype=float), True
	gam   = np.zeros((J, Q))
	dist  = np.zeros(J)
	niter = 0
	for it in range(1 if fixed else max_iter):
		niter += 1
		yr = np.empty_like(y)
		for i in range(J):
			yr[i], gam[i], dist[i] = align_pair(tmpl, y[i], step_pattern, window, p)
		if verbose:
			print(f'iteration {it+1}: total distance = {dist.sum():.6g}')
		if fixed:
			break
		new    = yr.mean(axis=0)
		change = np.linalg.norm(new - tmpl) / max(np.linalg.norm(tmpl), 1e-12)
		tmpl   = new
		if change < tol:
			break
	yr = np.array([_warp.apply_warp(y[i], gam[i])  for i in range(J)])
	return dict(y=yr, warps=gam, template=tmpl, distance=dist, niter=niter)
