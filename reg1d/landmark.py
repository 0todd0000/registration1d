'''
Landmark registration.

Implemented directly from the description in

    Kneip A, Gasser T (1992). Statistical tools to analyze data
        representing a sample of curves. Annals of Statistics 20: 1266-1305.
    Ramsay JO, Silverman BW (2005). Functional Data Analysis, 2nd ed.
        Springer.  (Section 7.3, "Landmark registration")

Each observation is annotated with K landmark times (for example the
times of a local minimum, a zero crossing and a local maximum). The warp
for observation i is the monotone interpolant through the points
(target_k, landmark_ik), k=1..K, together with the fixed end points
(0,0) and (1,1), where the target times are (by default) the mean
landmark times across observations. Applying the warp moves every
landmark of observation i to its target time.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import interpolate, signal
from . import warp as _warp



def landmark_warp(landmarks, targets, Q, kind='pchip'):
	'''
	Warp mapping the target times onto the observed landmark times.

	*landmarks* : (K,) observed landmark times in [0,1]
	*targets*   : (K,) target times in [0,1]
	*kind*      : 'linear' (piecewise linear) or 'pchip' (monotone cubic)

	Returns a (Q,) warp gamma with gamma(target_k) = landmark_k.
	'''
	x   = np.concatenate([[0.0], np.asarray(targets, dtype=float), [1.0]])
	y   = np.concatenate([[0.0], np.asarray(landmarks, dtype=float), [1.0]])
	o   = np.argsort(x)
	x, y = x[o], y[o]
	if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
		raise ValueError('landmarks and targets must both be strictly increasing within (0,1)')
	t   = _warp.grid(Q)
	if kind == 'linear':
		g = np.interp(t, x, y)
	elif kind == 'pchip':
		g = interpolate.PchipInterpolator(x, y)(t)
	else:
		raise ValueError("kind must be 'linear' or 'pchip'")
	return _warp.normalize_warp(g)


def detect_landmarks(y, kinds=('min', 'zero', 'max')):
	'''
	Simple automatic landmark detection for a (Q,) or (J,Q) array.

	*kinds* is a sequence drawn from:
	    'min'   time of the global minimum
	    'max'   time of the global maximum
	    'zero'  time of the zero crossing (linearly interpolated) between the
	            global minimum and the global maximum, taking the crossing
	            adjacent to the maximum if there are several; if none
	            exists, the time of the smallest absolute value between them

	The detected landmarks are returned in increasing time order per
	observation, as a (K,) or (J,K) array. For data with more complex
	structure, supply landmarks explicitly (e.g. from find_peaks) instead.
	'''
	y  = np.asarray(y, dtype=float)
	if y.ndim == 2:
		return np.array([detect_landmarks(yy, kinds)  for yy in y])
	Q  = y.size
	t  = _warp.grid(Q)
	imin, imax = int(np.argmin(y)), int(np.argmax(y))
	lm = {}
	if 'min' in kinds:
		lm['min'] = t[imin]
	if 'max' in kinds:
		lm['max'] = t[imax]
	if 'zero' in kinds:
		a, b = sorted([imin, imax])
		seg  = y[a:b+1]
		s    = np.sign(seg)
		cr   = np.where(np.diff(s) != 0)[0]
		if cr.size > 0:
			# the crossing adjacent to the global maximum (the last one if the
			# maximum follows the minimum, otherwise the first)
			k    = a + (cr[-1] if imax > imin else cr[0])
			# linear interpolation of the crossing time
			y0, y1 = y[k], y[k+1]
			frac   = y0 / (y0 - y1) if (y0 != y1) else 0.0
			lm['zero'] = t[k] + frac * (t[1] - t[0])
		else:
			lm['zero'] = t[a + int(np.argmin(np.abs(seg)))]
	return np.sort(np.array([lm[k]  for k in kinds]))


def peaks_as_landmarks(y, n_peaks, **kwargs):
	'''
	Times of the *n_peaks* most prominent local maxima of a (Q,) array,
	in increasing time order (thin wrapper around scipy.signal.find_peaks;
	keyword arguments are passed through).
	'''
	y  = np.asarray(y, dtype=float)
	t  = _warp.grid(y.size)
	ind, props = signal.find_peaks(y, prominence=kwargs.pop('prominence', 0), **kwargs)
	prom = props['prominences']
	top  = ind[np.argsort(prom)[::-1][:n_peaks]]
	return np.sort(t[top])


def align_group(y, landmarks=None, targets='mean', kind='pchip', kinds=('min', 'zero', 'max')):
	'''
	Landmark-register a set of observations.

	*y*          : (J,Q) array
	*landmarks*  : (J,K) array of landmark times in (0,1); if None they are
	               detected automatically with detect_landmarks(y, kinds)
	*targets*    : 'mean' (mean landmark times across observations),
	               'median', or a (K,) array of target times
	*kind*       : interpolation of the warp between landmarks ('pchip' or 'linear')

	Returns a dict with keys 'y', 'warps', 'landmarks', 'targets'.
	'''
	y   = np.atleast_2d(np.asarray(y, dtype=float))
	J, Q = y.shape
	if landmarks is None:
		landmarks = detect_landmarks(y, kinds)
	landmarks = np.atleast_2d(np.asarray(landmarks, dtype=float))
	if isinstance(targets, str):
		targets = landmarks.mean(axis=0) if targets == 'mean' else np.median(landmarks, axis=0)
	targets = np.asarray(targets, dtype=float)
	gam = np.array([landmark_warp(lm, targets, Q, kind)  for lm in landmarks])
	yr  = np.array([_warp.apply_warp(y[i], gam[i])  for i in range(J)])
	return dict(y=yr, warps=gam, landmarks=landmarks, targets=targets)
