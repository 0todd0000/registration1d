'''
Public registration interface.

All group-registration functions accept a (J,Q) array of observations
sampled on a common grid and return a RegistrationResult, which can be
unpacked as  (yr, wf) = register_xxx(y)  for compatibility with nlreg1d.

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



class RegistrationResult(object):
	'''
	Container for the output of a group registration.

	Attributes:
	    y        : (J,Q) registered observations
	    y0       : (J,Q) input observations
	    warps    : Warp1DList  (y[i] = y0[i]( warps[i](t) ))
	    template : (Q,) template to which the observations were aligned (or None)
	    method   : name of the method
	    info     : dict of method-specific extras (iterations, costs, landmarks, ...)

	Tuple unpacking returns (y, warps-as-array):

	>>> yr, wf = reg1d.register_srsf( y )
	'''
	def __init__(self, y, y0, warps, template=None, method=None, info=None):
		self.y        = np.asarray(y, dtype=float)
		self.y0       = np.asarray(y0, dtype=float)
		self.warps    = warps if isinstance(warps, _warp.Warp1DList) else _warp.Warp1DList(warps)
		self.template = None if template is None else np.asarray(template, dtype=float)
		self.method   = method
		self.info     = {} if info is None else info

	def __iter__(self):
		yield self.y
		yield self.warps.asarray()

	def __repr__(self):
		s  = f'RegistrationResult ({self.method})\n'
		s += f'    y        : {self.y.shape}\n'
		s += f'    warps    : {self.warps.shape}\n'
		s += f'    template : {None if self.template is None else self.template.shape}\n'
		s += f'    info     : {list(self.info.keys())}\n'
		return s

	@property
	def J(self):
		return self.y.shape[0]

	@property
	def Q(self):
		return self.y.shape[1]

	@property
	def displacement_fields(self):
		return self.warps.displacement_field()

	def plot(self, group=None, **kwargs):
		from . import plot
		return plot.plot_registration(self, group=group, **kwargs)

	def sse(self):
		'''Total squared deviation from the cross-sectional mean, before and after.'''
		f = lambda a: float(((a - a.mean(axis=0))**2).sum())
		return f(self.y0), f(self.y)



# ---------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------

def register_linear(y, n=101, kind='linear'):
	'''
	Linearly register (interpolate) one observation, or a sequence of
	observations of arbitrary lengths, to *n* equally spaced points.

	>>> yi = reg1d.register_linear( y, n=101 )     # y: (Q,) array  ->  (n,)
	>>> yi = reg1d.register_linear( ylist, n=101 ) # list of arrays ->  (J,n)
	'''
	return _linear.resample(y, n=n, kind=kind)


def register_shift(y, **kwargs):
	'''
	Shift registration:  gamma_i(t) = t + delta_i  (least-squares, Procrustes iteration).
	Keyword arguments: template, max_iter, tol, center, fill_value, max_shift, verbose.
	'''
	r = _linear.align_group(y, method='shift', **kwargs)
	return RegistrationResult(r['y'], y, _AffineWarps(r['warps']), r['template'], 'shift',
		dict(shift=r['params'], niter=r['niter']))


def register_affine(y, **kwargs):
	'''
	Affine registration:  gamma_i(t) = a_i t + b_i  (least-squares, Procrustes iteration).
	Keyword arguments: template, max_iter, tol, center, fill_value, max_shift, scale_range, verbose.
	'''
	r = _linear.align_group(y, method='affine', **kwargs)
	return RegistrationResult(r['y'], y, _AffineWarps(r['warps']), r['template'], 'affine',
		dict(params=r['params'], niter=r['niter']))


class _AffineWarps(_warp.Warp1DList):
	'''Warp1DList whose elements are not normalised to [0,1] (affine maps may leave the unit interval).'''
	def __init__(self, w):
		list.__init__(self, [_AffineWarp(ww)  for ww in np.atleast_2d(w)])

class _AffineWarp(_warp.Warp1D):
	def __init__(self, w):
		self.w = np.asarray(w, dtype=float)



# ---------------------------------------------------------------------
# nonlinear
# ---------------------------------------------------------------------

def register_srsf(y, template='karcher', max_iter=20, tol=1e-3, center=True, max_step=6,
	nsub=4, lam=0.0, smooth=0, verbose=False):
	'''
	Elastic (SRSF / Fisher-Rao) registration by dynamic programming with
	an iteratively updated Karcher-mean template.

	>>> yr, wf = reg1d.register_srsf( y, max_iter=5 )

	*y*         : (J,Q) array
	*template*  : 'karcher' | 'first' | int | (Q,) array
	*max_iter*  : maximum number of template updates
	*center*    : center the warps (Karcher mean of the warps = identity)
	*max_step*  : slope set for dynamic programming (6 -> local slopes 1/6 ... 6)
	*nsub*      : sub-samples per grid step in the segment-cost integrals
	*lam*       : penalty on departure from the identity warp (0 = none)
	*smooth*    : moving-average passes before differentiation
	'''
	r = _srsf.align_group(y, template=template, max_iter=max_iter, tol=tol, center=center,
		max_step=max_step, nsub=nsub, lam=lam, smooth=smooth, verbose=verbose)
	return RegistrationResult(r['y'], y, r['warps'], r['template'], 'srsf',
		dict(niter=r['niter'], cost=r['cost'], q=r['q']))


def register_dtw(y, template='mean', max_iter=10, step_pattern='symmetric2', window=None,
	p=2, verbose=False):
	'''
	Dynamic time warping registration with an iteratively refined mean template.

	*template*     : 'mean' | 'medoid' | int | (Q,) array
	*step_pattern* : 'symmetric1' | 'symmetric2' | 'asymmetric' | 'strict'
	*window*       : Sakoe-Chiba band half-width (fraction of the domain), or None
	*p*            : exponent of the local distance |x_i - y_j|^p
	'''
	r = _dtw.align_group(y, template=template, max_iter=max_iter, step_pattern=step_pattern,
		window=window, p=p, verbose=verbose)
	return RegistrationResult(r['y'], y, r['warps'], r['template'], 'dtw',
		dict(niter=r['niter'], distance=r['distance']))


def register_landmark(y, landmarks=None, targets='mean', kind='pchip', kinds=('min', 'zero', 'max')):
	'''
	Landmark registration.

	*landmarks* : (J,K) landmark times in (0,1), or None for automatic
	              detection of the landmark kinds in *kinds*
	*targets*   : 'mean' | 'median' | (K,) array
	*kind*      : warp interpolation between landmarks: 'pchip' | 'linear'
	'''
	r = _landmark.align_group(y, landmarks=landmarks, targets=targets, kind=kind, kinds=kinds)
	return RegistrationResult(r['y'], y, r['warps'], None, 'landmark',
		dict(landmarks=r['landmarks'], targets=r['targets']))


def register_continuous(y, template='mean', n_basis=4, lam=1e-2, max_iter=5, center=True, verbose=False):
	'''
	Continuous (parametric, penalised least-squares) registration with
	smooth monotone warps gamma = int exp(W), W in a sine basis.

	*n_basis* : number of basis functions (degrees of freedom of each warp)
	*lam*     : roughness penalty on W (dimensionless; 0 = none)
	'''
	r = _continuous.align_group(y, template=template, n_basis=n_basis, lam=lam, max_iter=max_iter,
		center=center, verbose=verbose)
	return RegistrationResult(r['y'], y, r['warps'], r['template'], 'continuous',
		dict(niter=r['niter'], coef=r['coef']))


METHODS = {
	'shift'      : register_shift,
	'affine'     : register_affine,
	'srsf'       : register_srsf,
	'dtw'        : register_dtw,
	'landmark'   : register_landmark,
	'continuous' : register_continuous,
}


def register(y, method='srsf', **kwargs):
	'''Dispatch to one of the register_* functions by name (see METHODS).'''
	return METHODS[method](y, **kwargs)
