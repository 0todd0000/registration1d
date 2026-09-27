'''
Tests for reg1d.  Run with:  python -m pytest tests
'''

import numpy as np
import pytest
import reg1d
from reg1d import warp, srsf, dtw, landmark, continuous, linear


Q = 101
t = np.linspace(0, 1, Q)


def _bump(t):
	return np.exp(-((t-0.4)/0.1)**2) - 0.7*np.exp(-((t-0.7)/0.08)**2)


@pytest.fixture
def dorn():
	d  = reg1d.data.Dorn2012()
	return reg1d.register_linear(d.y, n=Q), d.group


# ------------------------------------------------------------------ warps

def test_invert_compose_identity():
	g   = warp.random_warp(1, Q, sigma=0.5, random_state=1)
	gi  = warp.invert(g)
	assert np.allclose(warp.compose(g, gi), t, atol=5e-3)
	assert np.allclose(warp.compose(gi, g), t, atol=5e-3)


def test_random_warp_valid():
	g   = warp.random_warp(5, Q, sigma=1.0, random_state=2)
	assert g.shape == (5, Q)
	assert np.all(warp.is_valid_warp(g))


def test_psi_roundtrip():
	g   = warp.random_warp(1, Q, sigma=0.8, random_state=3)
	assert np.allclose(warp.psi_to_warp(warp.warp_to_psi(g)), g, atol=5e-3)


def test_center_warps():
	g   = warp.random_warp(6, Q, sigma=0.8, random_state=4)
	gc, gm = warp.center_warps(g)
	assert np.abs(warp.karcher_mean_warp(gc) - t).max() < 5e-3


def test_warp_objects():
	g   = warp.random_warp(3, Q, sigma=0.5, random_state=5)
	wl  = warp.Warp1DList(g)
	assert wl.shape == (3, Q)
	y   = _bump(t)
	assert wl.apply(y).shape == (3, Q)
	assert isinstance(wl.mean(), warp.Warp1D)
	assert isinstance(wl[0].inverse(), warp.Warp1D)


# ------------------------------------------------------------------ srsf

def test_srsf_inverse():
	y   = _bump(t)
	q   = srsf.srsf(y)
	assert np.allclose(srsf.srsf_inverse(q, y[0]), y, atol=2e-2)


def test_srsf_pair_recovers_warp():
	y1  = _bump(t)
	g   = warp.random_warp(1, Q, sigma=1.0, random_state=0)
	y2  = warp.apply_warp(y1, g)
	ya, gam = srsf.align_pair(y1, y2)
	assert np.abs(gam - warp.invert(g)).max() < 0.03
	assert np.abs(ya - y1).max() < 0.1


def test_srsf_group_reduces_variance():
	y1  = _bump(t)
	ys  = np.array([warp.apply_warp(y1, warp.random_warp(1, Q, sigma=0.8, random_state=i))  for i in range(6)])
	res = reg1d.register_srsf(ys)
	s0, s1 = res.sse()
	assert s1 < 0.1 * s0
	assert np.all(warp.is_valid_warp(res.warps.asarray()))
	# centered warps: Karcher mean should be the identity
	assert np.abs(res.warps.mean().w - t).max() < 1e-2


def test_srsf_dorn(dorn):
	yi, group = dorn
	yr, wf = reg1d.register_srsf(yi, max_iter=5)
	assert yr.shape == wf.shape == (8, Q)
	assert np.all(warp.is_valid_warp(wf))
	# the propulsive peak (global maximum) should be tightly aligned after registration
	assert np.argmax(yr, axis=1).std() < np.argmax(yi, axis=1).std()


def test_elastic_distances():
	y1  = _bump(t)
	g   = warp.random_warp(1, Q, sigma=1.0, random_state=0)
	y2  = warp.apply_warp(y1, g)
	assert srsf.amplitude_distance(y1, y2) < 0.2
	assert srsf.phase_distance(y1, y2) > 0.05
	assert srsf.phase_distance(y1, y1) < 1e-6


# ------------------------------------------------------------------ dtw

def test_dtw_identity():
	y   = _bump(t)
	path, dist = dtw.dtw_path(y, y)
	assert dist == 0
	assert np.all(path[:, 0] == path[:, 1])


def test_dtw_patterns(dorn):
	yi, group = dorn
	for sp in dtw.STEP_PATTERNS:
		res = reg1d.register_dtw(yi, step_pattern=sp, max_iter=2)
		assert np.all(np.isfinite(res.y))
		assert np.all(warp.is_valid_warp(res.warps.asarray()))


def test_dtw_window(dorn):
	yi, group = dorn
	res = reg1d.register_dtw(yi, window=0.1, max_iter=2)
	assert np.abs(res.warps.displacement()).max() <= 0.1 + 1e-9


# ------------------------------------------------------------------ landmark

def test_landmark_alignment(dorn):
	yi, group = dorn
	res = reg1d.register_landmark(yi, kinds=('zero', 'max'))
	# all maxima now coincide at the target time (to within one grid step)
	imax = np.argmax(res.y, axis=1)
	assert imax.std() <= 1.0
	assert np.all(warp.is_valid_warp(res.warps.asarray()))


def test_landmark_explicit():
	y   = np.array([_bump(t), _bump(warp.random_warp(1, Q, sigma=0.5, random_state=7))])
	lm  = np.array([[np.argmax(yy)/(Q-1)]  for yy in y])
	res = reg1d.register_landmark(y, landmarks=lm, targets=[0.4])
	assert np.allclose(np.argmax(res.y, axis=1), 40, atol=1)


# ------------------------------------------------------------------ continuous

def test_continuous_recovers_smooth_warp():
	y1  = _bump(t)
	g   = warp.random_warp(1, Q, sigma=0.5, n_basis=2, random_state=8)
	y2  = warp.apply_warp(y1, g)
	ya, gam, c = continuous.align_pair(y1, y2, n_basis=4)
	assert np.abs(ya - y1).max() < 0.1


def test_continuous_group(dorn):
	yi, group = dorn
	res = reg1d.register_continuous(yi, n_basis=6, lam=1e-3, max_iter=3)
	s0, s1 = res.sse()
	assert s1 < s0
	assert np.all(warp.is_valid_warp(res.warps.asarray()))


# ------------------------------------------------------------------ linear

def test_resample_ragged():
	y   = [np.sin(np.linspace(0, 3, n))  for n in (50, 80, 120)]
	yi  = reg1d.register_linear(y, n=Q)
	assert yi.shape == (3, Q)
	assert np.allclose(yi[:, 0], 0) and np.allclose(yi[:, -1], np.sin(3))


def test_shift_recovers_delta():
	y1  = _bump(t)
	y2  = np.interp(t + 0.05, t, y1)
	ya, w, d = linear.shift_pair(y1, y2)
	assert abs(d - (-0.05)) < 5e-3


def test_affine_recovers_params():
	y1  = _bump(t)
	y2  = np.interp(0.9*t + 0.03, t, y1)      # y2(t) = y1(0.9 t + 0.03)  ->  y1(s) = y2((s - 0.03)/0.9)
	ya, w, (a, b) = linear.affine_pair(y1, y2)
	assert abs(a - 1/0.9) < 0.02 and abs(b + 0.03/0.9) < 0.01


def test_group_linear(dorn):
	yi, group = dorn
	for f in (reg1d.register_shift, reg1d.register_affine):
		res = f(yi)
		assert res.y.shape == (8, Q)
		s0, s1 = res.sse()
		assert s1 < s0


# ------------------------------------------------------------------ api

def test_result_unpacking(dorn):
	yi, group = dorn
	res = reg1d.register(yi, 'landmark')
	yr, wf = res
	assert yr.shape == wf.shape
	assert isinstance(res.warps, warp.Warp1DList)
