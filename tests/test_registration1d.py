'''
Tests for reg1d.  Run with:  python -m pytest tests
'''

import numpy as np
import pytest
import registration1d as reg1d
from registration1d import warp, srsf, dtw, landmark, continuous, linear


Q = 101
t = np.linspace(0, 1, Q)


def _bump(t):
    return np.exp(-((t-0.4)/0.1)**2) - 0.7*np.exp(-((t-0.7)/0.08)**2)


@pytest.fixture
def dorn():
    d  = reg1d.data.Dorn2012()
    return reg1d.register_linear(d.y, n=Q).y, d.group


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
    res = reg1d.register_linear(y, n=Q)
    yi  = res.y
    assert res.islinear and yi.shape == (3, Q)
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


# ------------------------------------------------------------------ result objects and grids

def test_result_classes(dorn):
    yi, group = dorn
    assert reg1d.register_linear(yi).islinear
    assert reg1d.register_shift(yi).islinear
    assert reg1d.register_affine(yi).islinear
    r = reg1d.register_srsf(yi, max_iter=1)
    assert isinstance(r, reg1d.NonlinearRegistrationResult) and not r.islinear and r.isnonlinear
    assert isinstance(r, reg1d.RegistrationResult)


def test_explicit_time_grid(dorn):
    yi, group = dorn
    r0 = reg1d.register_srsf(yi, max_iter=2)
    r1 = reg1d.register_srsf(yi, t=np.linspace(0, 100, Q), max_iter=2)      # uniform, different units
    assert np.allclose(r0.warps.asarray(), r1.warps.asarray())
    assert r1.t[-1] == 100 and r1.warps_t.max() == 100
    assert np.allclose(r1.displacement_fields_t, 100 * r0.displacement_fields)
    tn = np.linspace(0, 1, Q)**1.5                                            # non-uniform
    r2 = reg1d.register_landmark(yi, t=tn, kinds=('zero', 'max'))
    assert r2.y.shape == (8, Q) and np.allclose(r2.t, np.linspace(0, 1, Q))


def test_apply_unapply(dorn):
    yi, group = dorn
    r  = reg1d.register_srsf(yi, max_iter=3)
    z  = np.gradient(yi, axis=1)
    assert r.apply(z).shape == z.shape
    assert np.allclose(r.apply(yi), r.y)
    back = r.unapply(r.y)
    assert np.abs(back - yi).max() < 0.05 * np.abs(yi).max()      # interpolation error only
    assert r.unapply(r.template).shape == (8, Q)
    assert r.inverse_warps.shape == (8, Q)


# ------------------------------------------------------------------ srsf options

def test_srsf_options(dorn):
    yi, group = dorn
    base = reg1d.register_srsf(yi, max_iter=2)
    for kw in [dict(method='median'), dict(smooth='spline'), dict(smooth=2), dict(band=0.15),
               dict(refine=True), dict(lam=0.5), dict(parallel=2)]:
        r = reg1d.register_srsf(yi, max_iter=2, **kw)
        assert r.y.shape == (8, Q) and np.all(warp.is_valid_warp(r.warps.asarray())), kw
    rb = reg1d.register_srsf(yi, max_iter=2, band=0.05)
    assert np.abs(rb.warps.displacement()).max() <= 0.05 + 2.0/Q
    rp = reg1d.register_srsf(yi, max_iter=2, parallel=2)
    assert np.allclose(rp.warps.asarray(), base.warps.asarray())


def test_srsf_multivariate():
    y1  = np.column_stack([_bump(t), np.sin(2*np.pi*t)])
    g   = warp.random_warp(1, Q, sigma=0.4, random_state=11)
    y2  = np.column_stack([warp.apply_warp(y1[:, 0], g), warp.apply_warp(y1[:, 1], g)])
    ya, gam = srsf.align_pair(y1, y2)
    assert np.abs(gam - warp.invert(g)).max() < 0.03
    Y   = np.stack([y1, y2, y2], axis=0)                                      # (3,Q,2)
    r   = reg1d.register_srsf(Y, max_iter=3)
    assert r.y.shape == (3, Q, 2) and r.template.shape == (Q, 2)
    assert r.apply(Y).shape == (3, Q, 2)


# ------------------------------------------------------------------ dtw options

def test_dtw_derivative_and_smoothing(dorn):
    yi, group = dorn
    r  = reg1d.register_dtw(yi, derivative=True, step_pattern='strict', smooth=0.03, max_iter=2)
    g  = r.warps.asarray()
    assert np.all(warp.is_valid_warp(g))
    assert np.gradient(g, axis=1).min() > 0                                   # strictly increasing
    rd = reg1d.register_dtw(yi, template='dba', max_iter=2)
    assert rd.y.shape == (8, Q)


def test_smooth_warp_preserves_validity():
    g  = warp.random_warp(3, Q, sigma=0.5, random_state=12)
    gs = warp.smooth_warp(g, 0.05)
    assert np.all(warp.is_valid_warp(gs))
    assert np.abs(gs - g).max() < 0.1


# ------------------------------------------------------------------ linear options

def test_affine_cover_zero_fill(dorn):
    yi, group = dorn
    r = reg1d.register_affine(yi, cover=True, fill_value='zero')
    a, b = r.info['params'][:, 0], r.info['params'][:, 1]
    assert np.all(b <= 1e-9) and np.all(a + b >= 1 - 1e-9)
    assert np.allclose(r.y[:, 0], yi[:, 0]) or np.abs(r.y[:, 0]).max() <= np.abs(yi[:, 0]).max() + 1e-9
    assert np.abs(r.y[:, -1]).max() <= np.abs(yi[:, -1]).max() + 1e-9


def test_affine_respects_search_box():
    # featureless smooth noise: without bounds on the local refinement the
    # scale factor drifts to arbitrarily large values (the curve is squeezed
    # to a point and the zero fill trivially minimises the residual)
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(3)
    y   = gaussian_filter1d(rng.standard_normal((12, 101)), 6, axis=1)
    for cover in (False, True):
        r = reg1d.register_affine(y, cover=cover, fill_value='zero', scale_range=(0.7, 1.4), max_shift=0.2,
                                  center=False)   # centering rescales the parameters by their mean
        a, b = r.info['params'][:, 0], r.info['params'][:, 1]
        assert np.all(a <= 1.4 + 1e-6) and np.all(a >= 0.7 - 1e-6)
        assert np.all(np.abs(b) <= 0.2 + 1e-6)


def test_fill_values():
    y = _bump(t)
    for fv in ('edge', 'zero', 'extrapolate', 0.5):
        out = linear._eval(y, t + 0.1, fv)
        assert out.shape == y.shape and np.all(np.isfinite(out))


# ------------------------------------------------------------------ additional methods

def test_bayes_pair_and_group():
    from registration1d import bayes
    y1  = _bump(t)
    g   = warp.random_warp(1, Q, sigma=0.3, n_basis=2, random_state=5)
    y2  = warp.apply_warp(y1, g) + 0.02 * np.random.default_rng(3).standard_normal(Q)
    c0  = bayes.initial_coef(srsf.align_pair(y1, y2)[1], 8)
    s   = bayes.sample_pair(y1, y2, n_samples=300, burn=500, c0=c0, random_state=1)
    sm  = bayes.summarize(s['warps'])
    assert s['warps'].shape == (300, Q)
    assert 0.05 < s['accept'] < 0.6
    assert np.abs(sm['warp_mean'] - warp.invert(g)).mean() < 0.02
    Y   = np.array([y1, y2])
    r   = reg1d.register_bayes(Y, n_samples=100, burn=100, random_state=2, max_iter=2)
    assert r.info['samples'].shape == (2, 100, Q) and r.info['disp_ci'].shape == (2, 2, Q)


def test_pairwise_and_sim(dorn):
    yi, group = dorn
    r = reg1d.register_pairwise(yi[:4])
    assert r.info['pairwise'].shape == (4, 4, Q) and np.all(warp.is_valid_warp(r.warps.asarray()))
    r = reg1d.register_sim(yi, n_basis=4, max_iter=2)
    assert r.info['amplitude'].shape == (8, 2) and np.all(warp.is_valid_warp(r.warps.asarray()))


# ------------------------------------------------------------------ stats helpers

def test_stats_permutation():
    from registration1d import stats
    rng = np.random.default_rng(0)
    yA  = rng.standard_normal((10, Q))
    yB  = rng.standard_normal((10, Q)) + np.where((t > 0.4) & (t < 0.6), 3.0, 0.0)
    res = stats.permutation_ttest2(yA, yB, n_perm=200, random_state=0)
    assert res['p'] < 0.05 and len(res['clusters']) >= 1
    covered = np.zeros(Q, dtype=bool)
    for lo, hi in res['clusters']:
        covered[lo:hi+1] = True
    assert covered[45:56].mean() > 0.8 and covered[:30].mean() < 0.2


# ------------------------------------------------------------------ real-time registration

def test_realtime_srsf_dorn():
    d  = reg1d.data.Dorn2012()
    r  = reg1d.register_srsf(list(d.y), t='fs=1000', max_iter=3)
    assert r.info['realtime'] and r.y.shape == (8, 101)
    assert np.allclose(r.info['warps_realtime'][:, -1], r.info['durations'])
    assert np.all(warp.is_valid_warp(r.warps.asarray()))
    assert np.argmax(r.y, axis=1).std() < np.argmax(r.y0, axis=1).std()
    z  = r.apply(list(d.y))                       # same warps applied to a ragged variable
    assert np.allclose(z, r.y)
    rn = reg1d.register_srsf(reg1d.register_linear(d.y).y, max_iter=3)
    assert np.abs(r.warps.asarray() - rn.warps.asarray()).max() < 0.1     # similar, not identical


def test_realtime_pair_recovers_time_scaling():
    # the same physical event sampled with two different durations should align exactly
    t1 = np.linspace(0, 1, 201); t2 = np.linspace(0, 1, 101)
    f  = lambda s: np.exp(-((s-0.4)/0.1)**2)
    q1 = srsf.srsf(f(t1)) / np.sqrt(1.0); q2 = srsf.srsf(f(t2)) / np.sqrt(1.0)
    jidx = srsf.align_srsf_pair(q1, q2, dt1=t1[1]-t1[0], dt2=t2[1]-t2[0], return_index=True)
    assert np.abs(jidx / 100 - t1).max() < 0.02


def test_realtime_dtw_and_landmark():
    d  = reg1d.data.Dorn2012()
    r  = reg1d.register_dtw(list(d.y), t=0.001, derivative=True, smooth=0.03, max_iter=2)
    assert r.y.shape == (8, 101) and np.all(warp.is_valid_warp(r.warps.asarray()))
    r  = reg1d.register_landmark(list(d.y), t=0.001, kinds=('zero', 'max'))
    assert r.y.shape == (8, 101) and np.argmax(r.y, axis=1).std() <= 1.0
    with pytest.raises(RuntimeError):
        reg1d.register_dtw(list(d.y), t=0.001, step_pattern='strict', max_iter=1)


# ------------------------------------------------------------------ centering options and auto lam

@pytest.mark.parametrize('method,kw', [('srsf', dict(max_iter=2)), ('dtw', dict(step_pattern='strict', smooth=0.03, max_iter=2)),
                                       ('continuous', dict(max_iter=2)), ('sim', dict(max_iter=2)), ('pairwise', {})])
def test_centering_options(dorn, method, kw):
    yi, group = dorn
    for c in ('karcher', 'pointwise', 'anchor', 'none'):
        r = reg1d.register(yi, method, center=c, anchor='max', **kw)
        g = r.warps.asarray()
        assert r.info['center'] == c
        if c == 'karcher':
            assert np.abs(warp.karcher_mean_warp(g) - t).max() < 5e-3
        if c == 'pointwise':
            assert np.abs(g.mean(axis=0) - t).max() < 1e-9
        if c == 'anchor':
            assert abs(np.argmax(r.y, axis=1).mean() - np.argmax(yi, axis=1).mean()) <= 1.0
    r_true  = reg1d.register(yi, method, center=True, **kw)
    r_false = reg1d.register(yi, method, center=False, **kw)
    assert r_true.info['center'] == ('pointwise' if method == 'dtw' else 'karcher')
    assert r_false.info['center'] == 'none'


def test_dtw_default_pointwise_and_landmark_default_none(dorn):
    yi, group = dorn
    assert reg1d.register_dtw(yi, max_iter=1).info['center'] == 'pointwise'
    assert reg1d.register_landmark(yi, kinds=('zero', 'max')).info['center'] == 'none'
    assert reg1d.register_bayes(yi[:2], n_samples=20, burn=20, random_state=0, max_iter=1).info['center'] == 'karcher'


def test_auto_lam():
    A   = reg1d.data.SimulatedA()
    r   = reg1d.register_srsf(A.y, max_iter=3)
    assert 40 < r.info['lam'] < 60                         # median total variation of the observations
    assert np.isclose(r.info['lam'], srsf.auto_lam(srsf.srsf(A.y)))
    r0  = reg1d.register_srsf(A.y, max_iter=3, lam=0)
    assert r0.info['lam'] == 0
    # the penalty suppresses the noise-driven warps in the flat tails of dataset A
    assert np.abs(r.displacement_fields[:, :15]).max() < np.abs(r0.displacement_fields[:, :15]).max()


# ------------------------------------------------------------------ pyqtgraph backend (optional)

def test_plotqt_backend(dorn, tmp_path):
    pg = pytest.importorskip('pyqtgraph')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from registration1d import plotqt
    yi, group = dorn
    r   = reg1d.register_srsf(yi, max_iter=1)
    win = r.plot(group=group, backend='pyqtgraph')
    assert len(win.plots) == 3
    img = plotqt.to_image(win, tmp_path / 'reg.png')
    assert img.width() > 0 and (tmp_path / 'reg.png').exists()
    w   = plotqt.plot_displacement_fields(r.warps.asarray(), group=group)
    assert plotqt.to_image(w, size=(300, 200)).height() == 200
    # drawing into a caller-supplied PlotItem returns that same object
    glw = pg.GraphicsLayoutWidget(); item = glw.addPlot()
    assert plotqt.plot_curves(yi, group=group, plot=item) is item
