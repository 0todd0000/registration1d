'''
Generate, execute and render the demonstration notebooks.

    python make_notebooks.py          # writes *.ipynb and html/*.html
'''

import os, sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter


HERE = os.path.dirname(os.path.abspath(__file__))


def md(s):
	return nbf.v4.new_markdown_cell(s.strip('\n'))

def code(s):
	return nbf.v4.new_code_cell(s.strip('\n'))


SETUP = '''
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # so that this notebook finds reg1d without installation
import numpy as np
from matplotlib import pyplot as plt
import reg1d
print('reg1d version:', reg1d.__version__)
'''

PLOT_DORN = '''
dataset = reg1d.data.Dorn2012()
speed   = dataset.group     # running speed code (0-3)
y       = dataset.y         # object array: 8 observations, 185-383 frames each
print(dataset)

def plot_Dorn2012(y, xlabel='Frame number', ylabel='Anterioposterior GRF (N)', title=None, ax=None):
    ax = plt.axes() if ax is None else ax
    reg1d.plot.plot_curves(y, group=speed, ax=ax, colors=['k','b','g','r'],
                           x=None if xlabel.startswith('Frame') else 'percent',
                           labels=[f'Speed = {i}' for i in range(4)])
    ax.axhline(0, color='k', ls=':')
    ax.set_xlabel(xlabel, size=12)
    ax.set_ylabel(ylabel, size=12)
    if title is not None:
        ax.set_title(title, size=14)
    return ax

plt.figure(figsize=(8,5))
plot_Dorn2012(y)
plt.show()
'''


# ---------------------------------------------------------------------------
# notebook 1
# ---------------------------------------------------------------------------

nb1 = [
md('''
# 1 — Registration with reg1d

This notebook reproduces the workflow of `nlreg1d`'s notebook *3-Registration* using `reg1d`:
linear registration by interpolation to a common number of frames, followed by nonlinear
(elastic, SRSF-based) registration.

`reg1d` differs from `nlreg1d` in one important respect: it depends only on **numpy**, **scipy**
and **matplotlib**. The SRSF registration is implemented directly from the mathematics of the
square-root slope framework (Srivastava et al. 2011) rather than by wrapping `fdasrsf`, so
`fdasrsf` and `scikit-fda` do not need to be installed.

The dataset is the `Dorn2012` anteroposterior ground reaction force (GRF) dataset: running at four
speeds (coded 0–3 in increasing order), two trials per speed.
'''),
code(SETUP),
md('### Load data'),
code(PLOT_DORN),
md('''
### Linear registration

The number of frames decreases with speed because stance time decreases. `register_linear`
interpolates each observation to `n` equally spaced points. It accepts a single observation
(as in `nlreg1d`) or a sequence of observations of different lengths.
'''),
code('''
yi = reg1d.register_linear(y, n=101)      # (8,101) array
print(yi.shape)

plt.figure(figsize=(8,5))
plot_Dorn2012(yi, xlabel='Time (%)')
plt.show()
'''),
md('''
### Nonlinear registration

Local extrema are not aligned in time after linear registration. `register_srsf` aligns the
observations elastically: each observation's square-root slope function (SRSF) is aligned to
an iteratively updated Karcher-mean template by dynamic programming, and the resulting warps are
centred so that their Karcher mean is the identity (the same procedure as `fdasrsf`'s
`srsf_align` with `center=True`).

Like `nlreg1d.register_srsf`, the function can be unpacked into two arrays:

- `yr` : registered data, (J,Q)
- `wf` : optimal warp functions, (J,Q)

but it actually returns a `RegistrationResult` object that also carries the template, the
warps as a `Warp1DList`, and method-specific information.
'''),
code('''
yr, wf = reg1d.register_srsf(yi, max_iter=5)

plt.figure(figsize=(8,5))
plot_Dorn2012(yr, xlabel='Time (%)')
plt.show()
'''),
code('''
result = reg1d.register_srsf(yi, max_iter=5)
print(result)
print('iterations:', result.info['niter'])
print('SRSF cost per iteration:', result.info['cost'].round(1))
'''),
md('''
### Warp functions and displacement fields

`result.warps` is a `Warp1DList`. Its `displacement_field()` method returns the deviation from
linear time expressed on the original time axis (the quantity plotted in `nlreg1d`).
'''),
code('''
wlist = result.warps                       # Warp1DList
d     = wlist.displacement_field()         # (8,101)

fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
plot_Dorn2012(wf, xlabel='Time (%)', ylabel='Warped time', title='Warp functions', ax=AX[0])
AX[0].plot([0,100],[0,1],'k:')
plot_Dorn2012(d, xlabel='Time (%)', ylabel='Deviations from linear time', title='Displacement fields', ax=AX[1])
plt.tight_layout()
plt.show()
'''),
md('''
### Three-panel summary

Every `RegistrationResult` has a `plot` method giving a before / after / warps summary.
'''),
code('''
result.plot(group=speed, colors=['k','b','g','r'])
plt.show()
'''),
md('''
### Agreement with fdasrsf (optional)

If `fdasrsf` happens to be installed, the cell below compares the warps found by `reg1d` with
those found by `fdasrsf.fdawarp.srsf_align` for the same data and settings. `reg1d` is **not**
a dependency-free copy of `fdasrsf` — the dynamic programming grid, the numerical derivative used
for the SRSF and the centring step are all independent implementations — so the two are expected
to agree closely but not exactly. This cell is skipped silently if `fdasrsf` is unavailable.
'''),
code('''
try:
    import fdasrsf, io, contextlib
    t   = np.linspace(0, 1, 101)
    with contextlib.redirect_stdout(io.StringIO()):
        fw = fdasrsf.fdawarp(yi.T, t)
        fw.srsf_align(MaxItr=5)
    wf_fdasrsf = fw.gam.T
    print('fdasrsf version:', fdasrsf.__version__)
    print('max |warp difference| per observation:', np.abs(wf - wf_fdasrsf).max(axis=1).round(3))
    fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
    plot_Dorn2012(fw.fn.T, xlabel='Time (%)', title='fdasrsf', ax=AX[0])
    plot_Dorn2012(yr, xlabel='Time (%)', title='reg1d', ax=AX[1])
    plt.tight_layout()
    plt.show()
except ImportError:
    print('fdasrsf is not installed; comparison skipped')
'''),
]


# ---------------------------------------------------------------------------
# notebook 2
# ---------------------------------------------------------------------------

nb2 = [
md('''
# 2 — Registration methods compared

`reg1d` currently implements two linear and four nonlinear registration methods, all callable
through the same interface and all returning a `RegistrationResult`:

| function | type | warp family |
|---|---|---|
| `register_linear` | linear | interpolation to a common grid (temporal normalisation) |
| `register_shift` | linear | γ(t) = t + δ |
| `register_affine` | linear | γ(t) = a t + b |
| `register_srsf` | nonlinear | elastic (SRSF / Fisher–Rao), dynamic programming |
| `register_dtw` | nonlinear | dynamic time warping (step patterns, optional window) |
| `register_landmark` | nonlinear | monotone interpolant through landmarks |
| `register_continuous` | nonlinear | smooth parametric γ = ∫exp(W), penalised least squares |

This notebook applies each of them to the `Dorn2012` dataset and comments on their suitability.
'''),
code(SETUP),
code(PLOT_DORN.replace("plt.figure(figsize=(8,5))\nplot_Dorn2012(y)\nplt.show()", "yi = reg1d.register_linear(y, n=101)")),
md('''
## Linear methods

Shift and affine registration translate (and, for affine, uniformly stretch) each observation in
time to best match the cross-sectional mean, with a few Procrustes iterations. Values needed from
outside the original domain are held at the boundary value. These methods can only correct
*global* timing differences; they cannot align features that move in opposite directions.
'''),
code('''
colors = ['k','b','g','r']
res_shift  = reg1d.register_shift(yi)
res_affine = reg1d.register_affine(yi)
print('shifts (fraction of stance):', res_shift.info['shift'].round(3))
print('affine (a, b):'); print(res_affine.info['params'].round(3))
res_shift.plot(group=speed, colors=colors, titles=('Linear', 'Shift-registered', 'Warps'))
res_affine.plot(group=speed, colors=colors, titles=('Linear', 'Affine-registered', 'Warps'))
plt.show()
'''),
md('''
## Landmark registration

Landmarks are event times that should coincide after registration. `detect_landmarks` provides a
simple automatic option: the global minimum, the zero crossing adjacent to the global maximum, and
the global maximum. For these data the global **minimum is not a reliable landmark** (the braking
phase has two dips and the deeper one changes between trials), so only the zero crossing and the
maximum are used below. The warp between landmarks is a monotone cubic (PCHIP) interpolant.
'''),
code('''
lm = reg1d.landmark.detect_landmarks(yi, kinds=('zero','max'))
print('landmark times (zero crossing, maximum):'); print(lm.round(3))
res_lm = reg1d.register_landmark(yi, kinds=('zero','max'))
print('targets:', res_lm.info['targets'].round(3))
res_lm.plot(group=speed, colors=colors, titles=('Linear', 'Landmark-registered', 'Warps'))
plt.show()
'''),
md('''
Landmarks may also be supplied explicitly, one row per observation, in normalised time (0–1).
The example below uses the two braking dips found with `scipy.signal.find_peaks` on the negated
signal, plus the propulsive maximum; observations for which the requested number of peaks is not
found are a common failure mode of automatic landmark detection and here would need manual
inspection.
'''),
code('''
from scipy.signal import find_peaks
lm2 = []
for yy in yi:
    ind,_ = find_peaks(-yy, prominence=20)
    dips  = np.sort(ind[np.argsort(-yy[ind])[::-1][:2]])   # two most negative dips
    lm2.append(np.r_[dips/100, np.argmax(yy)/100])
lm2 = np.array(lm2)
print(lm2.round(2))
res_lm2 = reg1d.register_landmark(yi, landmarks=lm2)
res_lm2.plot(group=speed, colors=colors, titles=('Linear', 'Landmark-registered (3 landmarks)', 'Warps'))
plt.show()
'''),
md('''
## Dynamic time warping

DTW minimises the accumulated pointwise amplitude difference along a monotone path. Its
strengths are simplicity and speed; its weaknesses for registration are that (i) it compares raw
amplitudes, so observations with different amplitudes (as here: peak GRF roughly doubles across
speeds) are matched by amplitude rather than by shape, producing the characteristic "staircase"
paths, and (ii) the path may run horizontally or vertically, so the warp is not strictly
increasing (flat segments). A Sakoe–Chiba window or a slope-constrained step pattern
(`'strict'`, slopes between 1/2 and 2) mitigates the latter.
'''),
code('''
res_dtw  = reg1d.register_dtw(yi)                              # symmetric2, unconstrained
res_dtw2 = reg1d.register_dtw(yi, step_pattern='strict')       # slopes limited to [1/2, 2]
res_dtw3 = reg1d.register_dtw(yi, window=0.1)                  # |gamma(t) - t| <= 0.1
res_dtw.plot(group=speed, colors=colors, titles=('Linear', 'DTW (symmetric2)', 'Warps'))
res_dtw2.plot(group=speed, colors=colors, titles=('Linear', 'DTW (strict slopes)', 'Warps'))
res_dtw3.plot(group=speed, colors=colors, titles=('Linear', 'DTW (window 0.1)', 'Warps'))
plt.show()
'''),
md('''
## Continuous (parametric) registration

Following Ramsay & Li (1998), the warp is γ(t) = ∫₀ᵗ exp(W) / ∫₀¹ exp(W) with W expanded in a
few sine functions, and the coefficients are fitted by penalised least squares to the
cross-sectional mean. The warps are very smooth and have few degrees of freedom, which is
attractive when timing differences are gradual; they cannot follow sharp local features such
as the double braking dip.
'''),
code('''
res_c4 = reg1d.register_continuous(yi, n_basis=4)
res_c8 = reg1d.register_continuous(yi, n_basis=8, lam=1e-3)
res_c4.plot(group=speed, colors=colors, titles=('Linear', 'Continuous (4 basis fns)', 'Warps'))
res_c8.plot(group=speed, colors=colors, titles=('Linear', 'Continuous (8 basis fns, penalised)', 'Warps'))
plt.show()
'''),
md('''
## Elastic (SRSF) registration

The SRSF approach compares *slopes* rather than amplitudes, is invariant to the amplitude scaling
that dominates these data, and yields proper diffeomorphic warps. It is the method used in the
`nlreg1d` paper and is the recommended default.
'''),
code('''
res_srsf = reg1d.register_srsf(yi, max_iter=5)
res_srsf.plot(group=speed, colors=colors, titles=('Linear', 'SRSF-registered', 'Warps'))
plt.show()
'''),
md('''
## Side-by-side comparison

Two simple descriptive quantities are tabulated for each method: the residual sum of squares about
the cross-sectional mean (an amplitude-dominated criterion that favours DTW, which is allowed to
match amplitudes directly) and the standard deviation (in % stance) of the time of the propulsive
peak (the global maximum).
'''),
code('''
results = {
    'linear only' : None,
    'shift'       : res_shift,
    'affine'      : res_affine,
    'landmark'    : res_lm,
    'dtw'         : res_dtw,
    'dtw (strict)': res_dtw2,
    'continuous'  : res_c8,
    'srsf'        : res_srsf,
}
print(f"{'method':14s} {'SSE about mean':>16s} {'SD of peak time (%)':>22s}")
for name,r in results.items():
    yy  = yi if r is None else r.y
    sse = ((yy - yy.mean(axis=0))**2).sum()
    sd  = np.argmax(yy, axis=1).std()
    print(f"{name:14s} {sse:16.3g} {sd:22.2f}")
'''),
code('''
fig,AX = plt.subplots(2, 4, figsize=(16,7))
for ax,(name,r) in zip(AX.ravel(), results.items()):
    yy = yi if r is None else r.y
    plot_Dorn2012(yy, xlabel='Time (%)', title=name, ax=ax)
    if ax is not AX[0,0]:
        ax.get_legend().remove()
plt.tight_layout()
plt.show()
'''),
md('''
## Summary of suitability for the Dorn2012 dataset

- **SRSF** (`register_srsf`): the most appropriate general-purpose method for these data; aligns
  the propulsive peak and zero crossing well and does a reasonable job on the double braking dip.
- **Landmark**: excellent where the landmarks are reliable (zero crossing, maximum) and it is the
  only method that guarantees a chosen event is aligned exactly; automatic landmark detection is
  fragile for the braking dips.
- **DTW**: fast, but amplitude-driven; produces staircase warps on these data unless constrained,
  and its warps are not diffeomorphisms. Useful as a quick first look or with a tight window.
- **Continuous**: produces smooth, low-dimensional warps; reasonable for the gradual speed-related
  timing shift but cannot resolve the braking dips.
- **Shift / affine**: correct only global offsets; of limited value for these data, but useful as a
  baseline or a first stage before nonlinear registration.
'''),
]


# ---------------------------------------------------------------------------
# notebook 3
# ---------------------------------------------------------------------------

nb3 = [
md('''
# 3 — Warps, simulation and registration accuracy

This notebook demonstrates the warp utilities in `reg1d.warp` (composition, inversion,
displacement fields, Karcher means, random warps) and uses simulated data with known warps to check
how accurately each nonlinear method recovers them.
'''),
code(SETUP),
md('''
### Random warps

`random_warp` draws warps as random points on the unit sphere of the square-root-slope
representation ψ = √γ′ (the construction used in the SRSF literature). `sigma` controls the
strength and `n_basis` the roughness.
'''),
code('''
t  = np.linspace(0, 1, 101)
fig,AX = plt.subplots(1, 3, figsize=(15,4))
for ax,sigma in zip(AX, [0.2, 0.5, 1.0]):
    w = reg1d.random_warp(J=10, Q=101, sigma=sigma, random_state=0)
    reg1d.plot.plot_warps(w, ax=ax, legend=False)
    ax.set_title(f'sigma = {sigma}')
plt.show()
'''),
md('''
### Warp algebra

`Warp1D` objects support application to data, composition and inversion; `Warp1DList` adds
Karcher means and centring. Inversion followed by composition returns the identity to within
interpolation error.
'''),
code('''
w   = reg1d.Warp1D( reg1d.random_warp(Q=101, sigma=0.5, random_state=1) )
wi  = w.inverse()
print('max |w o w^-1 - t| =', np.abs(w.compose(wi).w - t).max())

f   = np.exp(-((t-0.4)/0.1)**2) - 0.7*np.exp(-((t-0.7)/0.08)**2)
fw  = w.apply(f)                  # f( w(t) )
fig,AX = plt.subplots(1, 3, figsize=(15,4))
AX[0].plot(t, f, 'k', label='f'); AX[0].plot(t, fw, 'r', label='f o w'); AX[0].legend()
AX[1].plot(t, w.w, 'r', label='w'); AX[1].plot(t, wi.w, 'b', label='w^-1'); AX[1].plot([0,1],[0,1],'k:'); AX[1].legend()
AX[2].plot(t, w.displacement(), 'r', label='w(t) - t'); AX[2].plot(t, w.displacement_field(), 'b', label='displacement field'); AX[2].axhline(0, color='k', ls=':'); AX[2].legend()
plt.show()
'''),
md('''
### Karcher mean and centring

A set of warps is centred by composing each with the inverse of their Karcher mean; the Karcher
mean of the centred set is the identity. Registration functions with `center=True` (the default
for SRSF and continuous registration) do this automatically so that the registered data keep the
average timing of the original data.
'''),
code('''
W  = reg1d.Warp1DList( reg1d.random_warp(J=8, Q=101, sigma=0.5, random_state=2) )
Wc = W.center()
fig,AX = plt.subplots(1, 2, figsize=(10,4))
W.plot(ax=AX[0], legend=False);  AX[0].plot(t, W.mean().w, 'r', lw=3);  AX[0].set_title('original warps and Karcher mean')
Wc.plot(ax=AX[1], legend=False); AX[1].plot(t, Wc.mean().w, 'r', lw=3); AX[1].set_title('centred warps and Karcher mean')
plt.show()
'''),
md('''
### Recovery of known warps

A smooth template is warped by random warps, and each method is asked to undo the warping. The
recovered warps are compared with the truth after both are centred (registration can only
recover warps up to a common warp of the template). Note that the template is flat near both
ends of the domain, where no method can identify the warp; the parametric (continuous) method
extrapolates its smooth warp into those regions whereas the dynamic-programming methods default
to a straight path, which is the main source of the difference in warp error below.
'''),
code('''
rng   = np.random.default_rng(3)
J     = 10
f0    = np.exp(-((t-0.35)/0.08)**2) - 0.6*np.exp(-((t-0.6)/0.06)**2) + 0.4*np.exp(-((t-0.85)/0.05)**2)
wtrue = reg1d.random_warp(J=J, Q=101, sigma=0.4, n_basis=3, random_state=3)
y     = np.array([reg1d.warp.apply_warp(f0, w) for w in wtrue])      # y_i = f0 o w_i
y    += 0.01 * rng.standard_normal(y.shape)

# the registration warp should be (approximately) the inverse of the generating warp
winv  = reg1d.warp.center_warps( reg1d.warp.invert(wtrue) )[0]

methods = {
    'srsf'       : lambda y: reg1d.register_srsf(y),
    'dtw'        : lambda y: reg1d.register_dtw(y),
    'landmark'   : lambda y: reg1d.register_landmark(y, kinds=('min','max')),
    'continuous' : lambda y: reg1d.register_continuous(y, n_basis=6),
}
fig,AX = plt.subplots(2, 4, figsize=(16,7))
print(f"{'method':12s} {'RMS warp error':>16s} {'residual SSE':>14s}")
for k,(name,fn) in enumerate(methods.items()):
    r    = fn(y)
    wrec = reg1d.warp.center_warps(r.warps.asarray())[0]
    err  = np.sqrt(((wrec - winv)**2).mean())
    print(f"{name:12s} {err:16.4f} {r.sse()[1]:14.4f}")
    AX[0,k].plot(t, r.y.T, lw=0.8); AX[0,k].plot(t, f0, 'k', lw=2); AX[0,k].set_title(f'{name}: registered')
    AX[1,k].plot(t, winv.T, 'k', lw=0.5); AX[1,k].plot(t, wrec.T, 'r', lw=0.8); AX[1,k].set_title('true (black) vs recovered (red)')
plt.tight_layout()
plt.show()
'''),
md('''
### Elastic distances

`reg1d.srsf` also provides the amplitude and phase distances of the SRSF framework, which are
useful for clustering and hypothesis testing on timing versus amplitude effects.
'''),
code('''
d_amp = np.array([[reg1d.srsf.amplitude_distance(a, b) for b in y] for a in y])
d_pha = np.array([[reg1d.srsf.phase_distance(a, b) for b in y] for a in y])
fig,AX = plt.subplots(1, 2, figsize=(10,4))
for ax,D,s in zip(AX, [d_amp, d_pha], ['amplitude distance', 'phase distance']):
    im = ax.imshow(D, cmap='viridis'); ax.set_title(s); plt.colorbar(im, ax=ax)
plt.show()
'''),
]


def build(cells, name):
	nb = nbf.v4.new_notebook()
	nb['cells'] = cells
	nb['metadata']['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
	ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
	ep.preprocess(nb, {'metadata': {'path': HERE}})
	fpath = os.path.join(HERE, name + '.ipynb')
	nbf.write(nb, fpath)
	html, _ = HTMLExporter().from_notebook_node(nb)
	os.makedirs(os.path.join(HERE, 'html'), exist_ok=True)
	with open(os.path.join(HERE, 'html', name + '.html'), 'w') as f:
		f.write(html)
	print('wrote', fpath)


if __name__ == '__main__':
	build(nb1, '1-Registration')
	build(nb2, '2-Methods')
	build(nb3, '3-Warps')
