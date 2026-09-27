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
sys.path.insert(0, os.path.abspath('../src'))   # so that this notebook finds registration1d without installation
import numpy as np
from matplotlib import pyplot as plt
import registration1d as reg1d
print('registration1d version:', reg1d.__version__)
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
# 1 — Registration with registration1d

This notebook reproduces the workflow of `nlreg1d`'s notebook *3-Registration* using `registration1d`:
linear registration by interpolation to a common number of frames, followed by nonlinear
(elastic, SRSF-based) registration.

`registration1d` differs from `nlreg1d` in one important respect: it depends only on **numpy**, **scipy**
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
(as in `nlreg1d`) or a sequence of observations of different lengths. Like every other
`register_*` function it returns a `RegistrationResult`; the resampled array is its `.y`
attribute (tuple unpacking `yi, wf = ...` also works, the warps being identities here).
'''),
code('''
lin = reg1d.register_linear(y, n=101)
print(lin)
yi  = lin.y                                # (8,101) array
print(yi.shape, 'original lengths:', lin.info['lengths'])

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
print('elasticity penalty used (lam="auto" = median total variation of the observations):', round(result.info['lam'], 1))
print('centering:', result.info['center'])
'''),
md('''
Two defaults differ from `fdasrsf`: the elasticity penalty `lam` is data-adaptive (`'auto'`,
the median total variation of the observations; `fdasrsf` uses 0) because unpenalised
alignment produces noise-driven warps in flat regions (see the *Bayesian-vs-nlreg1d* notebook),
and the warps are centred so that their Karcher mean is the identity (`center='karcher'`; see
the *WarpCentering* notebook). Both can be switched off (`lam=0`, `center=False`).
'''),
md('''
### Warp functions and displacement fields

`result.warps` is a `Warp1DList`. Its `displacement_field()` method returns the deviation from
linear time expressed on the original time axis (the quantity plotted in `nlreg1d`).
The result also knows its time grid: `result.t`, `result.warps_t` and
`result.displacement_fields_t` give the same quantities in the units of the grid passed with
the keyword `t=` (here percent stance).
'''),
code('''
result = reg1d.register_srsf(yi, t=np.linspace(0, 100, 101), max_iter=5)   # grid in percent stance
wlist  = result.warps                       # Warp1DList (normalised time)
d      = wlist.displacement_field()         # (8,101), normalised time
print('displacement in % stance, range:', result.displacement_fields_t.min().round(2), result.displacement_fields_t.max().round(2))

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
### Applying the warps to other variables, and undoing them

A common biomechanical workflow is to register on one variable (e.g. the GRF) and apply the
same warps to other variables measured simultaneously (joint angles, EMG, ...):
`result.apply(z)`. The inverse operation `result.unapply(z)` maps registered-time quantities
(e.g. the template, or a statistical result) back onto each observation's own time base.
'''),
code('''
dydt = np.gradient(yi, axis=1)                    # a second variable on the same time base (here: the loading rate)
dydt_registered = result.apply(dydt)
y_back = result.unapply(result.y)                 # approximately the original yi
print('max |unapply(y) - yi| / max|yi| =', (np.abs(y_back - yi).max() / np.abs(yi).max()).round(4))
fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
plot_Dorn2012(dydt, xlabel='Time (%)', ylabel='Loading rate (N / frame)', title='second variable, linear', ax=AX[0])
plot_Dorn2012(dydt_registered, xlabel='Time (%)', ylabel='Loading rate (N / frame)', title='same warps applied', ax=AX[1])
plt.tight_layout(); plt.show()
'''),
md('''
### Agreement with fdasrsf (optional)

If `fdasrsf` happens to be installed, the cell below compares the warps found by `registration1d` with
those found by `fdasrsf.fdawarp.srsf_align` for the same data and settings. `registration1d` is **not**
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
    yr, wf = reg1d.register_srsf(yi, max_iter=5, lam=0)      # fdasrsf settings: no penalty
    print('fdasrsf version:', fdasrsf.__version__)
    print('max |warp difference| per observation:', np.abs(wf - wf_fdasrsf).max(axis=1).round(3))
    fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
    plot_Dorn2012(fw.fn.T, xlabel='Time (%)', title='fdasrsf', ax=AX[0])
    plot_Dorn2012(yr, xlabel='Time (%)', title='registration1d', ax=AX[1])
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

`registration1d` currently implements two linear and four nonlinear registration methods, all callable
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
| `register_sim` | nonlinear | self-modelling / shape-invariant model (amplitude + smooth warp) |
| `register_pairwise` | nonlinear | pairwise synchronisation (template-free) |
| `register_bayes` | nonlinear | Bayesian registration (posterior samples of the warps) — see notebook *Bayesian-vs-nlreg1d* |

This notebook applies each of them to the `Dorn2012` dataset and comments on their suitability.
All functions return a `RegistrationResult` (`LinearRegistrationResult` or
`NonlinearRegistrationResult`; `result.islinear` tells which).
'''),
code(SETUP),
code(PLOT_DORN.replace("plt.figure(figsize=(8,5))\nplot_Dorn2012(y)\nplt.show()", "yi = reg1d.register_linear(y, n=101).y")),
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
#### End-point effects in affine registration

With the default settings an affine map γ(t) = a t + b with a + b < 1 cuts off the end of an
observation (the registered curve ends at y(a+b) rather than at y(1) ≈ 0), and values needed
from outside the original domain are held at the boundary value (`fill_value='edge'`). For
forces, which vanish outside stance, two options restore physical plausibility:

- `fill_value='zero'`: outside the original domain the force is zero (rather than the edge value);
- `cover=True`: the affine map is constrained to b ≤ 0 and a + b ≥ 1, so that γ([0,1]) ⊇ [0,1]
  and the whole original observation, including both zero ends, is retained. (With `cover=True`
  the warps are not re-centred, because re-centring would re-introduce cut-off ends.)

`fill_value='extrapolate'` (linear extrapolation of the end slopes) is also available.
'''),
code('''
res_affine_cover = reg1d.register_affine(yi, cover=True, fill_value='zero')
print('(a, b) with cover=True:'); print(res_affine_cover.info['params'].round(3))
fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
plot_Dorn2012(res_affine.y, xlabel='Time (%)', title="affine, default ('edge' fill, unconstrained)", ax=AX[0])
plot_Dorn2012(res_affine_cover.y, xlabel='Time (%)', title="affine, cover=True, fill_value='zero'", ax=AX[1])
plt.tight_layout(); plt.show()
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
#### Smoother, physically plausible DTW warps

Plain DTW produces piecewise paths whose warps have discontinuous slopes (and flat segments),
which is unacceptable for signals such as forces that have smooth first derivatives. Three
remedies are available, separately or combined:

1. **Derivative DTW** (`derivative=True`; Keogh & Pazzani 2001) matches derivative estimates
   instead of values. Matching is then shape-based rather than amplitude-based, which removes
   most of the amplitude-driven staircase matching seen above. The path itself is still piecewise.
2. **Slope-constrained step patterns** (`step_pattern='strict'`, local slopes in [1/2, 2]) forbid
   horizontal and vertical runs, so the warp is strictly increasing.
3. **Warp smoothing** (`smooth=sigma`): the square-root slope √γ′ of each DTW warp is smoothed
   with a Gaussian kernel (width `sigma` in normalised time) and re-integrated. Because √γ′ ≥ 0,
   the result is always a valid, strictly increasing warp with a continuous derivative, and the
   total amount of warping is preserved. The same operation is available as
   `reg1d.warp.smooth_warp` and `Warp1DList.smooth` for any warp.

Note that the smoothing acts on the warp, not on the DTW objective: a smoothed DTW warp is no
longer optimal for the DTW criterion. It is a post-hoc regularisation, analogous to smoothing
a landmark warp; an alternative is to use the DTW warp only to initialise a smooth parametric
refinement (see the continuous method).
'''),
code('''
res_ddtw   = reg1d.register_dtw(yi, derivative=True)
res_ddtw_s = reg1d.register_dtw(yi, derivative=True, step_pattern='strict', smooth=0.03)
res_dtw_s  = reg1d.register_dtw(yi, step_pattern='strict', smooth=0.03)
res_ddtw.plot(group=speed, colors=colors, titles=('Linear', 'Derivative DTW', 'Warps'))
res_ddtw_s.plot(group=speed, colors=colors, titles=('Linear', 'Derivative DTW, strict, smoothed (0.03)', 'Warps'))
res_dtw_s.plot(group=speed, colors=colors, titles=('Linear', 'DTW, strict, smoothed (0.03)', 'Warps'))
plt.show()

# first derivatives of the registered data: smooth for the smoothed warps
fig,AX = plt.subplots(1, 3, figsize=(15,4))
for ax,(name,r) in zip(AX, [('DTW (symmetric2)', res_dtw), ('DTW strict + smooth', res_dtw_s), ('SRSF', reg1d.register_srsf(yi, max_iter=5))]):
    plot_Dorn2012(np.gradient(r.y, axis=1), xlabel='Time (%)', ylabel='dF/dt (N / frame)', title=name + ': first derivative', ax=ax)
    if ax is not AX[0]: ax.get_legend().remove()
plt.tight_layout(); plt.show()
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
## Self-modelling (shape-invariant model) registration

`register_sim` fits y_i = a_i μ(γ_i) + b_i (Kneip & Gasser 1988; Gervini & Gasser 2004): a common
shape function μ, per-observation amplitude scale and offset, and smooth parametric warps.
Modelling amplitude explicitly is attractive for these data, where peak force roughly doubles
across speeds.
'''),
code('''
res_sim = reg1d.register_sim(yi, n_basis=6)
print('amplitude (a, b) per observation:'); print(res_sim.info['amplitude'].round(2))
res_sim.plot(group=speed, colors=colors, titles=('Linear', 'Self-modelling registration', 'Warps'))
plt.show()
'''),
md('''
## Pairwise synchronisation

`register_pairwise` (Tang & Müller 2008) aligns every pair of observations and takes each
observation's warp as the Karcher mean of its warps to all others. It needs no template, at the
cost of J(J−1)/2 pairwise alignments (here with the SRSF engine).
'''),
code('''
res_pw = reg1d.register_pairwise(yi, engine='srsf')
res_pw.plot(group=speed, colors=colors, titles=('Linear', 'Pairwise synchronisation (SRSF engine)', 'Warps'))
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
    'ddtw+smooth' : res_ddtw_s,
    'continuous'  : res_c8,
    'sim'         : res_sim,
    'pairwise'    : res_pw,
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
fig,AX = plt.subplots(3, 4, figsize=(16,10.5))
for ax,(name,r) in zip(AX.ravel(), results.items()):
    yy = yi if r is None else r.y
    plot_Dorn2012(yy, xlabel='Time (%)', title=name, ax=ax)
    if ax is not AX[0,0]:
        ax.get_legend().remove()
for ax in AX.ravel()[len(results):]:
    ax.axis('off')
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
  and its warps are not diffeomorphisms. Derivative DTW with a slope-constrained step pattern and
  warp smoothing gives physically plausible warps and a good alignment of the propulsive peak.
- **Self-modelling**: handles the amplitude differences explicitly; smooth warps; good alignment
  of the main features, with the same local-minimum caveat as the continuous method.
- **Pairwise synchronisation**: results close to SRSF (same engine) without a template.
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


# ---------------------------------------------------------------------------
# notebook 4: multivariate
# ---------------------------------------------------------------------------

nb4 = [
md("""
# 4 — Multivariate (vector-valued) registration

The original Dorn et al. (2012) data contain all three ground reaction force components
(anteroposterior, vertical, mediolateral) for 18 trials at four running speeds. Registering the
three components *jointly* — one warp per trial, chosen so that all three components are aligned
at once — is preferable to registering them separately (which would give three different time
axes for the same trial) or to registering on one component and hoping the others follow.

`register_srsf` accepts a (J,Q,D) array and then uses the vector SRSF
q = f′ / √‖f′‖ (Srivastava et al. 2011; the same dynamic programming applies, with the
squared Euclidean norm of the D-dimensional difference in the segment costs).
"""),
code(SETUP),
code("""
dataset = reg1d.data.Dorn2012MV()
print(dataset)
Y       = dataset.resample(101)            # (18,101,3): linear registration of every component
speed   = dataset.group                    # 0..3
comps   = dataset.components
colors  = ['k','b','g','r']

def plot3(Y, title, axes=None):
    fig,AX = plt.subplots(1, 3, figsize=(15,4)) if axes is None else (None, axes)
    for k,ax in enumerate(AX):
        reg1d.plot.plot_curves(Y[:,:,k], group=speed, ax=ax, colors=colors, x='percent', legend=(k==0),
                               labels=[f'{v} m/s' for v in np.unique(dataset.speed)])
        ax.set_title(f'{title}: {comps[k]}'); ax.set_xlabel('Time (%)')
    AX[0].set_ylabel('Force (N)')
    return AX

plot3(Y, 'linear'); plt.tight_layout(); plt.show()
"""),
md("""
### Joint (multivariate) SRSF registration
"""),
code("""
res_mv = reg1d.register_srsf(Y, max_iter=5)
print(res_mv)
plot3(res_mv.y, 'multivariate SRSF'); plt.tight_layout(); plt.show()
fig,ax = plt.subplots(figsize=(5,4))
res_mv.warps.plot(ax=ax, group=speed, colors=colors, legend=False); ax.set_title('warps (one per trial)')
plt.show()
"""),
md("""
### Comparison with univariate registration on each component

Registering each component separately gives three different warps per trial (and therefore
three inconsistent time axes); registering on one component only (here the vertical force,
the usual choice) and applying its warps to the others (`result.apply`) keeps one time axis but
aligns the other components only insofar as their features co-occur with those of the driving
component.
"""),
code("""
res_v   = reg1d.register_srsf(Y[:,:,1], max_iter=5)          # vertical only
Y_from_v = res_v.apply(Y)                                    # apply vertical warps to all components
res_sep = [reg1d.register_srsf(Y[:,:,k], max_iter=5) for k in range(3)]
Y_sep   = np.stack([r.y for r in res_sep], axis=2)

def event_sd(Y):
    # SD (in % stance) of three event times: AP propulsive peak, vertical rising edge (50% of max), ML largest |F|
    ap = np.argmax(Y[:,:,0], axis=1)
    v  = np.array([np.argmax(yy > 0.5*yy.max()) for yy in Y[:,:,1]])
    ml = np.argmax(np.abs(Y[:,:,2]), axis=1)
    return [float(np.std(e).round(2)) for e in (ap, v, ml)]
print('SD (% stance) of event times [AP peak, vertical 50%-rise, ML |peak|]')
print('  linear only            ', event_sd(Y))
print('  multivariate SRSF      ', event_sd(res_mv.y))
print('  vertical-driven warps  ', event_sd(Y_from_v))
print('  separate per component ', event_sd(Y_sep), '  (three different time axes per trial)')
fig,AX = plt.subplots(2, 3, figsize=(15,8))
plot3(Y_from_v, 'vertical-driven', AX[0]); plot3(Y_sep, 'separate', AX[1])
plt.tight_layout(); plt.show()
"""),
md("""
The joint registration is the only approach that yields a single, physically consistent time
axis per trial while using the information in all three components — but a single warp per
trial is necessarily a compromise between the components: in the table above the joint warps
align the anteroposterior peak much better than linear registration, while the vertical rising
edge is aligned less well than when the vertical component alone drives the warps. Which
compromise is right depends on the research question. Note also that the vector SRSF weights
components by their slopes, so the large vertical force dominates and the small mediolateral
component contributes little; rescaling components (e.g. to unit variance) before registration
changes this weighting and may be desirable when the components have very different magnitudes.
"""),
]


# ---------------------------------------------------------------------------
# notebook: warp centering issue
# ---------------------------------------------------------------------------

nb_center = [
md("""
# Warp centering

### The issue

Group registration produces J warps γ_i and registered observations y_i∘γ_i. The registered
observations live on the time axis of the **template**, but the template's own timing is
arbitrary: it depends on which observation seeded the iteration, on the method, and on the
initial cross-sectional mean. Any common warp η can be applied to all γ_i (γ_i ↦ γ_i∘η) without
changing the *relative* alignment of the observations; it only changes the time axis on which
the registered data and the template are reported. Iterating the template (registering to an
arbitrary template, replacing it by the mean of the registered curves, and repeating) removes
the seed's influence on the template's *shape*, but not on its *timing*: the alignment cost is
invariant to a common warp, so the iteration cannot detect a timing bias, let alone remove it.

**Centering** picks one representative from the family {γ_i∘η}: each warp is composed with
the inverse of a common reference warp, γ_i ↦ γ_i∘w_ref⁻¹, and the reference is chosen so that
the registered data have a meaningful common time axis.

### The decision

Every nonlinear `register_*` function has a `center` keyword with the same four options
(`reg1d.warp.center_warps`):

| `center=` | reference warp w_ref | default for |
|---|---|---|
| `'karcher'` | Karcher (Fréchet) mean of the warps under the Fisher–Rao metric; afterwards the Karcher mean of the warps is the identity, i.e. the registered data have the average timing of the sample | SRSF, continuous, self-modelling, pairwise, Bayesian (all produce invertible warps) |
| `'pointwise'` | arithmetic (pointwise) mean of the warps; well defined also for warps with flat segments | DTW (raw DTW paths may contain horizontal runs, γ′ = 0, for which the Karcher mean is degenerate) |
| `'anchor'` | a monotone warp chosen so that a user-specified event keeps its mean time (`anchor=` a (J,) array of event times, or `'max'` / `'min'`) | — (offered as an explicit choice) |
| `'none'` | identity: no centering; the registered data keep the template's time axis | landmark registration (mean targets already anchor the registered data at the landmarks, which *is* the anchor convention) |

`center=True` selects the method's default and `center=False` is `'none'`, so that any
procedure can be run uncentred for comparison. The method used is recorded in
`result.info['center']`.
"""),
code(SETUP),
code("""
dataset = reg1d.data.Dorn2012()
speed   = dataset.group
yi      = reg1d.register_linear(dataset.y, n=101).y
colors  = ['k','b','g','r']
t       = np.linspace(0, 1, 101)

def show(results, titles):
    fig,AX = plt.subplots(len(results), 3, figsize=(15, 3.6*len(results)))
    AX = np.atleast_2d(AX)
    for row,(r,name) in enumerate(zip(results, titles)):
        reg1d.plot.plot_curves(r.y, group=speed, ax=AX[row,0], colors=colors, x='percent', legend=(row==0)); AX[row,0].set_title(f'{name}: registered')
        r.warps.plot(ax=AX[row,1], group=speed, colors=colors, legend=False)
        AX[row,1].plot(t, r.warps.mean().w, 'm', lw=3, label='Karcher mean'); AX[row,1].plot(t, r.warps.asarray().mean(axis=0), 'c--', lw=2, label='pointwise mean'); AX[row,1].legend(); AX[row,1].set_title(f'{name}: warps')
        reg1d.plot.plot_displacement_fields(r.warps.asarray(), group=speed, ax=AX[row,2], colors=colors, legend=False); AX[row,2].set_title(f'{name}: displacement fields')
    plt.tight_layout(); plt.show()

def report(r, name):
    g = r.warps.asarray()
    print(f"{name:32s} center={r.info['center']:9s} |Karcher mean - id|max = {np.abs(reg1d.warp.karcher_mean_warp(g)-t).max():.4f}   "
          f"|pointwise mean - id|max = {np.abs(g.mean(axis=0)-t).max():.4f}   mean peak time = {np.argmax(r.y,axis=1).mean():.2f} % (original {np.argmax(yi,axis=1).mean():.2f} %)")
"""),
md("""
### 1. SRSF: the four options

The relative alignment of the observations is identical in all four cases; only the common
time axis differs. With `'karcher'` the average timing of the sample is preserved (the mean
peak time equals that of the original data); with `'none'` the data inherit the timing of the
seed observation; with `'anchor'` the propulsive peak keeps its mean time exactly.
"""),
code("""
res = {c: reg1d.register_srsf(yi, max_iter=5, center=c, anchor='max') for c in ('karcher', 'pointwise', 'anchor', 'none')}
for c,r in res.items():
    report(r, f'SRSF, center={c!r}')
show(list(res.values()), [f"center='{c}'" for c in res])
"""),
md("""
### 2. Why it matters: displacement-field statistics

A two-sample test on displacement fields (the nlreg1d timing test) compares group means of
d_i(t). A common warp adds (approximately) the same displacement to every observation, so a
*difference* between groups is nearly invariant to centering — but the individual displacement
fields, one-sample questions ("is this group's timing different from linear time?"), and
comparisons between methods or studies are not. Centering makes the reference the average timing
of the sample, the only choice that does not depend on an arbitrary observation.
"""),
code("""
from registration1d import stats
groupAB = np.where(speed <= 1, 0, 1)         # slow (0,1) vs fast (2,3)
for c,r in res.items():
    d  = r.displacement_fields
    tt = stats.permutation_ttest2(d[groupAB==0], d[groupAB==1], n_perm=500, random_state=0)
    one = np.abs(d.mean(axis=0)).max()
    print(f"center={c!r:12s}: two-sample max|t| = {np.abs(tt['t']).max():.2f} (p = {tt['p']:.3f}, clusters {tt['clusters']});   max |mean displacement| = {one:.4f}")
"""),
md("""
### 3. DTW: pointwise mean by default, Karcher mean once the warps are diffeomorphic

Raw DTW warps (`symmetric2`) have flat segments; their Karcher mean is still computable but
the inverse of a flat warp is not unique, so the pointwise mean is the default. With
`step_pattern='strict'` and/or `smooth>0` the warps are strictly increasing and `'karcher'`
behaves as for SRSF.
"""),
code("""
r1 = reg1d.register_dtw(yi)                                                   # default: pointwise
r2 = reg1d.register_dtw(yi, center='karcher')
r3 = reg1d.register_dtw(yi, step_pattern='strict', smooth=0.03)               # pointwise
r4 = reg1d.register_dtw(yi, step_pattern='strict', smooth=0.03, center='karcher')
r5 = reg1d.register_dtw(yi, center=False)
for r,name in [(r1,'DTW symmetric2'),(r2,'DTW symmetric2'),(r3,'DTW strict+smooth'),(r4,'DTW strict+smooth'),(r5,'DTW symmetric2')]:
    report(r, name)
print('minimum warp slope, symmetric2:', np.gradient(r1.warps.asarray(), axis=1).min().round(4), '  strict+smooth:', np.gradient(r3.warps.asarray(), axis=1).min().round(4))
show([r1, r3], ["DTW symmetric2, center='pointwise'", "DTW strict+smooth, center='pointwise'"])
"""),
md("""
### 4. Landmark registration: mean targets are the anchor convention

With `targets='mean'` every landmark is moved to its mean time across observations, so the
registered data are anchored to the sample's average timing *at the landmarks* by construction.
This is the `'anchor'` convention applied at K events at once; its Karcher mean is close to,
but not exactly, the identity. Other centering methods would move the landmarks off their
targets and are offered only for comparison.
"""),
code("""
r_lm  = reg1d.register_landmark(yi, kinds=('zero','max'))                    # default: none (mean targets)
r_lmk = reg1d.register_landmark(yi, kinds=('zero','max'), center='karcher')
for r,name in [(r_lm,'landmark, mean targets'),(r_lmk,'landmark, then Karcher')]:
    report(r, name)
    print('    SD of peak time after registration (%):', np.argmax(r.y, axis=1).std().round(2))
"""),
md("""
### 5. The other methods

Continuous, self-modelling, pairwise-synchronisation and Bayesian registration all produce
strictly increasing warps and default to `'karcher'`; for Bayesian registration the reference
warp is computed from the posterior-mean warps and applied to every posterior sample, so the
credible bands are reported on the same axis. Pairwise synchronisation is template-free and its
warps are already nearly centred (the mean of the pairwise warps is close to the identity);
Karcher centering makes this exact.
"""),
code("""
for name,fn in [('continuous', lambda c: reg1d.register_continuous(yi, n_basis=6, center=c)),
                ('sim',        lambda c: reg1d.register_sim(yi, n_basis=6, max_iter=3, center=c)),
                ('pairwise',   lambda c: reg1d.register_pairwise(yi, center=c))]:
    for c in ('karcher', 'none'):
        report(fn(c), f'{name}')
"""),
md("""
### Summary

- Centering changes only the common time axis, never the relative alignment.
- `'karcher'` is the default wherever warps are invertible; `'pointwise'` is the default for raw
  DTW; landmark registration is anchored through its mean targets; `'anchor'` is available
  everywhere for an event-based reference; `center=False` gives the uncentred result for
  comparison.
- For displacement-field statistics the between-group difference is nearly invariant to the
  choice, but one-sample statements and comparisons across methods or studies require a
  reproducible convention — which is what the defaults provide.
"""),
]


# ---------------------------------------------------------------------------
# notebook: Bayesian vs nlreg1d
# ---------------------------------------------------------------------------

nb_bayes = [
md("""
# Bayesian registration versus the nlreg1d timing analysis

### Background

The nlreg1d paper tests for **timing effects** between two groups by (1) registering all
observations with SRSF alignment, (2) computing each observation's displacement field
d_i(t) (deviation from linear time), and (3) running a two-sample test on the d_i(t) with
nonparametric (permutation, max-t) inference; amplitude effects are tested the same way on the
registered observations. Registration is treated as a deterministic preprocessing step: each
warp is a point estimate, and the uncertainty of the registration itself does not enter the
inference.

**Bayesian registration** (Cheng, Dryden & Huang 2016; Lu, Herbei & Kurtek 2017) instead treats
each warp as an unknown with a posterior distribution: with a Gaussian error model in SRSF space
and a smoothness prior on the warp, MCMC yields posterior samples of γ_i, hence posterior means
and credible bands for the displacement fields. `reg1d.bayes` implements a simplified version
(finite cosine basis in the tangent space of the identity warp, pCN Metropolis sampling, Gibbs
update of the noise variance; see `ALGORITHMS.md`).

This notebook reproduces the nlreg1d analysis of the two simulated datasets — **A** (pure
amplitude effect) and **B** (pure timing effect) — with `registration1d`, then repeats it with Bayesian
registration and compares the conclusions.
"""),
code(SETUP),
code("""
from registration1d import stats, bayes
np.random.seed(0)
t      = np.linspace(0, 1, 101)
colors = ['0.0', (0.3,0.5,0.99)]

def load(name):
    ds = reg1d.data.SimulatedA() if name == 'A' else reg1d.data.SimulatedB()
    return ds.y, ds.group

def plot_groups(y, group, ax, title, ylabel=''):
    reg1d.plot.plot_curves(y, group=group, ax=ax, colors=colors, x='percent', lw=0.7,
                           labels=['Group 0', 'Group 1'])
    ax.set_title(title); ax.set_xlabel('Domain position (%)'); ax.set_ylabel(ylabel)

def plot_test(res, ax, title):
    x = np.linspace(0, 100, res['t'].size)
    ax.plot(x, res['t'], 'k'); ax.axhline(res['threshold'], color='r', ls='--'); ax.axhline(-res['threshold'], color='r', ls='--')
    for lo,hi in res['clusters']:
        ax.axvspan(x[lo], x[hi], color='r', alpha=0.2)
    ax.axhline(0, color='k', lw=0.5); ax.set_title(f"{title}  (p = {res['p']:.3f})"); ax.set_xlabel('Domain position (%)'); ax.set_ylabel('t')

fig,AX = plt.subplots(1, 2, figsize=(12,4))
for ax,name in zip(AX, 'AB'):
    y,g = load(name); plot_groups(y, g, ax, f'Dataset {name}', 'Dependent variable')
plt.tight_layout(); plt.show()
"""),
md("""
## 1. The nlreg1d analysis with registration1d (point-estimate registration)

SRSF registration (5 iterations, as in `fig_datasetA.py` / `fig_datasetB.py`), followed by
permutation two-sample tests on the registered data (amplitude) and on the displacement fields
(timing). `reg1d.stats.timing_test` wraps both tests; the permutation inference uses the maximum
|t| over the domain (the "tmax" inference of SnPM).
"""),
code("""
def analyse(lam, show=True):
    out = {}
    if show:
        fig,AX = plt.subplots(2, 4, figsize=(18,8))
    for row,name in enumerate('AB'):
        y,g   = load(name)
        r     = reg1d.register_srsf(y, max_iter=5, lam=lam)
        ta,tt = stats.timing_test(r, g, n_perm=1000, random_state=0)
        out[name] = dict(result=r, amp=ta, tim=tt)
        print(f'lam = {lam!s:>5s} (used {r.info["lam"]:.1f}), dataset {name}: amplitude p = {ta["p"]:.3f} {ta["clusters"]},  timing p = {tt["p"]:.3f} {tt["clusters"]}')
        if show:
            plot_groups(r.y, g, AX[row,0], f'{name}: registered (lam = {lam})', 'DV')
            plot_test(ta, AX[row,1], f'{name}: amplitude test')
            plot_groups(r.displacement_fields, g, AX[row,2], f'{name}: displacement fields', 'Displacement')
            plot_test(tt, AX[row,3], f'{name}: timing test')
    if show:
        plt.tight_layout(); plt.show()
    return out

point0 = analyse(lam=0)
"""),
md("""
Dataset B behaves as in the paper (timing effect, no amplitude effect). Dataset A shows the
expected amplitude effect, **but also a spurious timing effect near the start of the domain**.
The displacement fields show why: away from the bump the simulated curves are flat and noisy, and
the SRSF (the square root of the derivative) amplifies that noise, so the dynamic programme finds
large, noise-driven warps in the flat regions where the objective is nearly indifferent. The
same happens with `fdasrsf` (whose warps differ from `registration1d`'s in exactly these regions and
which, on the same data, gives a timing-test p-value of about 0.08 with the permutation test
used here), i.e. the outcome of the timing test in flat regions is decided by algorithmic details
rather than by the data.

The remedy in the SRSF framework is the elasticity penalty `lam`, which penalises departure of
√γ′ from 1 (the same `lam` as in `fdasrsf.srsf_align`). Its scale is that of the squared SRSF
distance, so for these data (SRSF values of order 10) a value of the order of 100 is needed to
dominate the noise-driven cost differences while leaving the genuine alignment of the peak
intact:
"""),
code("""
point = analyse(lam=100)
_     = analyse(lam='auto', show=False)
"""),
md("""
With `lam = 100` both datasets behave exactly as intended: A has an amplitude effect and no
timing effect, B a timing effect and no amplitude effect. Following this finding the default of
`register_srsf` is now `lam='auto'`, the median total variation ∫|f′| of the observations
(here about 49 for A and 45 for B), which gives the same conclusions; `lam=0` reproduces the
unpenalised `fdasrsf` behaviour. `lam=100` is kept for the remainder of this notebook (both for
the SRSF template and for the initialisation of the Bayesian chains). This sensitivity to `lam`
in flat regions is itself a strong argument for uncertainty-aware registration.

## 2. Bayesian registration

For each observation, `register_bayes` samples the posterior of its warp to the SRSF
Karcher-mean template. The point-estimate (dynamic programming) warp initialises each chain.
The credible band of the displacement field is the new information: it says how well determined
each observation's timing is.
"""),
code("""
import time
post = {}
for name in 'AB':
    y,g = load(name)
    t0  = time.time()
    rb  = reg1d.register_bayes(y, n_samples=1500, burn=1000, K=8, tau=0.3, random_state=1, max_iter=5, lam=100)
    post[name] = rb
    print(f'Dataset {name}: {time.time()-t0:.0f} s, mean acceptance rate {rb.info["accept"].mean():.2f}')

fig,AX = plt.subplots(2, 2, figsize=(12,8))
x = np.linspace(0, 100, 101)
for row,name in enumerate('AB'):
    rb = post[name]; y,g = load(name)
    for i in range(rb.J):
        ci = rb.info['disp_ci'][i]
        AX[row,0].fill_between(x, ci[0], ci[1], color=colors[g[i]], alpha=0.15)
    plot_groups(rb.displacement_fields, g, AX[row,0], f'{name}: posterior-mean displacement fields with 95% credible bands', 'Displacement')
    # posterior of the group difference in mean displacement (registration uncertainty only)
    S    = rb.info['samples']                                      # (J,S,Q) warp samples
    D    = np.array([reg1d.warp.displacement_field(s) for s in S]) # (J,S,Q) displacement samples
    diff = D[g==1].mean(axis=0) - D[g==0].mean(axis=0)             # (S,Q)
    lo,hi = np.percentile(diff, [2.5, 97.5], axis=0)
    AX[row,1].fill_between(x, lo, hi, color='r', alpha=0.25, label='95% credible band')
    AX[row,1].plot(x, diff.mean(axis=0), 'r', label='posterior mean')
    AX[row,1].axhline(0, color='k', lw=0.5); AX[row,1].legend()
    AX[row,1].set_title(f'{name}: group difference in mean displacement (registration uncertainty only)')
plt.tight_layout(); plt.show()
"""),
md("""
Note that the credible band of the *group difference* (right column) excludes zero over most of
the domain for dataset A even though A has no timing effect. This is not a contradiction: that
band quantifies only how precisely each individual warp is determined by its own observation;
it says nothing about whether the two groups of warps differ relative to the between-subject
variability, which is what the frequentist timing test measures. A Bayesian answer to the
group question requires either a hierarchical model or the propagation described next.

## 3. Propagating registration uncertainty into the timing test

The credible band of the group difference above reflects only the uncertainty of the
registration (how well each warp is determined given its observation), not the between-subject
variability that the frequentist test is built on. The two sources can be combined in a
simple *posterior-predictive* way: for each posterior draw s, take the s-th warp sample of every
observation, compute the displacement fields, and run the nlreg1d timing test; the distribution
of the resulting test statistics and p-values across draws shows whether the conclusion of the
point-estimate analysis is robust to registration uncertainty.
"""),
code("""
n_draw = 60
fig,AX = plt.subplots(1, 2, figsize=(12,4))
for ax,name in zip(AX, 'AB'):
    rb = post[name]; y,g = load(name)
    S  = rb.info['samples']; idx = np.linspace(0, S.shape[1]-1, n_draw).astype(int)
    pvals, tmaxs = [], []
    for s in idx:
        d  = reg1d.warp.displacement_field(S[:, s, :])
        tt = stats.permutation_ttest2(d[g==0], d[g==1], n_perm=300, random_state=int(s))
        pvals.append(tt['p']); tmaxs.append(np.abs(tt['t']).max())
        ax.plot(x, tt['t'], color='0.6', lw=0.5)
    ax.plot(x, point[name]['tim']['t'], 'k', lw=2, label='point-estimate registration')
    ax.axhline(point[name]['tim']['threshold'], color='r', ls='--', label='threshold (point estimate)')
    ax.axhline(-point[name]['tim']['threshold'], color='r', ls='--')
    ax.set_title(f'{name}: timing-test t curves over {n_draw} posterior draws'); ax.legend(); ax.set_xlabel('Domain position (%)')
    pvals = np.array(pvals)
    print(f'Dataset {name}: point-estimate p = {point[name]["tim"]["p"]:.3f};  over posterior draws: median p = {np.median(pvals):.3f}, '
          f'fraction of draws with p < 0.05 = {(pvals < 0.05).mean():.2f}, max|t| range = [{min(tmaxs):.2f}, {max(tmaxs):.2f}]')
plt.tight_layout(); plt.show()
"""),
md("""
## 4. How the two approaches relate

- **Same estimand, different treatment of uncertainty.** Both approaches quantify timing as a
  displacement field derived from SRSF warps. nlreg1d uses one warp per observation and puts all
  uncertainty into the between-subject variability of the displacement fields; Bayesian
  registration adds a second layer — the uncertainty of each warp given its observation — which
  is invisible to the point-estimate analysis.
- **When it matters.** Around sharp features the posterior of each warp is narrow, the
  posterior-mean warps are close to the dynamic-programming warps, and the timing test's
  conclusion is stable across posterior draws. Registration uncertainty becomes important for
  noisy data, in flat regions of the observations (where the warp is poorly identified — the
  credible bands widen there, and, as section 1 showed, point-estimate methods can produce
  spurious group differences there unless penalised), and near the domain boundaries.
- **What Bayesian registration offers that nlreg1d cannot.** (i) Per-observation credible
  bands for timing (useful for flagging observations whose registration is unreliable, e.g. in a
  front end); (ii) a principled way to propagate registration uncertainty into downstream tests
  (posterior-predictive checks as above, or a fully hierarchical model with group-level warp
  distributions, which would replace the permutation test altogether); (iii) posterior
  probabilities such as P(displacement difference > 0 at t).
- **Costs and caveats.** MCMC is roughly two orders of magnitude slower than dynamic programming;
  results depend on the prior scale (`tau`), the basis size (`K`) and the noise model; and the
  credible bands of this simple model are known to be optimistic because SRSF residuals are
  autocorrelated (the `n_eff` argument tempers the likelihood to compensate). The template is
  fixed at the SRSF Karcher mean; a full treatment would also put a prior on the template.
- **Suggested direction.** A hierarchical Bayesian model (observation warps ~ group-level warp
  distributions on the sphere of √γ′, with a prior on the group difference) would give a direct
  Bayesian counterpart to the nlreg1d timing test, including the amplitude test on the aligned
  functions. The pieces needed (tangent-space parameterisation, pCN sampling) are in `reg1d.bayes`.
"""),
]


# ---------------------------------------------------------------------------
# notebook: real-time registration
# ---------------------------------------------------------------------------

nb_rt = [
md("""
# Real-time versus normalised-time registration

### The issue

The usual workflow (nlreg1d, and notebooks 1–2 here) linearly registers all observations to a
common grid over [0, 1] *before* nonlinear registration. That step does two things: it resamples
(harmless for smooth data), and it rescales every observation's time axis by its own duration.
The rescaling changes the first derivatives: after normalisation, dF/ds of a short (fast) trial
is inflated relative to a long (slow) trial by the ratio of their durations. Since the SRSF is
q = sign(f′)√|f′| — and derivative DTW likewise works on f′ — the elastic alignment then compares
slopes expressed in different physical time units.

`registration1d` now offers **real-time registration**: pass the observations as a list (different
lengths allowed) together with their sampling interval / frequency / time vectors, and

- the observations are never resampled before alignment;
- SRSFs are computed with derivatives in physical time;
- the dynamic programme runs between each observation's own grid and a reference grid
  (n_ref points over [0, T_ref], T_ref = mean duration by default);
- the warps Γ_i map reference time onto each observation's real time (`info['warps_realtime']`,
  in seconds); the normalised warps γ_i(s) = Γ_i(s T_ref)/T_i are stored in `result.warps` as usual;
- `info['displacement_realtime']` = Γ_i(τ) − τ T_i/T_ref is the deviation from a pure linear
  rescaling, in seconds.

This notebook compares the two workflows on the `Dorn2012` data (185–383 frames; taking 1 kHz
sampling, stance durations of 0.18–0.38 s across the four speeds — a factor of two, so the
difference in derivative scaling is large).
"""),
code(SETUP),
code("""
dataset = reg1d.data.Dorn2012()
speed   = dataset.group
ylist   = list(dataset.y)                       # ragged list: 8 observations, 185-383 frames
fs      = 1000.0                                # assumed sampling frequency (Hz)
colors  = ['k','b','g','r']
labels  = [f'Speed = {i}' for i in range(4)]
print('lengths:', [len(yy) for yy in ylist])
print('durations (s):', np.round([len(yy)/fs for yy in ylist], 3))

fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
for i,yy in enumerate(ylist):
    AX[0].plot(np.arange(yy.size)/fs, yy, color=colors[speed[i]])
    AX[0].set_xlabel('Time (s)'); AX[0].set_title('original observations, real time')
    AX[1].plot(np.arange(yy.size)/fs, np.gradient(yy, 1/fs)/1000, color=colors[speed[i]])
    AX[1].set_xlabel('Time (s)'); AX[1].set_ylabel('dF/dt (kN/s)'); AX[1].set_title('first derivatives in real time')
AX[0].set_ylabel('Anteroposterior GRF (N)')
plt.tight_layout(); plt.show()
"""),
md("""
## 1. SRSF registration: normalised time versus real time
"""),
code("""
yi     = reg1d.register_linear(ylist, n=101).y
r_norm = reg1d.register_srsf(yi, max_iter=5)                          # normalised-time workflow
r_real = reg1d.register_srsf(ylist, t=f'fs={fs:g}', max_iter=5)       # real-time workflow
print(r_real)
print(f'reference duration T_ref = {r_real.t[-1]:.3f} s')

fig,AX = plt.subplots(2, 3, figsize=(16,8))
for row,(r,name) in enumerate([(r_norm,'normalised time'),(r_real,'real time')]):
    reg1d.plot.plot_curves(r.y0, group=speed, ax=AX[row,0], colors=colors, x='percent', labels=labels); AX[row,0].set_title(f'{name}: linearly rescaled (before)')
    reg1d.plot.plot_curves(r.y,  group=speed, ax=AX[row,1], colors=colors, x='percent', legend=False); AX[row,1].set_title(f'{name}: registered')
    r.warps.plot(ax=AX[row,2], group=speed, colors=colors, legend=False); AX[row,2].set_title(f'{name}: normalised warps')
    for ax in AX[row,:2]: ax.set_xlabel('Time (% of reference duration)')
plt.tight_layout(); plt.show()

print('max |normalised warp difference| per observation:', np.abs(r_real.warps.asarray() - r_norm.warps.asarray()).max(axis=1).round(3))
print(f'SD of propulsive-peak time (% of domain): before {np.argmax(yi,axis=1).std():.2f}, '
      f'normalised {np.argmax(r_norm.y,axis=1).std():.2f}, real {np.argmax(r_real.y,axis=1).std():.2f}')
"""),
md("""
The two workflows give similar but not identical warps. The differences are largest for the
observations whose duration is furthest from the reference (the slowest and fastest trials): in
normalised time their SRSFs are scaled by √(T_ref/T_i) relative to real time, which changes the
relative weight of their features in the alignment cost and hence the compromise the dynamic
programme strikes between aligning the braking dips and the propulsive peak.

### Real-time warps and displacement fields

In real time, "no warping" is not the identity but the linear rescaling Γ(τ) = τ·T_i/T_ref
(the dotted lines below). The real-time displacement field is the deviation from that line, in
seconds: it says by how much a feature of observation i was moved, in physical time.
"""),
code("""
tau = r_real.t
fig,AX = plt.subplots(1, 3, figsize=(16,4.5))
for i in range(8):
    c = colors[speed[i]]
    AX[0].plot(tau, r_real.info['warps_realtime'][i], color=c)
    AX[0].plot([0, tau[-1]], [0, r_real.info['durations'][i]], color=c, ls=':', lw=0.8)
    AX[1].plot(tau, 1000*r_real.info['displacement_realtime'][i], color=c)
    AX[2].plot(tau, 1000*r_norm.displacement_fields[i]*r_real.info['durations'][i], color=c)
AX[0].set_xlabel('Reference time (s)'); AX[0].set_ylabel('Observation time (s)'); AX[0].set_title('real-time warps (dotted: linear rescaling)')
AX[1].axhline(0, color='k', ls=':'); AX[1].set_xlabel('Reference time (s)'); AX[1].set_ylabel('ms'); AX[1].set_title('real-time displacement (deviation from linear rescaling)')
AX[2].axhline(0, color='k', ls=':'); AX[2].set_xlabel('Reference time (s)'); AX[2].set_ylabel('ms'); AX[2].set_title('normalised-time displacement, converted to ms')
plt.tight_layout(); plt.show()
"""),
md("""
## 2. Derivative information

The right-hand panel of the first figure showed that the fastest trials have loading rates
several times those of the slowest. After normalisation this information is partly mixed with
the duration: a steep slope in normalised time may be a steep slope in real time, or a short
trial. Real-time registration keeps loading rates in physical units throughout, so the registered
derivatives (`result.apply` on the ragged list of real-time derivatives) remain comparable across
speeds.
"""),
code("""
dydt      = [np.gradient(yy, 1/fs)/1000 for yy in ylist]          # kN/s, each on its own grid
dydt_real = r_real.apply(dydt)                                      # (8,101) on the reference axis
dydt_norm = r_norm.apply(np.gradient(yi, axis=1)*100/1000)           # normalised-time derivative per % stance, kN per % 
fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
reg1d.plot.plot_curves(dydt_real, group=speed, ax=AX[0], colors=colors, x='percent', labels=labels); AX[0].set_title('real-time registration: dF/dt (kN/s)')
reg1d.plot.plot_curves(dydt_norm, group=speed, ax=AX[1], colors=colors, x='percent', legend=False); AX[1].set_title('normalised-time registration: dF/ds (kN per % stance)')
for ax in AX: ax.set_xlabel('Time (% of domain)')
plt.tight_layout(); plt.show()
"""),
md("""
## 3. Real-time DTW and landmark registration

The same ragged input works for `register_dtw` and `register_landmark`. For DTW, derivative
estimates are divided by the sampling interval so that they, too, are compared in physical
time. Note that slope-constrained step patterns (`'strict'`) cannot bridge length ratios beyond
their slope range (here up to 383/101 ≈ 3.8 against a 101-point reference); use `symmetric2`
with warp smoothing, or set `n_ref` close to the observation lengths.
"""),
code("""
r_dtw = reg1d.register_dtw(ylist, t=1/fs, derivative=True, smooth=0.03)
r_lm  = reg1d.register_landmark(ylist, t=1/fs, kinds=('zero','max'))
print('landmark targets (s):', r_lm.info['targets'].round(3))
fig,AX = plt.subplots(1, 2, figsize=(12,4.5))
reg1d.plot.plot_curves(r_dtw.y, group=speed, ax=AX[0], colors=colors, x='percent', labels=labels); AX[0].set_title('real-time derivative DTW (smoothed)')
reg1d.plot.plot_curves(r_lm.y,  group=speed, ax=AX[1], colors=colors, x='percent', legend=False); AX[1].set_title('real-time landmark registration')
plt.tight_layout(); plt.show()
"""),
md("""
## 4. Which one to use?

- If the scientific time axis is **normalised** (percent stance, percent gait cycle) and
  duration is not of interest, the classical workflow is appropriate, and the SRSF's amplitude
  sensitivity to the duration rescaling is part of that model.
- If **physical time** matters — loading rates, absolute timing hypotheses, comparing conditions
  with very different durations (as here: a factor of two across speeds) — register in real time.
  The registered data still live on a common reference axis (so downstream statistics are
  unchanged), but the warps are physically interpretable (seconds onto seconds), the derivatives
  keep their units, and the displacement fields separate "this trial is shorter" (the linear
  rescaling) from "this feature occurred earlier within the trial" (the deviation from it).
- In both cases the differences between the two workflows on these data are moderate (normalised
  warp differences up to a few percent of the domain), concentrated in the trials whose duration
  is furthest from the reference; for datasets with smaller duration variation they will be
  smaller still.
"""),
]


# ---------------------------------------------------------------------------
# notebook: PyQtGraph backend
# ---------------------------------------------------------------------------

nb_qt = [
md("""
# PyQtGraph backend

`registration1d.plot` draws with Matplotlib. For use inside a Qt application (PyQt6 / PySide6),
`registration1d.plotqt` provides the same helpers on top of [PyQtGraph](https://www.pyqtgraph.org)
(MIT licence), which renders through Qt's scene graph and stays interactive — pan, zoom, hover —
with thousands of curves. It is an optional dependency:

    pip install "registration1d[qt]" PyQt6      # or PySide6

The functions mirror the Matplotlib ones (`plot_curves`, `plot_warps`, `plot_displacement_fields`,
`plot_registration`) and `RegistrationResult.plot(backend='pyqtgraph')` is a shortcut. Two design
points matter for embedding:

- every function accepts an existing PyQtGraph `PlotItem` / `PlotWidget` through the `plot=`
  argument (or a `GraphicsLayoutWidget` through `win=`) and draws into it, so an application keeps
  ownership of its widgets; with no target a stand-alone widget is created and returned;
- nothing starts a Qt event loop. Call `plotqt.app().exec()` for a stand-alone window, or let the
  host application run its loop.

This notebook runs *off-screen* (no display is needed) and shows the PyQtGraph output as PNG images
rendered with `plotqt.to_image`, next to the corresponding Matplotlib figures for comparison.
"""),
code("""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'      # render without a display (must precede the Qt import)
""" + SETUP.strip('\n') + """
from registration1d import plotqt
from IPython.display import Image, display
import pyqtgraph as pg
print('pyqtgraph', pg.__version__, '| Qt binding:', pg.Qt.QT_LIB)
"""),
code("""
dataset = reg1d.data.Dorn2012()
speed   = dataset.group
yi      = reg1d.register_linear(dataset.y, n=101).y
colors  = ['k','b','g','r']
labels  = [f'Speed = {i}' for i in range(4)]
result  = reg1d.register_srsf(yi, max_iter=5)
"""),
md("""
### 1. Three-panel registration summary — PyQtGraph versus Matplotlib
"""),
code("""
win = result.plot(group=speed, colors=colors, backend='pyqtgraph')      # GraphicsLayoutWidget, 3 PlotItems
plotqt.to_image(win, 'pg_registration.png', size=(1200, 360))
display(Image('pg_registration.png'))

fig, AX = result.plot(group=speed, colors=colors)                        # Matplotlib, for comparison
plt.show()
"""),
md("""
### 2. Individual panels, and drawing into caller-owned widgets

An application typically creates its own layout and hands the plot items to `plotqt`. Below a
2 × 2 `GraphicsLayoutWidget` is filled with raw observations (different lengths), registered
observations, warps and displacement fields; the return value of each call is the item passed in.
"""),
code("""
glw = pg.GraphicsLayoutWidget(title='registration1d')
p_raw  = glw.addPlot(title='raw observations (frames)')
p_reg  = glw.addPlot(title='SRSF-registered')
glw.nextRow()
p_warp = glw.addPlot(title='warps')
p_disp = glw.addPlot(title='displacement fields')

plotqt.plot_curves(list(dataset.y), group=speed, plot=p_raw, colors=colors, labels=labels, xlabel='Frame', ylabel='GRF (N)')
plotqt.plot_curves(result.y, group=speed, plot=p_reg, colors=colors, x='percent', legend=False, xlabel='Time (%)')
plotqt.plot_warps(result.warps.asarray(), group=speed, plot=p_warp, colors=colors, legend=False)
plotqt.plot_displacement_fields(result.warps.asarray(), group=speed, plot=p_disp, colors=colors, legend=False)
plotqt.to_image(glw, 'pg_panels.png', size=(1000, 650))
display(Image('pg_panels.png'))
"""),
code("""
fig, AX = plt.subplots(2, 2, figsize=(11, 7))
reg1d.plot.plot_curves(list(dataset.y), group=speed, ax=AX[0,0], colors=colors, labels=labels); AX[0,0].set_title('raw observations (frames)')
reg1d.plot.plot_curves(result.y, group=speed, ax=AX[0,1], colors=colors, x='percent', legend=False); AX[0,1].set_title('SRSF-registered')
reg1d.plot.plot_warps(result.warps.asarray(), group=speed, ax=AX[1,0], colors=colors, legend=False); AX[1,0].set_title('warps')
reg1d.plot.plot_displacement_fields(result.warps.asarray(), group=speed, ax=AX[1,1], colors=colors, legend=False); AX[1,1].set_title('displacement fields')
plt.tight_layout(); plt.show()
"""),
md("""
### 3. Themes

Stand-alone widgets use a white background with black foreground (`plotqt.set_theme('light')`,
applied automatically) so that the default Matplotlib colours (black for the first group) remain
visible. `set_theme('dark')` gives PyQtGraph's native look; `set_theme('app')` leaves PyQtGraph's
global options untouched for applications that manage their own theme.
"""),
code("""
plotqt.set_theme('dark')
win_dark = plotqt.plot_registration(result, group=speed, colors=['w','b','g','r'])
plotqt.to_image(win_dark, 'pg_dark.png', size=(1200, 360))
display(Image('pg_dark.png'))
plotqt.set_theme('light')
"""),
md("""
### 4. Interactive use

In a script or application the same objects are shown as live windows:

    win = result.plot(group=speed, backend='pyqtgraph')
    win.show()
    plotqt.app().exec()

The linked axes of the *Before* / *After* panels pan and zoom together, and PyQtGraph's context
menu (right click) offers export to PNG / SVG / CSV, which is the route to publication figures
from within an application.
"""),
code("""
for f in ('pg_registration.png', 'pg_panels.png', 'pg_dark.png'):
    os.remove(f)                       # the images are embedded in the notebook outputs
"""),
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


ALL = {
    '1-Registration'      : nb1,
    '2-Methods'           : nb2,
    '3-Warps'             : nb3,
    '4-Multivariate'      : nb4,
    'WarpCentering'       : nb_center,
    'Bayesian-vs-nlreg1d' : nb_bayes,
    'RealTimeRegistration': nb_rt,
    'PyQtGraph-backend'   : nb_qt,
}


if __name__ == '__main__':
    names = sys.argv[1:] or list(ALL)
    for name in names:
        build(ALL[name], name)
