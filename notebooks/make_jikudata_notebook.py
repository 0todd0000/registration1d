'''
Generate, execute and render notebooks/jiku-data-datasets.ipynb.

    python make_jikudata_notebook.py

Requires the jiku-data package (pip install jiku-data). Execution takes
roughly 30-45 minutes because every method is run on every dataset.
'''

import os, sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import numpy as np
import jikudata as jd

HERE = os.path.dirname(os.path.abspath(__file__))


def md(s):
    return nbf.v4.new_markdown_cell(s.strip('\n'))

def code(s):
    return nbf.v4.new_code_cell(s.strip('\n'))


INTRO = md('''
# jiku-data datasets

This notebook applies the registration methods of `registration1d` to every one-dimensional
dataset distributed with [jiku-data](https://pypi.org/project/jiku-data/) (`pip install jiku-data`;
GPLv3), i.e. all datasets with `dim == 1`, and uses each dataset to raise implications for two
open development questions:

1. **Automated or semi-automated parameter tuning** — which quantities could be estimated from
   the data to choose method parameters (or the method itself), and where such automation is
   *not* possible because the choice depends on the research question.
2. **Experimental design** — each dataset carries a design description (`dataset.params.testname`,
   `dataset.design`). Whether design information (groups, pairing, repeated measures, covariates)
   should enter the registration stage, and how it would interact with the subsequent analysis,
   is discussed but *not* implemented here: every registration below is design-blind.

Each dataset has its own level-2 section. Implications are reported only where a dataset raises
something not already raised by an earlier dataset; the many simulated `SPM1D_ANOVA*` datasets
therefore mostly refer back to the first of their kind.

### What each figure shows

The figure for each dataset has the layout of the comparison figure in notebook 2: the linearly
registered data (`linear only`) and the same data after each method, coloured by the design's
first factor (or by quartile of a continuous covariate). Settings are the package defaults,
except that iteration counts are reduced for datasets with more than 40 observations (marked
`fast` in the title) to keep the total run time reasonable, and pairwise synchronisation
(O(J²) alignments) is run only for J ≤ 20. Multivariate datasets are shown through their first
component: the univariate methods are run on that component alone, whereas the SRSF panel is
the *joint* (vector-SRSF) registration of all components. Landmark registration uses the
automatic global-minimum / global-maximum landmarks and is marked "not applicable" where an
extremum falls on the domain boundary.

The affine panel uses `cover=True` with edge fill (the zero fill of notebook 2 is specific to
data that start and end at zero, such as ground reaction forces).

Below every figure a short table reports, for each method, the residual sum of squares about the
cross-sectional mean relative to the linearly registered data (`SSE ratio`), the largest
displacement (`max |disp|`, fraction of the domain) and the run time. The SSE ratio is a
descriptive quantity only: as the featureless datasets below show, a flexible warp reduces it on
pure noise, so a small ratio is not evidence of a successful registration.
''')


SETUP = code('''
import sys, os, time, warnings, textwrap
sys.path.insert(0, os.path.abspath('../src'))   # so that this notebook finds registration1d without installation
import numpy as np
from matplotlib import pyplot as plt
import registration1d as reg1d
import jikudata as jd
warnings.filterwarnings('ignore')
print('registration1d', reg1d.__version__, '| jiku-data', jd.__version__)

COLORS = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'orange']

def load(name):
    """Return (y, group, group_label, dataset) with y a (J,Q) or (J,Q,D) float array."""
    d = getattr(jd, name)()
    y = np.asarray(d.y, dtype=float)
    x = np.asarray(d.x)
    J = y.shape[0]
    if x.ndim == 0:
        g, lab = np.zeros(J, int), 'single group'
    elif x.ndim == 2:
        g, lab = np.unique(x[:, 0], return_inverse=True)[1], 'factor A'
    elif np.issubdtype(x.dtype, np.floating) and np.unique(x).size > 8:
        g, lab = np.digitize(x, np.quantile(x, [.25, .5, .75])), 'covariate quartile'
    else:
        g, lab = np.unique(x, return_inverse=True)[1], 'group'
    return y, g, lab, d

def run_all(y):
    """Run every method; returns an ordered dict name -> (result or exception, seconds)."""
    J    = y.shape[0]
    y1   = y[:, :, 0] if y.ndim == 3 else y
    fast = J > 40
    mi   = 2 if fast else 5
    nb   = 4 if fast else 6
    out  = {}
    def run(name, fn):
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:
            r = e
        out[name] = (r, time.time() - t0)
    out['linear only'] = (None, 0.0)
    run('affine (cover)',          lambda: reg1d.register_affine(y1, cover=True, fill_value='edge', max_iter=mi))
    run('landmark (min, max)',     lambda: reg1d.register_landmark(y1, kinds=('min', 'max')))
    run('derivative DTW, smoothed',lambda: reg1d.register_dtw(y1, derivative=True, step_pattern='strict', smooth=0.03, max_iter=min(mi, 3)))
    run('continuous',              lambda: reg1d.register_continuous(y1, n_basis=nb, max_iter=mi))
    run('self-modelling',          lambda: reg1d.register_sim(y1, n_basis=nb, max_iter=mi))
    if J <= 20:
        run('pairwise (SRSF engine)', lambda: reg1d.register_pairwise(y1))
    else:
        run('SRSF, lam=0',         lambda: reg1d.register_srsf(y1, max_iter=5, lam=0, parallel=fast))
    run('SRSF' + (' (joint, multivariate)' if y.ndim == 3 else ''), lambda: reg1d.register_srsf(y, max_iter=5, parallel=fast))
    return out, fast

def show(name, y, g, lab, out, fast, d):
    y1  = y[:, :, 0] if y.ndim == 3 else y
    ug  = np.unique(g)
    fig, AX = plt.subplots(2, 4, figsize=(16, 7))
    sse0 = ((y1 - y1.mean(axis=0))**2).sum()
    rows = []
    for ax, (mname, (r, sec)) in zip(AX.ravel(), out.items()):
        if isinstance(r, Exception):
            ax.text(0.5, 0.5, 'not applicable:\\n' + textwrap.fill(str(r), 38), ha='center', va='center', transform=ax.transAxes, size=8)
            ax.set_title(mname, size=10); ax.set_xticks([]); ax.set_yticks([])
            rows.append((mname, None, None, sec)); continue
        yy = y1 if r is None else (r.y[:, :, 0] if r.y.ndim == 3 else r.y)
        reg1d.plot.plot_curves(yy, group=g, ax=ax, colors=COLORS, x='percent', legend=(r is None and ug.size > 1),
                               labels=[f'{lab} {u}' for u in ug] if ug.size > 1 else None, lw=0.7)
        ax.set_title(mname, size=10); ax.set_xlabel('domain (%)')
        sse = ((yy - yy.mean(axis=0))**2).sum()
        md_ = 0.0 if r is None else float(np.abs(r.warps.displacement()).max())
        rows.append((mname, sse / sse0, md_, sec))
    AX[0, 0].set_ylabel(name)
    fig.suptitle(f"{name}  (J={y.shape[0]}, Q={y.shape[1]}" + (f", D={y.shape[2]}" if y.ndim == 3 else '') +
                 f"; design: {d.design}, test: {d.params.testname}" + ('; fast settings' if fast else '') + ')', size=12)
    plt.tight_layout(); plt.show()
    print(f"{'method':28s} {'SSE ratio':>10s} {'max |disp|':>11s} {'seconds':>8s}")
    for mname, ratio, md_, sec in rows:
        print(f"{mname:28s} {'-' if ratio is None else f'{ratio:10.3f}':>10s} {'-' if md_ is None else f'{md_:11.3f}':>11s} {sec:8.1f}")

def analyse(name):
    y, g, lab, d = load(name)
    out, fast = run_all(y)
    show(name, y, g, lab, out, fast, d)
    return y, g, out
''')


# ---------------------------------------------------------------------------
# per-dataset commentary: heading text (implications); None = refer back
# ---------------------------------------------------------------------------

REFER_NESTED = 'No implications beyond those of **SPM1D_ANOVA2NESTED_2x2**.'
REFER_ONERM  = 'No implications beyond those of **SPM1D_ANOVA2NESTED_2x2** and **SPM1D_ANOVA2ONERM_2x2**.'
REFER_RM     = 'No implications beyond those of **SPM1D_ANOVA2NESTED_2x2** and **SPM1D_ANOVA2RM_2x2**.'
REFER_CROSS  = 'No implications beyond those of **SPM1D_ANOVA2NESTED_2x2** and **SPM1D_ANOVA2_2x2**.'

NOTES = {

'Besier2009muscleforces': '''
Ten knee-muscle forces during walking (D = 10), patellofemoral-pain patients versus controls
(two-sample Hotelling's T²). The forces span two orders of magnitude between muscles.

**Auto-tuning.** The joint (vector-SRSF) registration is dominated by the components with the
largest slopes, so with unequal scales the warps effectively register the two or three largest
muscles and ignore the rest. A data-driven default is to standardise each component (divide by
its pooled standard deviation, or by its median total variation) before computing the vector SRSF,
and to expose the per-component weights so that a user can favour the components that carry the
timing information. Whether equal weighting is *right* is a modelling choice, so the sensible
automation is "standardise by default, report the effective weight of each component" rather than
a hidden optimisation.

**Design.** With two groups and a pooled template, group differences in timing are absorbed by
the warps and removed from the registered amplitudes; with group-specific templates they would
instead remain in the amplitudes and vanish from the warps. Neither is neutral: the choice decides
which downstream test (amplitude or timing) can detect a group timing effect, so a design-aware
registration must be paired with a matching test rather than added silently.
''',

'Dorn2012': '''
Three-component ground reaction forces during running at four speeds (canonical correlation
analysis against speed). This is the development dataset of notebooks 1-4; see those for the
detailed findings (amplitude scaling across speeds, the double braking dip, real-time
registration, the multivariate compromise between components).

**Auto-tuning.** The `lam='auto'` default (median total variation) was derived on these data and
on simulated datasets A/B; the panel shows that it leaves the alignment essentially unchanged
relative to `lam=0` here. Speed-dependent stance duration is the case for which real-time
registration (`t='fs=...'`) rather than normalised time should be offered automatically whenever
the input is ragged.

**Design.** With a continuous covariate (speed) the natural downstream analysis is a regression of
displacement fields on speed. A design-aware alternative is to register to a *speed-conditional*
template (a smooth function of speed) rather than to the grand Karcher mean; that would remove the
speed trend in timing from the warps and put it into the template, which is appropriate only if
the timing trend is a nuisance rather than the effect of interest.
''',

'Dorn2012manova': '''
The same observations as **Dorn2012** with speed treated as a categorical factor (one-way MANOVA).

**Design.** Registration is identical to the regression case: the same warps serve both analyses.
This illustrates the general point that registration should not depend on the *test* to be run
afterwards, only (if at all) on the structure of the data; a registration that took the categorical
design into account (group templates) would give different warps for the same physical data
depending on how the analyst chose to code speed, which is undesirable.
''',

'Neptune1999kneekin': '''
Three-dimensional knee kinematics (D = 3), eight subjects in two conditions (paired Hotelling's
T²).

**Auto-tuning.** The curves share one dominant valley, so all methods agree; the differences lie in
how much of the between-subject offset in the flanks is treated as timing. This is a dataset on
which an identifiability diagnostic (the local magnitude of the SRSF relative to the noise level,
flagging regions where the warp is not determined by the data) would be informative, because the
flanks are nearly linear and the warps there are decided by the penalty rather than by features.

**Design.** A paired design suggests registering *within pairs* (the two conditions of a subject
to a subject-specific template, or one condition to the other) so that the warp of each pair
directly measures the within-subject timing effect, and subject-level timing idiosyncrasies never
enter the between-subject variance. That is a strictly stronger use of the design than a pooled
template and would change the downstream paired test on displacement fields into a one-sample test
on within-pair warps.
''',

'Pataky2014cop': '''
Centre-of-pressure trajectories (D = 2: mediolateral and anteroposterior) for two walking
conditions (paired Hotelling's T²).

**Auto-tuning.** These are planar *curves* parametrised by time, not two independent functions.
The joint vector-SRSF registration is the right object mathematically (it is the square-root
velocity function of the curve), but for curves one may want *shape* alignment that also removes
translation, rotation and scaling before registration; whether the coordinate frame is part of
the signal (as here: the foot's position on the plate is meaningful) or a nuisance is a modelling
decision that no data-driven rule can make. The implication for tuning is that multivariate inputs
should carry a flag distinguishing "several signals on a common time base" from "coordinates of one
curve".
''',

'PlantarArchAngle': '''
Plantar arch angle during stance for two conditions (two-sample t-test). The curves are almost
monotonic with a single sharp drop near the end of stance and no interior extrema.

**Auto-tuning.** Automatic extremum landmarks fail (the global minimum and maximum sit on the
domain boundary), and every elastic method has essentially one feature to align, the drop. The
SSE reduction is therefore small and mostly comes from warping the nearly linear middle section,
which is not identified by the data. A useful automatic screen is the number of interior extrema
per curve (here zero for most curves): with fewer than two, landmark registration should default to
slope-based events (e.g. the time of the steepest descent) and the elastic penalty should be
increased, since the warps are otherwise governed by noise.
''',

'Random': '''
Ten realisations of a smooth Gaussian random field (one-sample t-test), i.e. curves with no
common features at all.

**Auto-tuning.** Every method still "registers" these curves and reduces the residual sum of
squares, because a flexible warp can always move bumps of one realisation onto bumps of another.
This is the central hazard of automatic registration: an objective-function improvement is not
evidence of a real timing structure. A registrability test is needed before any tuning — for
example, compare the SRSF cost reduction obtained on the data with the reduction obtained on
surrogate data with the same smoothness but no common structure (phase-randomised or
time-reversed-and-shuffled copies); if the observed reduction is not larger than the surrogate
distribution, registration should be declined rather than tuned. See the additional figure under
**SPM1D_ANOVA2NESTED_2x2** for the size of this effect.

**Auto-tuning: the SSE ratio is not a tuning criterion.** The continuous (Ramsay-Li) method
reaches the smallest SSE ratio on these noise curves, with displacements of half the domain: its
low-dimensional warps *pinch* all curves towards common crossing points (visible as nodes in the
`continuous` panel) rather than align features, and the self-modelling method, which fits an
amplitude model as well, can even increase the residual. Any automatic rule that selects the
penalty or the number of basis functions by minimising the residual will therefore select the
most distorting warps; a tuning criterion must trade the residual against a warp-roughness or
displacement penalty (as `lam` does for SRSF) or be validated on held-out curves, and the
continuous method needs a data-driven `lam` of the same kind as the SRSF `lam='auto'` rule.
Pairwise synchronisation is the most conservative method on featureless data (SSE ratio close to
one on all the random-field datasets), because averaging the pairwise warps cancels the
observation-specific noise matches; that robustness is a reason to keep it as a reference despite
its O(J²) cost.
''',

'SPM1D_ANOVA2NESTED_2x2': '''
Simulated smooth random fields with a nested two-factor design (the first of 38 simulated
`SPM1D_ANOVA*` datasets, all of which are random fields without common features; the sections for
the others refer back here).

**Auto-tuning: registration of featureless data.** The additional figure below quantifies the
hazard raised under **Random**: SRSF registration removes 20-40 % of the residual sum of squares
of pure noise fields, most for rough fields (which offer more bumps to match) and, at every
smoothness, at least as much as it removes from the real SpeedGRF curves. A residual-based
criterion therefore cannot even distinguish data with features from data without. The
permutation/surrogate registrability test proposed under **Random** would reject registration for
all 38 simulated datasets; so would a simpler screen, the absence of extrema whose positions are
consistent across observations (e.g. the standard deviation of the global-maximum time exceeding a
sizeable fraction of the domain). Both are cheap and should run before any automatic tuning.

**A finding about the package.** The first run of this notebook used the affine panel to expose
a bug: on featureless data the local refinement of `register_affine` left the `scale_range` /
`max_shift` search box and shrank curves by factors of up to 100 (the residual of a curve squeezed
to a point and padded is trivially small). The refinement is now confined to the search box (and
a test added). Featureless data are a useful stress test for every method's constraints and
should be part of the test suite for any automatic tuning rule.

**Design.** In a nested design (factor B nested in A: e.g. subjects within groups) observations
of different nesting units are not exchangeable. Registering everything to one template mixes
unit-level timing into the residuals of the nested test; registering within units (unit-specific
templates, then aligning the unit templates) respects the nesting but produces warps at two
levels. Which level's warps carry the effect of interest depends on the hypothesis, so the design
can *structure* the registration but cannot choose it.
''',

'SPM1D_ANOVA2ONERM_2x2': '''
Simulated random fields with one repeated-measures factor.

**Design.** A repeated-measures factor identifies observations from the same subject. The
natural design-aware registration is hierarchical: a subject-level warp (the subject's mean
timing) composed with a condition-level warp. Registering to a pooled template instead
attributes the subject-level timing to the residual of every condition, inflating the error term
of the repeated-measures test exactly as unmodelled subject effects do in the amplitude analysis.
This is the strongest case for using design information at the registration stage — but the same
subject warps must then be treated as random effects in the timing analysis, or the test on
displacement fields will be anti-conservative.
''',

'SPM1D_ANOVA2RM_2x2': '''
Simulated random fields with two repeated-measures factors.

**Design.** With both factors within subjects the hierarchical decomposition of **SPM1D_ANOVA2ONERM_2x2**
has three levels (subject, factor A, factor B) and an interaction term in the warp group; the
composition of warps is not commutative, so the *order* in which effects are removed becomes a
modelling choice with no counterpart in amplitude ANOVA. This is the main reason for not
implementing design-dependent registration before the corresponding timing model is specified.
''',

'SPM1D_ANOVA2_2x2': '''
Simulated random fields with two crossed between-subject factors.

**Design.** Crossed factors raise the question of cell-wise versus marginal templates: a
template per cell removes the interaction in timing from the warps, marginal templates remove
the main effects only. Any such choice pre-empts the interaction test on displacement fields, so
if design information were used here it would have to be recorded in the result (which effects the
registration has already absorbed) for the downstream analysis to remain interpretable.
''',

'SimulatedPataky2015a': '''
Ten simulated curves with a single smooth signal plus smooth noise (one-sample t-test).

**Auto-tuning.** With J = 10 the template is an average of very few curves and the Karcher
iteration can be pulled towards one observation. A leave-one-out stability check (re-register
without each observation and measure how much the warps of the others change) is an inexpensive
automatic diagnostic of template dominance at small sample sizes, and its output (per-observation
influence) is exactly what an interactive tool should show.
''',

'SimulatedPataky2015b': '''
As **SimulatedPataky2015a** with a different noise realisation; no additional implications.
''',

'SimulatedPataky2015c': '''
Ten simulated curves whose single peak varies with a continuous covariate (linear regression).

**Auto-tuning.** Landmark registration on the automatic global extrema increases the residual
almost six-fold here: the curves are flat apart from the signal, so the global minimum and maximum
of each curve are noise extrema at unrelated positions, and forcing them to coincide destroys the
alignment. An automatic landmark rule must therefore test the consistency of the detected event
times across observations (their dispersion relative to the domain) and fall back to no landmarks
when it fails; prominence thresholds alone do not protect against this.

**Design.** The effect of interest is a regression of the curves on the covariate; if the
covariate also shifts the peak in time, registration moves that effect from the amplitude domain
into the warps. Whether that is desirable cannot be decided from the data: it is the difference
between asking "does the amplitude at a fixed time depend on x?" and "does the timing depend on
x?". Design-aware registration (a covariate-dependent template) would make the amplitude
regression blind to the timing effect by construction. The implication is that for regression
designs the software should report *both* analyses (amplitude after registration, displacement
fields against the covariate) rather than choose.
''',

'SimulatedTwoLocalMax': '''
Two groups of curves with two local maxima whose relative heights differ between groups
(two-sample t-test).

**Auto-tuning.** This is the peak-correspondence problem: when the dominant peak switches between
groups, global-extremum landmarks pair the wrong peaks and elastic methods may align peak 1 of one
group with peak 2 of the other. No objective-function criterion distinguishes the two solutions;
correspondence must come from prior knowledge (which peak is which) or from a landmark table that
a user can edit. The automatic detection should therefore return *all* prominent peaks with
their prominences (as `peaks_as_landmarks` does) and flag observations whose peak count differs
from the majority, instead of silently taking the global maximum.
''',

'SmallSampleLargePosNegEffects': '''
Eight curves in two groups with large opposite-sign differences (two-sample t-test).

**Design.** With J = 8 and strong group differences the pooled template lies between two quite
different shapes and represents neither group; the warps then partly encode "which group am I
in", and the two-sample test on displacement fields becomes a test of the template's
compromise rather than of timing. This is the situation in which group-specific templates are
most tempting and most dangerous (they remove exactly the effect under test). The safest
design-aware behaviour is diagnostic: report the distance of each group's mean from the template
and warn when the groups are far apart relative to their spread.
''',

'SpeedGRF': '''
Vertical ground reaction forces at 60 walking speeds (linear regression), sharply defined double
hump.

**Auto-tuning.** J = 60 curves with clear, consistent features: the method rankings are stable,
and this is the kind of data on which automatic tuning is meaningful. The tunable quantities are
few and observable: the elasticity penalty (`lam='auto'` here is dominated by the loading/unloading
slopes and leaves the warps small), the number of continuous-registration basis functions
(choose by the residual plateau), and the DTW smoothing width (choose as a multiple of the
correlation length of the curves, estimated from their autocorrelation). All three could be set
from the data with a one-line rule and reported.
''',

'SpeedGRFcategorical': '''
The **SpeedGRF** curves with speed coded as three categories (one-way ANOVA).

**Design.** Unbalanced or unequal-variance groups bias a pooled template towards the larger or
more homogeneous group; group-balanced templates (the Karcher mean of group means) remove that
bias without removing group timing differences from the warps, and are the one design-aware
choice that does *not* pre-empt any downstream test. This is the minimal, defensible use of design
information at the registration stage.
''',

'SpeedGRFcategoricalRM': '''
The categorical speed design with subjects as a repeated-measures factor (one-way RM ANOVA).

No implications beyond those of **SpeedGRFcategorical** and **SPM1D_ANOVA2ONERM_2x2**; this is
the real-data case for the hierarchical (subject × condition) warp decomposition.
''',

'Weather': '''
Daily mean temperature over the year at 35 Canadian weather stations in four climate regions
(one-way ANOVA), Q = 365.

**Auto-tuning.** The `continuous` panel shows the pinching artefact described under **Random**
on real data: with six basis functions and the default penalty, every station is forced
through two common crossing points. The domain is periodic: 31 December is adjacent to 1 January, but every method
here fixes the warp end points and cannot rotate the year. For periodic data the first
registration step should be a circular shift (a shift registration on the circle), and the
elastic warp should be applied to the shifted curves with the seam placed at a data-chosen
position (the coldest day, say). Periodicity is detectable automatically (the first and last
values agree to within the noise level for every curve), so this is a case where the *type* of
registration, not only its parameters, can be chosen from the data. Q = 365 also makes the
dynamic programme about ten times slower than at Q = 101; automatic down-sampling to a coarser
grid for the alignment, followed by evaluation of the warp on the full grid, would be a natural
speed rule.
''',
}

# families of simulated random fields: refer back
for n in ['SPM1D_ANOVA2NESTED_2x3', 'SPM1D_ANOVA2NESTED_3x3', 'SPM1D_ANOVA2NESTED_3x4', 'SPM1D_ANOVA2NESTED_3x5',
          'SPM1D_ANOVA2NESTED_4x4', 'SPM1D_ANOVA2NESTED_4x5', 'SPM1D_ANOVA3NESTED_2x2x2', 'SPM1D_ANOVA3NESTED_2x4x2']:
    NOTES[n] = REFER_NESTED
for n in ['SPM1D_ANOVA2ONERM_2x3', 'SPM1D_ANOVA2ONERM_3x3', 'SPM1D_ANOVA2ONERM_3x4', 'SPM1D_ANOVA2ONERM_3x5',
          'SPM1D_ANOVA2ONERM_4x4', 'SPM1D_ANOVA2ONERM_4x5', 'SPM1D_ANOVA3ONERM_2x2x2', 'SPM1D_ANOVA3ONERM_2x3x4']:
    NOTES[n] = REFER_ONERM
for n in ['SPM1D_ANOVA2RM_2x3', 'SPM1D_ANOVA2RM_3x3', 'SPM1D_ANOVA2RM_3x4', 'SPM1D_ANOVA2RM_3x5', 'SPM1D_ANOVA2RM_4x4',
          'SPM1D_ANOVA2RM_4x5', 'SPM1D_ANOVA3RM_2x2x2', 'SPM1D_ANOVA3RM_2x3x4', 'SPM1D_ANOVA3TWORM_2x2x2', 'SPM1D_ANOVA3TWORM_2x3x4']:
    NOTES[n] = REFER_RM
for n in ['SPM1D_ANOVA2_2x3', 'SPM1D_ANOVA2_3x3', 'SPM1D_ANOVA2_3x4', 'SPM1D_ANOVA2_3x5', 'SPM1D_ANOVA2_4x4', 'SPM1D_ANOVA2_4x5',
          'SPM1D_ANOVA3_2x2x2', 'SPM1D_ANOVA3_2x3x4']:
    NOTES[n] = REFER_CROSS


# extra figure for the featureless-data hazard (attached to SPM1D_ANOVA2NESTED_2x2)
EXTRA_REGISTRABILITY = code('''
# Registration of featureless data: SRSF SSE reduction on smooth Gaussian random fields of
# increasing smoothness (FWHM), with no common structure, versus the reduction on SpeedGRF.
from scipy.ndimage import gaussian_filter1d
rng = np.random.default_rng(0)
J, Q = 20, 101
fwhms = [5, 10, 20, 40]
ratios = []
for fwhm in fwhms:
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    noise = gaussian_filter1d(rng.standard_normal((J, 3 * Q)), sigma, axis=1)[:, Q:2*Q]
    noise /= noise.std()
    r = reg1d.register_srsf(noise, max_iter=5)
    ratios.append(r.sse()[1] / r.sse()[0])
y_grf = load('SpeedGRF')[0]
r_grf = reg1d.register_srsf(y_grf[:J], max_iter=5)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(fwhms, ratios, 'ko-', label='random fields (no common features)')
ax.axhline(r_grf.sse()[1] / r_grf.sse()[0], color='r', ls='--', label='SpeedGRF (real features)')
ax.set_xlabel('smoothness of the random field (FWHM, % of domain)')
ax.set_ylabel('SSE after / before SRSF registration')
ax.set_ylim(0, 1); ax.legend(); ax.set_title('SRSF registration reduces the residuals of pure noise')
plt.show()
print('SSE ratios for random fields:', np.round(ratios, 3), '| SpeedGRF:', round(r_grf.sse()[1] / r_grf.sse()[0], 3))
''')


def _shrink_images(nb, colors=64):
    """Re-encode the embedded PNGs as 64-colour palette images (55 figures with
    hundreds of thin lines each would otherwise make the notebook ~45 MB)."""
    import base64, io
    from PIL import Image
    for c in nb['cells']:
        for o in c.get('outputs', []):
            d = o.get('data', {})
            if 'image/png' in d:
                im  = Image.open(io.BytesIO(base64.b64decode(d['image/png']))).convert('RGB')
                im  = im.quantize(colors=colors, method=Image.MEDIANCUT)
                bio = io.BytesIO(); im.save(bio, 'PNG', optimize=True)
                d['image/png'] = base64.b64encode(bio.getvalue()).decode('ascii')


def build():
    names = [ (d() if callable(d) else d).name  for d in jd.get_datasets(dim=1) ]
    cells = [INTRO, SETUP]
    cells.append(md('### Datasets\n\n' + ', '.join(f'`{n}`' for n in names)))
    for n in names:
        note = NOTES.get(n, '(no commentary)')
        cells.append(md(f'## {n}\n\n{note.strip()}'))
        cells.append(code(f"y, g, out = analyse('{n}')"))
        if n == 'SPM1D_ANOVA2NESTED_2x2':
            cells.append(EXTRA_REGISTRABILITY)
    cells.append(md('''
## Summary of implications

**Automated parameter tuning** cannot be driven by the residual alone: on every one of the 38
featureless simulated datasets the residual falls under every elastic method, most under the
method with the most distorting warps. Tuning is realistic for a small set of observable quantities — the
elasticity penalty (already `lam='auto'`), the DTW smoothing width (from the curves' correlation
length), the number of parametric basis functions (from a residual plateau), per-component
standardisation for multivariate input, periodicity detection, and real-time versus normalised
time for ragged input — and should always be preceded by a *registrability* screen (surrogate
comparison or extremum-consistency check), because every method improves its objective on pure
noise. Choices that depend on the research question (which peak corresponds to which, whether a
covariate's timing effect is signal or nuisance, whether coordinates are a curve or separate
signals) cannot be automated and should be exposed as explicit, recorded decisions.

**Design information** can enter registration in three ways with increasing risk: (1)
group-balanced templates, which remove sample-composition bias without removing any effect
(safe); (2) hierarchical or paired templates that respect nesting, pairing or repeated measures,
which put subject-level timing where it belongs but require the matching mixed-effects timing
analysis; (3) condition- or covariate-specific templates, which move the effect under test out of
one domain (amplitude or timing) into the other and therefore pre-empt the analysis. Because the
registration determines what the subsequent tests can see, design-dependent registration should
be implemented, if at all, together with the analysis it is meant to serve (as in joint
registration-and-analysis models), and the result object should record which effects the
registration has absorbed.
'''))
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata']['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': HERE}})
    _shrink_images(nb)
    fpath = os.path.join(HERE, 'jiku-data-datasets.ipynb')
    nbf.write(nb, fpath)
    html, _ = HTMLExporter().from_notebook_node(nb)
    with open(os.path.join(HERE, 'html', 'jiku-data-datasets.html'), 'w') as f:
        f.write(html)
    print('wrote', fpath)


if __name__ == '__main__':
    build()
