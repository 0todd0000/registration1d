# registration1d — summary of findings and suggestions for further development

Date: 2026-09-29. Status: preliminary package (v0.0.2), 45 tests,
ten executed notebooks; source moved to the src layout for the public repository.

<span style="color:#ffd400">**Highlighted text (yellow) marks additions and changes made after the review
of the first version (2026-09-27, afternoon and evening).**</span>

## 1. What was built

A GPL-3.0-or-later Python package `registration1d` that depends only on numpy, scipy
and matplotlib and reproduces the workflow of `nlreg1d/Notebooks/3-Registration.ipynb`
(linear interpolation to 101 points followed by SRSF registration of the
Dorn2012 GRF data) **without fdasrsf or scikit-fda installed**. The package
contains:

- two linear methods (`register_shift`, `register_affine`) plus
  `register_linear` (interpolation, as in nlreg1d);
- <span style="color:#ffd400">seven</span> nonlinear methods: `register_srsf` (elastic / Fisher-Rao),
  `register_dtw`, `register_landmark`, `register_continuous` (Ramsay-Li),
  <span style="color:#ffd400">`register_sim` (self-modelling / shape-invariant model),
  `register_pairwise` (pairwise synchronisation), `register_bayes` (Bayesian
  registration with posterior warp samples)</span>;
- a warp toolkit (`reg1d.warp`: apply, compose, invert, displacement fields,
  Karcher means, centring, random warps, <span style="color:#ffd400">monotonicity-preserving warp
  smoothing</span>, `Warp1D` / `Warp1DList`);
- elastic amplitude and phase distances;
- a `RegistrationResult` object that unpacks as `(yr, wf)` for nlreg1d-style
  code and carries the template, warps and diagnostics <span style="color:#ffd400">— now returned by
  *every* `register_*` function including `register_linear`, as the
  subclasses `LinearRegistrationResult` / `NonlinearRegistrationResult` with an
  `islinear` attribute, and with `apply` / `unapply` / `inverse_warps` and an
  optional explicit time grid `t`</span>;
- <span style="color:#ffd400">two-sample t and permutation (max-t) inference and a
  `timing_test` helper that reproduces the nlreg1d amplitude + timing analysis
  (moved out of the package to `notebooks/util.py` on 2026-09-29)</span>;
- the Dorn2012 dataset <span style="color:#ffd400">(reduced AND three-component), the simulated
  datasets A and B</span>, six notebooks (`notebooks/*.ipynb` + `html/`),
  a test suite (`tests/`, <span style="color:#ffd400">described in `TESTS.md`</span>), `requirements.txt`, `pyproject.toml`.

Layout:

    registration1d/            package  (all code space-indented)
      warp.py         warps and warp algebra, smooth_warp
      srsf.py         SRSF transform (uni- and multivariate), dynamic programming (banded, penalised),
                      gradient refinement, Karcher mean / median alignment, parallel option, distances
      dtw.py          dynamic time warping, derivative DTW, DBA template, step patterns
      landmark.py     landmark registration and simple landmark detection
      continuous.py   Ramsay-Li penalised least-squares registration
      sim.py          self-modelling (shape-invariant model) registration
      pairwise.py     pairwise synchronisation
      bayes.py        Bayesian registration (pCN sampler)
      realtime.py     real-time registration of observations of different lengths
      linear.py       resampling, shift and affine registration (fill rules, cover constraint)
      reg.py          public register_* functions and result classes
      plot.py, plotqt.py (optional PyQtGraph backend)
    notebooks/        util.py (example datasets, permutation tests), data/*.npz, data/*.csv
                      1-Registration, 2-Methods, 3-Warps, 4-Multivariate, WarpCenteringIssue,
                      Bayesian-vs-nlreg1d, RealTimeRegistration (+ html/, make_notebooks.py)
                      (WarpCentering replaces the earlier WarpCenteringIssue)
                      PyQtGraph-backend; jiku-data-datasets (all 54 one-dimensional jiku-data
                      datasets, make_jikudata_notebook.py); power_simulated_datasets
                      (false-positive rate and power after registration, power1d
                      simulations, make_power_notebook.py + cached results .pkl)
    tests/            pytest suite (45 tests)
    TESTS.md          test strategy;  PAPER-IDEAS.md  publication ideas

## 2. Key findings

### 2.1 SRSF registration can be implemented from the mathematics in ~300 lines and matches fdasrsf

The dynamic-programming alignment of SRSFs is the heart of fdasrsf's
`srsf_align`. Re-implemented from Srivastava et al. (2011) with a vectorised
row-by-row recursion, it runs in about 0.05 s per pairwise alignment for
Q = 101 (max_step = 6, i.e. local slopes 1/6 ... 6) with no compiled code.
Group registration of the 8 Dorn2012 curves with 5 template updates takes
about 2 s. On the Dorn2012 data the warps agree with fdasrsf 2.7.2 to within
about 0.01 on the unit interval for 7 of 8 observations (0.06 for one, in
the region of the double braking dip where the objective is nearly flat),
and the registered curves are visually indistinguishable from the figure in
the nlreg1d notebook.

Two deliberate differences from fdasrsf: (i) the SRSF derivative is
`numpy.gradient` by default <span style="color:#ffd400">(a smoothing-spline derivative is now available
with `smooth='spline'`)</span>; (ii) the DP slope set is explicit and user-controllable (`max_step`).

<span style="color:#ffd400">**New (2.1a): the elasticity penalty matters for noisy, flat data.** On
the simulated dataset A (pure amplitude effect; flat, noisy tails) the
unpenalised SRSF registration produces large, noise-driven warps in the
flat regions — with registration1d *and* with fdasrsf (the two differ by up to 0.17
there, because the objective is nearly indifferent) — and the nlreg1d
timing test then reports a spurious timing effect near the start of the
domain (p = 0.002 with registration1d, p = 0.08 with fdasrsf: decided by algorithmic
detail, not by data). With the fdasrsf-style penalty `lam = 100` (its scale
is that of the squared SRSF distance, which for these data is of order 100)
both simulated datasets behave exactly as intended: A amplitude-only, B
timing-only. Recommendation: report `lam` with every SRSF analysis, and
choose it by inspecting the displacement fields in regions known to be
flat. Details in `Bayesian-vs-nlreg1d.ipynb`, section 1.</span>

### 2.2 Licensing: nothing stands in the way of GPLv3

All runtime dependencies are BSD/PSF licensed. fdasrsf (BSD-3) and
scikit-fda (BSD-3) *could* have been imported by a GPL package; dtw-python is
itself GPL-3.0-or-later. The decision to re-derive rather than wrap was
therefore not forced by licensing but by the wish to avoid fdasrsf's heavy
dependency chain (Cython, numba, cffi, joblib, patsy, ...) and its
compilation step. Details are in `GPL-COMPLIANCE.md` <span style="color:#ffd400">(unchanged by the new
methods: all were written from the mathematics; no new imports)</span>.

### 2.3 Suitability of the methods for the Dorn2012 data (notebook 2)

| method | verdict for these data |
|---|---|
| SRSF | best general-purpose choice; amplitude-invariant; proper diffeomorphic warps; aligns peak and zero crossing well, braking dips reasonably |
| landmark | exact alignment of chosen events (zero crossing, propulsive peak); the double braking dip defeats simple automatic landmark detection, so landmarks should be supplied or checked by hand |
| DTW | fast but amplitude-driven: with peak GRF doubling across speeds it matches amplitudes rather than shapes, producing staircase warps and flat (non-invertible) segments; <span style="color:#ffd400">**derivative DTW + slope-constrained step pattern + warp smoothing** (`derivative=True, step_pattern='strict', smooth=0.03`) removes the artefacts and gives physically plausible warps with continuous first derivatives (notebook 2, "Smoother, physically plausible DTW warps")</span> |
| continuous (Ramsay-Li) | smooth low-dimensional warps; good on Dorn2012 with 8 cosine terms (aligns both braking dips), but the objective is non-convex and local minima do occur in simulation; needs the penalty (default lam = 0.01) |
| <span style="color:#ffd400">self-modelling (sim)</span> | <span style="color:#ffd400">models amplitude explicitly (a_i, b_i), smooth warps; sensible on Dorn2012; same local-minimum caveat as the continuous method</span> |
| <span style="color:#ffd400">pairwise synchronisation</span> | <span style="color:#ffd400">template-free; with the SRSF engine gives results close to SRSF; J(J-1)/2 alignments</span> |
| <span style="color:#ffd400">Bayesian</span> | <span style="color:#ffd400">posterior-mean warps at least as accurate as the DP warps in simulation, plus credible bands; ~100x slower than DP; credible intervals optimistic (see 2.5)</span> |
| shift / affine | only global timing; little value here beyond a baseline. <span style="color:#ffd400">The Nelder-Mead refinement in `affine_pair` is confined to the `scale_range` / `max_shift` search box (found on featureless jiku-data fields, where the scale drifted to ~100; `test_affine_respects_search_box`).</span> <span style="color:#ffd400">**End-point effects** (cut-off ends when a + b < 1, edge-value padding) are removed by `cover=True` (γ([0,1]) ⊇ [0,1], so both zero ends are retained) together with `fill_value='zero'`; `'extrapolate'` is also available (notebook 2)</span> |

The scikit-fda methods that were considered are all represented: its
`ElasticRegistration` is SRSF (via fdasrsf), `LeastSquaresShiftRegistration`
is `register_shift`, `landmark_shift_registration` / `landmark_elastic_registration`
are `register_landmark` (with linear / PCHIP warps).

### <span style="color:#ffd400">2.4 Multivariate registration (notebook 4)</span>

<span style="color:#ffd400">The three-component Dorn2012 forces (18 trials) register jointly with the
vector SRSF q = f'/sqrt(||f'||) — one warp per trial — in about 10 s.
The joint warp is a compromise between components: it aligns the
anteroposterior peak much better than linear registration but the
vertical rising edge less well than vertical-only-driven warps. The
vertical force dominates the joint objective because the vector SRSF
weights components by their slopes; rescaling components before
registration changes that weighting. `result.apply(z)` applies one set of
warps to any other (J,Q) or (J,Q,D) variable measured on the same time base.</span>

### <span style="color:#ffd400">2.5 Bayesian registration and its relation to the nlreg1d timing test (notebook Bayesian-vs-nlreg1d)</span>

<span style="color:#ffd400">`reg1d.bayes` implements a simplified Cheng–Dryden–Huang / Lu–Herbei–Kurtek
model: tangent-space cosine basis at the identity warp, Gaussian error model
in SRSF space, pCN Metropolis sampling with step-size adaptation, Gibbs
update of the noise variance, DP initialisation. Findings: (i) the
posterior-mean warp is at least as accurate as the DP warp in simulation;
(ii) on datasets A and B the conclusions of the nlreg1d timing test are
stable across posterior draws of the warps (posterior-predictive
propagation), i.e. registration uncertainty does not change the conclusion
when the features are sharp; (iii) the credible bands widen in flat
regions, exactly where the point-estimate warps are unreliable (2.1a);
(iv) the credible intervals of this simple model are optimistic (coverage
~40 % at nominal 95 % in simulation) because SRSF residuals are strongly
autocorrelated while the likelihood treats the Q grid values as
independent; the `n_eff` argument tempers the likelihood but a principled
fix (whitened or Gaussian-process error model) is needed before the bands
are used quantitatively; (v) the natural next step is a hierarchical model
with group-level warp distributions, which would be a direct Bayesian
counterpart of the nlreg1d test (see PAPER-IDEAS.md, idea 2).</span>

### <span style="color:#ffd400">2.5a Real-time registration (notebook RealTimeRegistration)</span>

<span style="color:#ffd400">All nonlinear procedures in notebooks 1-2 ran on linearly registered
(normalised-time) data, as in nlreg1d. Normalisation rescales each
observation's time axis by its own duration, so first derivatives -- and
the SRSFs built from them -- are expressed in different physical time
units for observations of different durations (a factor of two across the
Dorn2012 speeds). `reg1d.realtime` now registers observations of different
lengths on their OWN grids: SRSF derivatives in physical time, dynamic
programming between each observation's grid and a reference grid (n_ref
points over the mean duration; generalised `align_srsf_pair` with dt1 !=
dt2 and Q1 != Q2), warps Gamma_i mapping reference seconds onto observation
seconds (`info['warps_realtime']`), and real-time displacement fields
defined as the deviation from the pure linear rescaling. Ragged input
(`list` of arrays plus `t=` as a sampling interval, 'fs=<Hz>' or time
vectors) triggers it in `register_srsf` (uni- and multivariate),
`register_dtw` (derivatives divided by the sampling interval) and
`register_landmark` (landmarks in seconds). On Dorn2012 the real-time and
normalised-time SRSF warps differ by at most 0.03 of the domain (most for
the trials whose duration is furthest from the mean), i.e. the classical
workflow is not badly wrong for these data, but real-time registration
keeps loading rates in physical units and separates "shorter trial" from
"earlier feature" in the displacement fields, which the normalised
workflow cannot. Slope-constrained DTW step patterns cannot bridge large
length ratios (e.g. 383 points versus a 101-point reference); a clear
error message says so.</span>

### <span style="color:#ffd400">2.5b PyQtGraph backend (notebook PyQtGraph-backend)</span>

<span style="color:#ffd400">`registration1d.plotqt` mirrors `registration1d.plot` on PyQtGraph
(MIT; optional `[qt]` extra, a Qt binding chosen by the user). Every
function draws into a caller-supplied PlotItem / PlotWidget /
GraphicsLayoutWidget or creates a stand-alone widget; no event loop is
started; `to_image` renders off-screen for notebooks and tests;
`set_theme` gives a Matplotlib-like light theme by default, PyQtGraph's
dark theme, or leaves an application's own theme alone;
`RegistrationResult.plot(backend='pyqtgraph')` is the shortcut. The
core package never imports Qt.</span>

### <span style="color:#ffd400">2.6 Speed: numba estimate</span>

<span style="color:#ffd400">Measured on this machine (2 cores), Q = 101, max_step = 6: the pure-numpy
pairwise DP takes 0.048 s; a numba `@njit` version of the same DP (segment
costs and recursion in one compiled loop) takes 0.016 s after a 1.3 s
compile — a **3x speed-up** (2.9x at Q = 201, 2.1x at Q = 501). The gain is
modest because the numpy version is already vectorised over one grid axis
and the segment-cost arrays. Projected group registration times (5
template iterations, one core): J = 100: ~25 s numpy / ~8 s numba;
J = 1000: ~4 min numpy / ~80 s numba; with `parallel=True` these divide by
the number of cores (the alignments are independent), which is the larger
lever. Readability: numba can be confined to one function
(`_dp_kernel(q1, q2, steps, nsub, lam)`) decorated with `@njit`, with a
pure-Python fallback when numba is not installed
(`try: from numba import njit  except ImportError: njit = lambda f: f`),
so the rest of `srsf.py` stays as it is; the compiled kernel is ~40 lines
of plain loops, arguably *more* readable than the vectorised numpy version.
Recommendation: not now; add as an optional accelerator if J ~ 1000 becomes
routine. numba is BSD-2 licensed (GPL-compatible).</span>

### <span style="color:#ffd400">2.6a Registration introduces regularity into noise (notebooks jiku-data-datasets, power_simulated_datasets)</span>

<span style="color:#ffd400">On the 38 featureless jiku-data random-field datasets every elastic
method reduced the residual; in controlled power1d simulations (smooth Gaussian noise,
FWHM 25, one-sample max-t test) the false-positive rate at the unregistered α = 0.05
threshold after SRSF registration rose from 0.15 (J = 5) to 0.75 (J = 10) and 1.00 (J = 50);
`lam='auto'` does not change this. Continuous registration inflates the maximum t most
(calibrated threshold 34 at J = 5) but plateaus near 0.5; the self-modelling method stays at
nominal. With thresholds calibrated per method no method beat the unregistered test in any
one-sample pulse condition (base: 0.62 unregistered vs 0.20–0.50 registered), the methods
converge only as amplitude or J grows, and SRSF/DTW inflate the recovered amplitude by 30–40 %
at unit amplitude. Consequences: a registrability screen before registration, a
null-calibrated (permutation-with-registration) test in `notebooks/util.py`, and a noise-based rather
than signal-based rule for `lam`. Random warps of the signal instead of position shifts
(Part C, 30 datasets per condition) do not change the one-sample picture; in a two-sample
design with a true group timing difference (Part D, 30 datasets per condition) registration
moves the effect from the registered amplitudes (power 0.30 → 0.03-0.07) into the warps, where
the displacement-field test finds it (SRSF 0.5, derivative DTW 0.8). Parts C and D are
approximate and refinable overnight with `make_power_notebook.py --simulate --n-two 200 ...`.</span>

### <span style="color:#ffd400">2.6b Package scope (2026-09-29)</span>

<span style="color:#ffd400">Standing rule: the package contains registration algorithms only.
Moved out to `notebooks/util.py` + `notebooks/data/`: `stats.py` (two-sample t, permutation
max-t inference, `timing_test`: simulation / demonstration statistics) and `data.py` with the
bundled datasets (demonstration data). The tests load the Dorn2012 and SimulatedA files
directly from `notebooks/data/`; the permutation-test test was dropped with the module.
Kept, with reasons: `warp.random_warp` (warp machinery, used by the tests to construct
ground truth and by the simulations); `landmark.detect_landmarks` / `peaks_as_landmarks`
(initial-guess front end for landmark registration, requested earlier);
`srsf.amplitude_distance` / `phase_distance` (elastic metrics, part of the SRSF method);
`realtime.py` (registration of ragged input). Flagged but kept pending a decision:
`plot.py` and `plotqt.py` are not registration algorithms; they back `result.plot()` and the
PyQtGraph backend that was requested explicitly, so they stay unless the rule is applied to
plotting too (then `result.plot` would move to `notebooks/util.py` as well).</span>

### 2.7 Things learned the hard way (worth keeping in mind)

- A registration warp should be reported together with its convention.
  `registration1d` uses y_registered(t) = y(gamma(t)) throughout (the fdasrsf and
  nlreg1d convention). The nlreg1d "deviation from linear time" plot is the
  displacement field -(gamma^{-1}(t) - t), not gamma(t) - t; both are
  available.
- Vertical runs at the ends of a DTW path must be collapsed to the fixed end
  points before normalising, otherwise the warp is rescaled and leaves the
  Sakoe-Chiba band.
- The Ramsay-Li parameterisation needs (a) a basis whose log-slope W is free
  at the boundaries (cosines, not sines), (b) coarse-to-fine fitting with a
  few starting points, and (c) a roughness penalty; without these the
  optimiser locks onto the wrong feature or drives the warp to extremes in
  flat regions of the template, where no warp is identifiable.
- Automatic landmark detection by global extrema is fragile whenever a
  feature is doubled (the two braking dips): the global minimum jumps
  between them across observations.
- The SRSF group algorithm's centring step matters: without it the
  registered curves inherit the timing of whichever observation seeded the
  template. <span style="color:#ffd400">Decided (3.4): Karcher mean by default for invertible warps,
  pointwise mean for DTW, anchoring for landmarks; see `WarpCentering.ipynb`.</span>
- <span style="color:#ffd400">Smoothing the square-root slope sqrt(gamma') and re-integrating is a safe
  way to smooth any warp: monotonicity, end points and the total amount of
  warping are preserved (`warp.smooth_warp`).</span>
- <span style="color:#ffd400">The Bayesian sampler with a noise-free observation degenerates (sigma^2 -> 0,
  acceptance -> 0); real data are never noise-free, but simulated tests must
  add noise.</span>

## 3. Suggestions for further development

### 3.1 Near term

1. **API review.** <span style="color:#ffd400">Done: every `register_*` returns a `RegistrationResult`
   (`LinearRegistrationResult` / `NonlinearRegistrationResult`, `islinear`).
   Remaining question: keep the tuple-unpacking magic (`yr, wf = ...`)?
   It is convenient for nlreg1d users but unusual; could be replaced by an
   explicit `result.astuple()`.</span>
2. **Non-uniform grids and explicit time vectors.** <span style="color:#ffd400">Done: optional `t`
   on every method (uniform grids in any units are used as is; non-uniform
   grids are resampled to a uniform grid with the same span and Q;
   `result.t`, `result.warps_t`, `result.displacement_fields_t`).
   Observations of different lengths: done properly, as real-time
   registration (2.5a) rather than by internal resampling.</span>
3. **SRSF options that fdasrsf users expect.** <span style="color:#ffd400">Done: `method='median'`,
   `smooth='spline'`, `parallel`, multivariate input, `band`, `refine`
   (smooth gradient-based refinement of the DP warp). Not done: fdasrsf's
   exact RBFGS algorithm; vertical/horizontal fPCA.</span>
4. **Speed.** <span style="color:#ffd400">See 2.6. Also worth doing regardless of numba: cache the
   per-step segment-cost geometry (the index arrays `I`, `Jx`) across
   observations, which is currently recomputed for every pair.</span>
5. **Inverse / composite warps in `RegistrationResult`.** <span style="color:#ffd400">Done: `apply`,
   `unapply`, `inverse_warps` (univariate and multivariate).</span>
6. **Tests.** <span style="color:#ffd400">See `TESTS.md` for the proposed suite (10 groups, tiers,
   GitHub Actions layout); the current 34 tests cover groups 1, 2, 4 and 6
   in part.</span>
7. <span style="color:#ffd400">**Default `lam` for SRSF.** Done: `lam='auto'` (median total variation
   of the observations; see 3.4).</span>

### 3.2 Additional registration methods

<span style="color:#ffd400">Implemented in this session (all from the mathematics): banded SRSF DP
(`band`), penalised SRSF (`lam`), gradient refinement (`refine`), derivative
DTW, DBA template (`template='dba'`), warp smoothing, self-modelling
registration, pairwise synchronisation, Bayesian registration.</span>

Still worth adding:

- **Elastic registration with fdasrsf's RBFGS** (Riemannian BFGS on the
  sphere) as an alternative to the DP + smooth refinement.
- **Penalised SRSF variants** with a proper second-order roughness penalty
  on gamma (the current `lam` is first-order: departure of sqrt(gamma')
  from 1).
- **Hierarchical Bayesian registration** (group-level warp distributions)
  and a calibrated error model (see 2.5).
- **Multi-resolution / coarse-to-fine SRSF** for very noisy data.
- **Amplitude-phase decomposition helpers**: vertical / horizontal fPCA
  (Tucker et al. 2013), the Karcher mean as an "elastic average curve", and
  the amplitude / phase distance matrices already present, packaged for
  clustering.
- <span style="color:#ffd400">**Front-end support for landmarks**: `detect_landmarks` as an initial
  guess plus a JSON-serialisable landmark table that a front end can edit
  and pass back as the `landmarks` argument; `peaks_as_landmarks` already
  covers the multi-peak case.</span>

### 3.3 Integration

- <span style="color:#ffd400">Done: `notebooks/util.py` (two-sample t, permutation max-t inference,
  `timing_test`; formerly `reg1d.stats`) reproduces the nlreg1d amplitude + timing analysis;
  `Bayesian-vs-nlreg1d.ipynb` shows the full workflow; `result.apply`
  covers the register-on-one-variable-apply-to-others workflow.</span>
- For spm1d users: a helper producing `(yr, d)` in the layout expected by
  `spm1d.stats.ttest2` is trivial (`result.y`, `result.displacement_fields`)
  and could be documented rather than coded.
- Once the API is stable, add GitHub Actions running `pytest` on Linux /
  macOS and building the notebooks (layout proposed in `TESTS.md`), and a
  Read the Docs site built from the docstrings.

### 3.4 Decisions taken (formerly "open questions")

- <span style="color:#ffd400">**Result classes.** All procedures return a `RegistrationResult`
  (`LinearRegistrationResult` / `NonlinearRegistrationResult`, `islinear`).
  The warps of the linear methods remain affine maps that may leave [0,1]
  (faithful to the method); with `cover=True` they contain [0,1].</span>
- <span style="color:#ffd400">**Warp centering.** Every nonlinear method has `center=` with the same
  four options ('karcher', 'pointwise', 'anchor', 'none'; True = the
  method's default, False = 'none'). Defaults: Karcher mean for all methods
  with invertible warps (SRSF, continuous, self-modelling, pairwise,
  Bayesian), pointwise mean for DTW (raw paths may be flat), none for
  landmark registration (mean targets already anchor the data at the
  landmarks). 'anchor' (`anchor=` event times or 'max'/'min') keeps a chosen
  event at its mean time. The method used is recorded in `info['center']`.
  Documented and demonstrated in `WarpCentering.ipynb` (replaces
  `WarpCenteringIssue.ipynb`).</span>
- <span style="color:#ffd400">**Default `lam` for SRSF.** `lam='auto'` = the median SRSF energy of the
  observations, which equals their median total variation int|f'|dt; it
  removes the spurious timing effect of simulated dataset A, keeps the
  genuine effect of B, and leaves the Dorn2012 warps essentially unchanged
  (SUMMARY 2.1a; `srsf.auto_lam`). `lam=0` reproduces fdasrsf. The value
  used is in `info['lam']`; notebook 1's fdasrsf comparison uses `lam=0`.</span>
- **`detect_landmarks`**: kept, as an initial guess for interactive refinement
  in a front end.

### 3.5 Open questions

- Keep the tuple-unpacking magic of `RegistrationResult` (`yr, wf = ...`)?
- <span style="color:#ffd400">The 'anchor' centering is exact only for invertible warps (for raw DTW
  paths the anchor's registered time is ambiguous on flat segments).</span>
- <span style="color:#ffd400">Real-time DTW with slope-constrained step patterns needs `n_ref` close
  to the observation lengths; whether to choose `n_ref` automatically
  (e.g. the median length) is undecided.</span>
