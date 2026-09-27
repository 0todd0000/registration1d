# reg1d — summary of findings and suggestions for further development

Date: 2026-09-27. Status: preliminary package (v0.0.1), 22 passing tests,
three executed notebooks.

## 1. What was built

A GPL-3.0-or-later Python package `reg1d` that depends only on numpy, scipy
and matplotlib and reproduces the workflow of `nlreg1d/Notebooks/3-Registration.ipynb`
(linear interpolation to 101 points followed by SRSF registration of the
Dorn2012 GRF data) **without fdasrsf or scikit-fda installed**. The package
contains:

- two linear methods (`register_shift`, `register_affine`) plus
  `register_linear` (interpolation, as in nlreg1d);
- four nonlinear methods: `register_srsf` (elastic / Fisher-Rao),
  `register_dtw`, `register_landmark`, `register_continuous` (Ramsay-Li);
- a warp toolkit (`reg1d.warp`: apply, compose, invert, displacement fields,
  Karcher means, centring, random warps, `Warp1D` / `Warp1DList`);
- elastic amplitude and phase distances;
- a `RegistrationResult` object that unpacks as `(yr, wf)` for nlreg1d-style
  code and carries the template, warps and diagnostics;
- the Dorn2012 dataset, three notebooks (`notebooks/*.ipynb` + `html/`),
  a test suite (`tests/`), `requirements.txt`, `pyproject.toml`.

Layout:

    reg1d/            package
      warp.py         warps and warp algebra
      srsf.py         SRSF transform, dynamic programming, Karcher-mean alignment, distances
      dtw.py          dynamic time warping
      landmark.py     landmark registration and simple landmark detection
      continuous.py   Ramsay-Li penalised least-squares registration
      linear.py       resampling, shift and affine registration
      reg.py          public register_* functions and RegistrationResult
      data.py, plot.py, data/Dorn2012-reduced.npz
    notebooks/        1-Registration, 2-Methods, 3-Warps (+ html/, make_notebooks.py)
    tests/            pytest suite

## 2. Key findings

### 2.1 SRSF registration can be implemented from the mathematics in ~300 lines and matches fdasrsf

The dynamic-programming alignment of SRSFs is the heart of fdasrsf's
`srsf_align`. Re-implemented from Srivastava et al. (2011) with a vectorised
row-by-row recursion, it runs in about 0.03 s per pairwise alignment for
Q = 101 (max_step = 6, i.e. local slopes 1/6 ... 6) with no compiled code.
Group registration of the 8 Dorn2012 curves with 5 template updates takes
about 2 s. On the Dorn2012 data the warps agree with fdasrsf 2.7.2 to within
about 0.01 on the unit interval for 7 of 8 observations (0.06 for one, in
the region of the double braking dip where the objective is nearly flat),
and the registered curves are visually indistinguishable from the figure in
the nlreg1d notebook.

Two deliberate differences from fdasrsf: (i) the SRSF derivative is
`numpy.gradient` (optionally after a moving average) rather than a smoothing
spline; (ii) the DP slope set is explicit and user-controllable (`max_step`).
Both could be aligned with fdasrsf exactly if bit-for-bit agreement were
ever wanted, but that is not a goal for a GPL re-derivation.

### 2.2 Licensing: nothing stands in the way of GPLv3

All runtime dependencies are BSD/PSF licensed. fdasrsf (BSD-3) and
scikit-fda (BSD-3) *could* have been imported by a GPL package; dtw-python is
itself GPL-3.0-or-later. The decision to re-derive rather than wrap was
therefore not forced by licensing but by the wish to avoid fdasrsf's heavy
dependency chain (Cython, numba, cffi, joblib, patsy, ...) and its
compilation step. Details are in `GPL-COMPLIANCE.md`.

### 2.3 Suitability of the methods for the Dorn2012 data (notebook 2)

| method | verdict for these data |
|---|---|
| SRSF | best general-purpose choice; amplitude-invariant; proper diffeomorphic warps; aligns peak and zero crossing well, braking dips reasonably |
| landmark | exact alignment of chosen events (zero crossing, propulsive peak); the double braking dip defeats simple automatic landmark detection, so landmarks should be supplied or checked by hand |
| DTW | fast but amplitude-driven: with peak GRF doubling across speeds it matches amplitudes rather than shapes, producing staircase warps and flat (non-invertible) segments; usable only with a tight window or a slope-constrained step pattern |
| continuous (Ramsay-Li) | smooth low-dimensional warps; surprisingly good on Dorn2012 with 8 cosine terms (aligns both braking dips), but the objective is non-convex and local minima do occur in simulation; needs the penalty (default lam = 0.01) |
| shift / affine | only global timing; little value here beyond a baseline |

The scikit-fda methods that were considered are all represented: its
`ElasticRegistration` is SRSF (via fdasrsf), `LeastSquaresShiftRegistration`
is `register_shift`, `landmark_shift_registration` / `landmark_elastic_registration`
are `register_landmark` (with linear / PCHIP warps). scikit-fda's
`ElasticRegistration` adds nothing to fdasrsf for this purpose.

### 2.4 Things learned the hard way (worth keeping in mind)

- A registration warp should be reported together with its convention.
  `reg1d` uses y_registered(t) = y(gamma(t)) throughout (the fdasrsf and
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
  template.

## 3. Suggestions for further development

### 3.1 Near term (before moving to a private repository)

1. **API review.** Decide whether `register_*` should return the
   `RegistrationResult` (current) or a plain `(yr, wf)` tuple with the
   object available via a keyword. The current object unpacks as a tuple,
   which is convenient but slightly magical.
2. **Non-uniform grids and explicit time vectors.** Everything currently
   assumes t = linspace(0,1,Q). Accepting an explicit `t` (and observations
   of different lengths in the nonlinear methods, by resampling internally)
   would remove the mandatory `register_linear` step.
3. **SRSF options that fdasrsf users expect:** `method="median"`
   (Karcher median), a smoothing-spline derivative (`smooth='spline'`),
   parallel pairwise alignment (`concurrent.futures`; the DP is embarrassingly
   parallel across observations), and multivariate observations (vector-valued
   SRSF, straightforward extension of the DP cost).
4. **Speed.** The DP is already vectorised over one axis; if larger Q or many
   observations are needed, numba (BSD) is an easy optional accelerator, or
   a small C extension. Q = 101 does not need it.
5. **Inverse / composite warps in `RegistrationResult`**, e.g.
   `result.unwarp(y_other)` to apply the found warps to another variable
   measured simultaneously (very common in biomechanics: register on GRF,
   apply to joint angles).
6. **Tests** for the plotting functions and for edge cases (constant
   observations, Q < 20, observations with NaN).

### 3.2 Additional registration methods worth adding

- **Elastic registration with an RBFGS / gradient refinement** after DP
  (fdasrsf `omethod="RBFGS"`): smoother warps, same objective.
- **Pairwise SRSF with a Sakoe-Chiba-type band** (cheap to add to the DP:
  mask the cost) to bound the amount of warping.
- **Penalised SRSF variants** with a proper roughness penalty on gamma
  (rather than the fdasrsf `lam` penalty on departure from identity).
- **Derivative / curvature DTW** (Keogh & Pazzani 2001), which removes the
  amplitude sensitivity that hurt plain DTW here; and **DBA** proper
  (Petitjean et al. 2011) for the DTW template.
- **Self-modelling / shape-invariant model registration** (Kneip & Gasser
  1988; Gervini & Gasser 2004) and **pairwise-synchronisation**
  (Tang & Müller 2008) — the classical FDA alternatives to SRSF.
- **Bayesian registration** (Cheng, Dryden & Huang 2016; Lu et al. 2017)
  giving uncertainty on the warps — relevant to hypothesis testing on timing.
- **Multi-resolution / coarse-to-fine SRSF** for very noisy data.
- **Amplitude-phase decomposition helpers**: vertical / horizontal fPCA
  (Tucker et al. 2013), the Karcher mean as an "elastic average curve", and
  the amplitude / phase distance matrices already present, packaged for
  clustering.

### 3.3 Integration

- Provide `Warp1DList.to_spm1d()`-style helpers or examples showing how to
  feed registered data and displacement fields into hypothesis tests, as in
  nlreg1d's notebooks 4 and 5.
- Once the API is stable, add GitHub Actions running `pytest` on Linux /
  macOS and building the notebooks, and a Read the Docs site built from the
  docstrings.

### 3.4 Open questions

- Should linear methods (shift / affine) return warps that leave [0,1]
  (current) or be clipped to boundary-preserving warps? The former is
  faithful; the latter fits the `Warp1D` algebra.
- Which centring convention should be the default for DTW and landmark
  registration? Currently SRSF and continuous registration centre their
  warps (Karcher mean = identity); DTW warps are not diffeomorphisms so they
  are left uncentred; landmark warps are centred implicitly by the mean
  target times.
- Whether to keep `detect_landmarks` at all, or replace it with a
  documented recipe based on `scipy.signal.find_peaks`.
