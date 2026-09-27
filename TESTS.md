# Test strategy for reg1d

This document describes the test suite in `tests/` (currently 37 tests,
`python -m pytest tests`) and the broader suite proposed for the package
before it is opened up. The tests are grouped by *what property they
protect*; each group states which tests already exist and which are proposed.

## 1. Mathematical identities of the warp toolkit (exist)

These are cheap, deterministic and catch most regressions in `reg1d.warp`.

- inverse and composition: `w o w^-1 = w^-1 o w = identity` to interpolation tolerance;
- `psi_to_warp(warp_to_psi(w)) = w`;
- random warps are valid (monotone, fixed end points);
- centered warps have the identity as Karcher mean;
- `smooth_warp` preserves validity and changes the warp only moderately.

Proposed additions: exactness of `displacement_field` against the analytic
displacement of a known warp (e.g. `gamma(t) = t^2`); `karcher_mean_warp` of
`{w, w^-1}` is close to the identity; `compose` is associative to tolerance.

## 2. Recovery of known warps (exist for every method)

Synthetic data with a known generating warp are the strongest functional
tests: `f o g` registered to `f` must return `g^-1`.

- SRSF pairwise (max error < 0.03), SRSF group (SSE reduced by > 90 %, Karcher mean of warps = identity);
- shift and affine recover known `delta`, `(a, b)`;
- continuous recovers a smooth warp;
- landmark places explicit landmarks exactly;
- multivariate SRSF recovers the warp of a 2-component observation;
- Bayesian posterior mean is close to the inverse generating warp; acceptance rate is in a sane range;
- real-time: the same event sampled at two rates aligns exactly through the two-grid DP; real-time SRSF / DTW / landmark on Dorn2012 give valid warps ending at each observation's duration.

Proposed additions: a **parametrised recovery matrix** (pytest parametrize
over method x warp strength x noise level x Q) with method-specific
tolerances, run in the "slow" tier; recovery with non-uniform `t`; recovery
when the template is given explicitly.

## 3. Agreement with reference implementations (partly exist, optional)

- DTW: identical path and distance to `dtw-python` for `symmetric1`, `symmetric2`, `asymmetric` and the Sakoe-Chiba window (verified by hand in this session; proposed as a test that is skipped when `dtw-python` is not installed);
- SRSF: warps within 0.02 (7 of 8 observations) of `fdasrsf.srsf_align` on Dorn2012 (verified in notebook 1; proposed as a skip-if-missing test);
- shift registration versus `scikit-fda` `LeastSquaresShiftRegistration` (proposed).

These tests are run only in a development environment where the reference
packages are installed; they are `pytest.importorskip`-guarded so that the
default suite stays dependency-free.

## 4. Invariants that every RegistrationResult must satisfy (partly exist)

For every method and a small set of inputs (Dorn2012, simulated A/B, a
constant observation, a single observation, J = 2):

- `result.y.shape == result.y0.shape`, `result.warps.shape == (J, Q)`;
- nonlinear warps are valid (`is_valid_warp`), linear results have `islinear = True`;
- `result.apply(result.y0) == result.y` exactly;
- `result.unapply(result.y)` is close to `result.y0` (interpolation error only);
- tuple unpacking `(yr, wf) = result` matches the attributes;
- `t` handling: uniform `t` in other units leaves warps unchanged and rescales `warps_t` / `displacement_fields_t`; non-uniform `t` gives a uniform `result.t` with the same span;
- no NaN or Inf anywhere in the outputs.

Currently covered for SRSF and landmark; proposed as a single parametrised
test over all entries of `reg.METHODS`.

## 5. Edge cases and error handling (proposed)

- constant observations (SRSF is identically zero: the DP must return the identity, not fail);
- very short observations (Q = 5, 10) and Q not equal between template and observation (must raise a clear `ValueError`);
- observations containing NaN (raise, or document masking);
- a band / window too narrow to admit a path (clear error message);
- `landmarks` outside (0,1) or non-monotone (clear error);
- degenerate affine maps (a -> 0) rejected by the bounds;
- Bayesian sampler with `K = 1`, `n_samples = 1`.

## 6. Behavioural tests on real data (exist in part)

Rather than exact values, these assert *qualitative* outcomes that must not
regress, e.g. on Dorn2012: after SRSF registration the SD of the propulsive
peak time is smaller than before; with `band = 0.05` no warp deviates more
than 0.05 from the identity; `cover=True` keeps zero end points. The
notebooks act as a further, human-inspected layer of behavioural testing;
a proposed GitHub Actions job executes them (`make_notebooks.py`) so that a
change that breaks a notebook is caught even if no assertion fails.

## 7. Numerical stability and determinism (proposed)

- results are identical across repeated runs (`parallel=True` versus `False`; fixed `random_state` for Bayesian and permutation code);
- results are stable to small perturbations of the data (registration of `y + 1e-8` equals registration of `y` to tolerance);
- float32 input is accepted and up-cast.

## 8. Plotting (proposed)

Smoke tests with the `Agg` backend: every plotting function and
`result.plot` returns axes and raises nothing for univariate and
multivariate results, with and without groups.

## 9. Performance guards (proposed, "slow" tier)

- pairwise SRSF alignment at Q = 101 under 0.1 s, group registration of 100 observations x 5 iterations under 60 s on the GitHub runner; these bound accidental algorithmic regressions (e.g. an O(Q^3) loop slipping in);
- memory: the DP segment-cost arrays are the largest allocation (about 23 x Q x Q x (di x nsub) doubles at Q = 101 -> a few MB); a test with Q = 1001 checks that memory stays below a few hundred MB.

## 10. Tiers and GitHub Actions

Proposed layout:

    tests/test_warp.py          groups 1
    tests/test_methods.py       groups 2, 4, 6  (fast)
    tests/test_edge_cases.py    group 5
    tests/test_reference.py     group 3 (importorskip)
    tests/test_slow.py          groups 2 (matrix), 7, 9  (marked slow)
    tests/test_plot.py          group 8

GitHub Actions: a matrix over Python 3.9-3.12 and Linux / macOS running the
fast tiers on every push, the slow tier and the notebook build nightly or
on pull requests, and a separate job that installs `fdasrsf` and
`dtw-python` and runs the reference-agreement tests.
