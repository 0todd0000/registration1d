# Algorithms in reg1d

Every function is listed with its provenance: **math** means it was written
directly from the cited mathematical description; no function was copied or
translated from a third-party package. Where a third-party package was used to
*check* a function numerically, that is stated under "validated against".

Conventions: observations are (Q,) or (J,Q) arrays sampled on the uniform grid
t = linspace(0,1,Q); a warp is a (Q,) array with gamma(0)=0, gamma(1)=1,
non-decreasing; applying a warp gives y(gamma(t)), so a registration warp for
observation i satisfies y_registered_i(t) = y_i(gamma_i(t)).

## 1. Registration functions (public interface, `reg1d.reg`)

| function | type | algorithm | provenance | validated against |
|---|---|---|---|---|
| `register_linear(y, n)` | linear | piecewise-linear (or cubic) interpolation of each observation onto n equally spaced points | math (trivial; mirrors `nlreg1d.register_linear`) | nlreg1d notebook 3 (identical output for linear interpolation) |
| `register_shift(y)` | linear | gamma(t)=t+delta; delta by bounded scalar minimisation of int (y(t+delta)-template)^2; Procrustes iteration on the cross-sectional mean; mean shift removed | math: Ramsay & Silverman (2005) §7.2 | recovers a known shift in `tests` |
| `register_affine(y)` | linear | gamma(t)=a t+b; coarse grid search + Nelder-Mead on the same criterion; mean affine map removed | math: natural extension of shift registration | recovers known (a,b) in `tests` |
| `register_srsf(y)` | nonlinear | SRSF transform; pairwise alignment by dynamic programming; Karcher-mean template iteration; Karcher-mean centring of warps | math: Srivastava et al. (2011), Tucker et al. (2013), Srivastava & Klassen (2016) ch. 4 & 8 | fdasrsf 2.7.2 `srsf_align` (max warp difference ~0.01 on Dorn2012) |
| `register_dtw(y)` | nonlinear | classical DTW with step patterns and optional Sakoe-Chiba band; path converted to a warp; iterative mean template (simple barycentre averaging) | math: Sakoe & Chiba (1978); Giorgino (2009) for step-pattern nomenclature | dtw-python 1.7.5 (identical paths and distances) |
| `register_landmark(y)` | nonlinear | monotone (PCHIP or linear) interpolant through (target_k, landmark_ik) with fixed end points; targets = mean/median landmark times | math: Kneip & Gasser (1992); Ramsay & Silverman (2005) §7.3 | landmarks coincide exactly after registration (`tests`) |
| `register_continuous(y)` | nonlinear | gamma = int exp(W) / int_0^1 exp(W), W = sum c_k sqrt(2) cos(k pi t); coefficients by L-BFGS-B (coarse-to-fine, multi-start on the first two coefficients) on int (y(gamma)-template)^2 + lam int W'^2; Procrustes iteration; Karcher-mean centring | math: Ramsay & Li (1998); Ramsay & Silverman (2005) §7.4 | recovers smooth warps in simulation (notebook 3) |
| `register(y, method)` | — | dispatcher over the above | — | — |
| `RegistrationResult` | — | container: `.y`, `.y0`, `.warps` (Warp1DList), `.template`, `.info`; tuple-unpacks to `(y, warps_array)` for nlreg1d compatibility | — | — |

## 2. SRSF machinery (`reg1d.srsf`)

| function | description | provenance |
|---|---|---|
| `srsf(y, smooth)` | q = sign(f') sqrt(abs(f')), f' by `numpy.gradient` (optional 3-point moving average first) | math. fdasrsf uses a smoothing-spline derivative instead; this is the main source of the small numerical differences between the two |
| `srsf_inverse(q, y0)` | f(t) = y0 + int_0^t q abs(q) | math |
| `warp_srsf(q, w)` | group action (q o w) sqrt(w') | math |
| `align_srsf_pair(q1, q2, max_step, nsub, lam)` | dynamic programming over a grid of Q x Q nodes with admissible steps (di,dj) = all coprime pairs with 1 <= di,dj <= max_step; segment costs computed vectorised for all start nodes with linear interpolation of q1, q2 at `nsub` sub-samples per grid step and trapezoidal integration; strictly increasing paths only; optional penalty lam (sqrt(slope)-1)^2 per segment (the fdasrsf `lam` convention); back-tracking; piecewise-linear warp | math: the DP formulation of Srivastava et al. (2011) §4; the vectorised row-by-row recursion is original |
| `align_pair(y1, y2)` | convenience: SRSF + DP + warp application | math |
| `align_group(y, ...)` | initial template = observation nearest the mean SRSF; iterate {align all to template; template = mean of aligned SRSFs} until relative change < tol; optional centring (see `warp.center_warps`); template returned in function space via `srsf_inverse` | math: Karcher mean algorithm of Srivastava et al. (2011) §5 / Tucker et al. (2013), same structure as fdasrsf `srsf_align(method="mean")` |
| `amplitude_distance(y1, y2)` | min over gamma of L2 distance between SRSFs | math |
| `phase_distance(y1, y2)` | arccos(int sqrt(gamma*')) | math |

## 3. DTW machinery (`reg1d.dtw`)

| function | description | provenance |
|---|---|---|
| `STEP_PATTERNS` | `symmetric1`, `symmetric2`, `asymmetric` (weights as in Giorgino 2009, reference sequence on the first axis), `strict` (steps (1,1),(1,2),(2,1): no flat segments) | math / nomenclature from Giorgino (2009); `strict` is a simple slope-constrained pattern in the spirit of Sakoe-Chiba's P=1 patterns |
| `local_cost(x, y, p)` | abs(x_i - y_j)^p | math |
| `dtw_path(x, y, step_pattern, window, p)` | accumulated-cost recursion with predecessor table and back-tracking; optional band abs(i/(n-1) - j/(m-1)) <= window | math: Sakoe & Chiba (1978) |
| `path_to_warp(path, Q1, Q2)` | for each reference index the matched query indices are averaged; skipped rows interpolated; end points fixed; normalised to [0,1] | original (a standard way of turning a DTW relation into a function) |
| `align_pair`, `align_group` | pairwise wrapper; iterative refinement of a mean / medoid / fixed template | math (barycentre-averaging idea of Petitjean et al. 2011, in its simplest form) |

## 4. Landmark machinery (`reg1d.landmark`)

| function | description | provenance |
|---|---|---|
| `landmark_warp(landmarks, targets, Q, kind)` | monotone interpolant (scipy `PchipInterpolator` or linear) through (0,0), (target_k, landmark_k), (1,1) | math: Ramsay & Silverman (2005) §7.3 (they use monotone smoothing; PCHIP is the simplest monotone interpolant) |
| `detect_landmarks(y, kinds)` | global min, global max, and the zero crossing adjacent to the global max | original heuristic, written for the anteroposterior GRF example |
| `peaks_as_landmarks(y, n_peaks)` | n most prominent maxima via scipy `find_peaks` | thin scipy wrapper |
| `align_group(y, landmarks, targets, kind)` | landmark registration of a set | math |

## 5. Continuous-registration machinery (`reg1d.continuous`)

| function | description | provenance |
|---|---|---|
| `coef_to_warp(c, Q)` | gamma = int_0^t exp(W) / int_0^1 exp(W), W = sum c_k sqrt(2) cos(k pi t) | math: Ramsay (1998) smooth monotone transformation; cosine basis chosen so that W is free at the boundaries |
| `align_pair(y_template, y, n_basis, lam, c0, c_max)` | L-BFGS-B minimisation of penalised SSE; coarse-to-fine over the number of coefficients with a 7-point multi-start on the first two stages; coefficient bounds +/- c_max | math: Ramsay & Li (1998) criterion; the optimisation strategy is original |
| `align_group(y, ...)` | Procrustes iteration on the cross-sectional mean; Karcher-mean centring | math: Ramsay & Silverman (2005) §7.4 |

## 6. Linear-registration machinery (`reg1d.linear`)

| function | description | provenance |
|---|---|---|
| `resample(y, n, kind)` | interpolation to n points; accepts one array or a ragged sequence | math (trivial) |
| `shift_pair`, `affine_pair` | least-squares fit of delta or (a,b) to a template, boundary values held ('edge') or constant | math: Ramsay & Silverman (2005) §7.2 |
| `align_group(y, method, ...)` | Procrustes iteration; removal of the mean shift / mean affine map | math |

## 7. Warp machinery (`reg1d.warp`)

| function | description | provenance |
|---|---|---|
| `apply_warp(y, w)` | y(w(t)) by linear interpolation | math |
| `compose(w1, w2)`, `invert(w)` | (w1 o w2)(t) = w1(w2(t)); inverse by swapping axes and re-interpolating | math |
| `derivative`, `displacement`, `displacement_field`, `identity`, `is_valid_warp`, `normalize_warp` | elementary utilities; `displacement_field` = -(w^{-1}(t) - t), the quantity plotted in nlreg1d as "deviation from linear time" | math; definition matched to nlreg1d's `Warp1D.dispf` |
| `warp_to_psi`, `psi_to_warp` | psi = sqrt(gamma'); gamma = int psi^2 | math: Srivastava et al. (2011) §3 |
| `karcher_mean_warp(w)` | intrinsic mean on the L2 unit sphere by iterated exp/log maps | math: Srivastava et al. (2011) §5.2; Srivastava & Klassen (2016) §4.10 |
| `center_warps(w)` | w_i o w_mean^{-1}, so that the Karcher mean is the identity | math: the centring step of fdasrsf `srsf_align(center=True)` |
| `random_warp(J, Q, sigma, n_basis)` | random tangent vector at the identity in a sine basis, exponential map to the sphere, integration to a warp | math: the construction used for simulation in the SRSF literature; equivalent in spirit to scikit-fda `make_random_warping` but written independently (parameterisation differs) |
| `Warp1D`, `Warp1DList` | object wrappers (apply, compose, inverse, displacement fields, Karcher mean, centring, plotting) | interface modelled on nlreg1d; implementation new |

## 8. Data and plotting

| function | description | provenance |
|---|---|---|
| `data.Dorn2012`, `data.load_dorn2012` | loader for the bundled reduced Dorn et al. (2012) GRF dataset (from the nlreg1d repository, MIT) | new loader, same data |
| `plot.plot_curves`, `plot_warps`, `plot_displacement_fields`, `plot_registration` | matplotlib helpers | new |

## References

- Dorn TW, Schache AG, Pandy MG (2012). J Exp Biol 215: 1944-1956.
- Giorgino T (2009). Computing and visualizing dynamic time warping alignments in R: the dtw package. J Stat Softw 31(7).
- Kneip A, Gasser T (1992). Statistical tools to analyze data representing a sample of curves. Ann Stat 20: 1266-1305.
- Petitjean F, Ketterlin A, Gancarski P (2011). A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition 44: 678-693.
- Ramsay JO (1998). Estimating smooth monotone functions. J R Stat Soc B 60: 365-375.
- Ramsay JO, Li X (1998). Curve registration. J R Stat Soc B 60: 351-363.
- Ramsay JO, Silverman BW (2005). Functional Data Analysis, 2nd ed. Springer. Chapter 7.
- Sakoe H, Chiba S (1978). Dynamic programming algorithm optimization for spoken word recognition. IEEE Trans ASSP 26: 43-49.
- Srivastava A, Wu W, Kurtek S, Klassen E, Marron JS (2011). Registration of functional data using Fisher-Rao metric. arXiv:1103.3817.
- Srivastava A, Klassen EP (2016). Functional and Shape Data Analysis. Springer.
- Tucker JD, Wu W, Srivastava A (2013). Generative models for functional data using phase and amplitude separation. Comput Stat Data Anal 61: 50-66.
