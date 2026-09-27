# Algorithms in registration1d

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
| `register_bayes(y)` | nonlinear | Bayesian pairwise registration to the SRSF Karcher-mean template: tangent-space cosine basis at the identity warp, Gaussian error model in SRSF space, pCN Metropolis with step-size adaptation, Gibbs update of sigma^2, DP initialisation; posterior-mean (Karcher) warps and credible bands | math: Cheng, Dryden & Huang (2016); Lu, Herbei & Kurtek (2017); pCN: Cotter et al. (2013) | posterior mean recovers known warps in simulation (`tests`) |
| `register_pairwise(y)` | nonlinear | all J(J-1) pairwise warps with a selectable engine (srsf / dtw / continuous); each observation's warp = Karcher mean of its warps to all others | math: Tang & Müller (2008) | valid warps, reduces peak-time spread on Dorn2012 |
| `register_sim(y)` | nonlinear | shape-invariant model y_i = a_i mu(gamma_i) + b_i; alternating fit of (gamma_i, a_i, b_i) with parametric warps and least-squares amplitude, and update of mu | math: Kneip & Gasser (1988); Gervini & Gasser (2004) | amplitude parameters recovered on Dorn2012 |
| `register(y, method)` | — | dispatcher over the above | — | — |
| `RegistrationResult` | — | base container: `.y`, `.y0`, `.t`, `.warps` (Warp1DList), `.template`, `.info`, `.islinear`; `apply(z)` / `unapply(z)` / `inverse_warps`; `warps_t`, `displacement_fields_t`; tuple-unpacks to `(y, warps_array)` for nlreg1d compatibility. Subclasses `LinearRegistrationResult` (linear, shift, affine) and `NonlinearRegistrationResult` | — | — |
| `_prepare_grid(y, t)` | — | optional explicit time grid: uniform grids in any units used as is; non-uniform grids resampled to a uniform grid with the same span and Q | — | — |

## 2. SRSF machinery (`reg1d.srsf`)

| function | description | provenance |
|---|---|---|
| `srsf(y, smooth)` | q = sign(f') sqrt(abs(f')), f' by `numpy.gradient` (optional 3-point moving average first, or `smooth='spline'`: derivative of a GCV smoothing spline, scipy `make_smoothing_spline`); for (J,Q,D) arrays the vector SRSF q = f'/sqrt(norm(f')) | math. fdasrsf uses a spline derivative by default; this is the main source of the small numerical differences between the two |
| `srsf_mv`, `srsf_inverse_mv` | vector SRSF and its inverse for (Q,D) observations | math: Srivastava et al. (2011) (SRVF of curves) |
| `srsf_inverse(q, y0)` | f(t) = y0 + int_0^t q abs(q) | math |
| `warp_srsf(q, w)` | group action (q o w) sqrt(w') | math |
| `align_srsf_pair(q1, q2, max_step, nsub, lam, band, dt1, dt2)` | dynamic programming over a grid of Q1 x Q2 nodes (Q1 = Q2 in normalised time; different sizes and sampling intervals dt1, dt2 in real time, with the slope of each segment expressed in time units; univariate or vector SRSFs; optional Sakoe-Chiba band `band` implemented by masking nodes) with admissible steps (di,dj) = all coprime pairs with 1 <= di,dj <= max_step; segment costs computed vectorised for all start nodes with linear interpolation of q1, q2 at `nsub` sub-samples per grid step and trapezoidal integration; strictly increasing paths only; optional penalty lam (sqrt(slope)-1)^2 per segment (the fdasrsf `lam` convention); back-tracking; piecewise-linear warp | math: the DP formulation of Srivastava et al. (2011) §4; the vectorised row-by-row recursion is original |
| `refine_warp(q1, q2, gam)` | smooth refinement of a DP warp: gamma = gam o eta with eta = int exp(W) in a small cosine basis, L-BFGS-B on the SRSF distance; accepted only if it lowers the objective | original (comparable in purpose to fdasrsf `omethod="RBFGS"`, different algorithm) |
| `align_pair(y1, y2)` | convenience: SRSF + DP (+ refinement) + warp application; univariate or multivariate | math |
| `align_group(y, ...)` | initial template = observation nearest the mean SRSF; `lam='auto'` -> `auto_lam` (median SRSF energy = median total variation int|f'|); iterate {align all to template; template = mean (or Weiszfeld-weighted median) of aligned SRSFs} until relative change < tol; centring per `center` (see `warp.center_warps`); template returned in function space via `srsf_inverse`; `parallel` uses `concurrent.futures.ProcessPoolExecutor` over observations | math: Karcher mean / median algorithms of Srivastava et al. (2011) §5 / Tucker et al. (2013), same structure as fdasrsf `srsf_align(method="mean"/"median")` |
| `amplitude_distance(y1, y2)` | min over gamma of L2 distance between SRSFs | math |
| `phase_distance(y1, y2)` | arccos(int sqrt(gamma*')) | math |

## 3. DTW machinery (`reg1d.dtw`)

| function | description | provenance |
|---|---|---|
| `STEP_PATTERNS` | `symmetric1`, `symmetric2`, `asymmetric` (weights as in Giorgino 2009, reference sequence on the first axis), `strict` (steps (1,1),(1,2),(2,1): no flat segments) | math / nomenclature from Giorgino (2009); `strict` is a simple slope-constrained pattern in the spirit of Sakoe-Chiba's P=1 patterns |
| `local_cost(x, y, p)` | abs(x_i - y_j)^p | math |
| `dtw_path(x, y, step_pattern, window, p)` | accumulated-cost recursion with predecessor table and back-tracking; optional band abs(i/(n-1) - j/(m-1)) <= window | math: Sakoe & Chiba (1978) |
| `path_to_warp(path, Q1, Q2)` | for each reference index the matched query indices are averaged; skipped rows interpolated; end points fixed; normalised to [0,1] | original (a standard way of turning a DTW relation into a function) |
| `derivative_estimate(y)` | d_i = ((y_i - y_{i-1}) + (y_{i+1} - y_{i-1})/2)/2 | math: Keogh & Pazzani (2001), derivative DTW |
| `align_pair(..., derivative, smooth)` | pairwise wrapper; optional derivative DTW; optional `warp.smooth_warp` of the DTW warp | math |
| `align_group(..., template)` | iterative refinement of a mean / medoid / fixed template, or true DBA (`template='dba'`: each template point = mean of all observation points matched to it) | math: Petitjean et al. (2011) |

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
| `shift_pair`, `affine_pair` | least-squares fit of delta or (a,b) to a template; fill rules 'edge', 'zero', 'extrapolate' or a constant; `cover=True` constrains b <= 0, a + b >= 1 so that the registered observation contains the whole original | math: Ramsay & Silverman (2005) §7.2; the cover constraint is original |
| `align_group(y, method, ...)` | Procrustes iteration; removal of the mean shift / mean affine map | math |

## 7. Warp machinery (`reg1d.warp`)

| function | description | provenance |
|---|---|---|
| `apply_warp(y, w)` | y(w(t)) by linear interpolation | math |
| `compose(w1, w2)`, `invert(w)` | (w1 o w2)(t) = w1(w2(t)); inverse by swapping axes and re-interpolating | math |
| `derivative`, `displacement`, `displacement_field`, `identity`, `is_valid_warp`, `normalize_warp` | elementary utilities; `displacement_field` = -(w^{-1}(t) - t), the quantity plotted in nlreg1d as "deviation from linear time" | math; definition matched to nlreg1d's `Warp1D.dispf` |
| `warp_to_psi`, `psi_to_warp` | psi = sqrt(gamma'); gamma = int psi^2 | math: Srivastava et al. (2011) §3 |
| `karcher_mean_warp(w)` | intrinsic mean on the L2 unit sphere by iterated exp/log maps | math: Srivastava et al. (2011) §5.2; Srivastava & Klassen (2016) §4.10 |
| `smooth_warp(w, sigma)` | Gaussian smoothing of psi = sqrt(gamma') (reflected at the boundaries) followed by squaring and re-integration: monotonicity, end points and total warping preserved | original (uses the sqrt-slope representation of Srivastava et al.) |
| `center_warps(w, method, anchor)` | w_i o w_ref^{-1} with w_ref = Karcher mean ('karcher'), pointwise mean ('pointwise'), the PCHIP warp mapping the mean registered anchor time to the mean original anchor time ('anchor'), or the identity ('none'); `resolve_center` maps True/False to each method's default | math: 'karcher' is the centring step of fdasrsf `srsf_align(center=True)`; 'pointwise' and 'anchor' are the natural alternatives (see WarpCentering notebook) |
| `random_warp(J, Q, sigma, n_basis)` | random tangent vector at the identity in a sine basis, exponential map to the sphere, integration to a warp | math: the construction used for simulation in the SRSF literature; equivalent in spirit to scikit-fda `make_random_warping` but written independently (parameterisation differs) |
| `Warp1D`, `Warp1DList` | object wrappers (apply, compose, inverse, displacement fields, Karcher mean, centring, plotting) | interface modelled on nlreg1d; implementation new |

## 7. Real-time registration (`reg1d.realtime`)

| function | description | provenance |
|---|---|---|
| `prepare(y, t)` | ragged input -> lists of observations and uniformly spaced time vectors (t: None = frames, scalar dt, 'fs=<Hz>', or one vector per observation) | — |
| `reference_axis(durations, n_ref, T_ref)` | uniform reference axis, n_ref points over the mean duration by default | — |
| `align_group_srsf(...)` | SRSFs with derivatives in physical time (q_real = q_unit / sqrt(T_i)); DP between the reference grid (dt1) and each observation's grid (dt2) via `srsf.align_srsf_pair(..., dt1, dt2, return_index=True)`; Karcher mean / median template on the reference axis; Karcher centring of the normalised warps; multivariate supported | math: the SRSF framework is defined for arbitrary domains; the two-grid DP is the standard DP with the slope expressed in time units |
| `align_group_dtw(...)` | DTW between each observation and the reference-axis template; derivative estimates divided by the sampling intervals | math: Sakoe & Chiba (1978); Keogh & Pazzani (2001) |
| `align_group_landmark(...)` | monotone interpolant through (0,0), (target_k, landmark_ik), (T_ref, T_i) in seconds | math: Ramsay & Silverman (2005) §7.3 |
| outputs | `warps` (normalised gamma_i = Gamma_i(s T_ref)/T_i), `warps_realtime` (Gamma_i in seconds), `displacement_realtime` (Gamma_i - tau T_i/T_ref), `y0` (linearly rescaled observations) | — |

## 7a. Bayesian registration (`reg1d.bayes`)

| function | description | provenance |
|---|---|---|
| `coef_to_warp(c, B, t)` | v = sum c_k sqrt(2) cos(k pi t); psi = exp_1(v) = cos|v| + sin|v| v/|v|; gamma = int psi^2 | math: tangent-space parameterisation of Srivastava et al. (2011) |
| `sample_pair(...)` | pCN Metropolis for c (proposal c' = sqrt(1-beta^2) c + beta xi, xi ~ prior N(0, tau^2/k^2)), adaptive beta during burn-in, Gibbs update sigma^2 ~ InvGamma(a0 + n/2, b0 + n E/2), optional likelihood tempering `n_eff` | math: Cheng, Dryden & Huang (2016) model; Lu et al. (2017) GP-prior view; Cotter, Roberts, Stuart & White (2013) pCN |
| `initial_coef(gam, K)` | inverse exponential map of sqrt(gamma') at the identity, projected on the basis | math |
| `summarize(samples)` | Karcher-mean warp, pointwise credible intervals of warp and displacement field | math |
| `align_group(...)` | fixed template (SRSF Karcher mean by default), one chain per observation, DP initialisation | math (simplification: no template update, no hierarchy) |

## 7b. Pairwise synchronisation (`reg1d.pairwise`) and self-modelling (`reg1d.sim`)

| function | description | provenance |
|---|---|---|
| `pairwise.align_group(y, engine)` | all pairwise warps gamma_ij (engine: srsf / dtw / continuous); gamma_i = Karcher mean_j gamma_ij | math: Tang & Müller (2008) |
| `sim.align_pair(mu, y)` | fit (gamma, a, b) with gamma = int exp(W) (cosine basis, coarse-to-fine, multi-start), (a,b) by least squares for each candidate warp | math: Kneip & Gasser (1988); Gervini & Gasser (2004) |
| `sim.align_group(y)` | alternate pair fits and shape-function update (cross-sectional mean of amplitude-normalised registered observations); Karcher centring | math |

## 7c. Statistics helpers (`reg1d.stats`)

| function | description | provenance |
|---|---|---|
| `ttest2(yA, yB)` | pointwise two-sample t (pooled variance) | math |
| `permutation_ttest2(...)` | permutation distribution of max |t| over the domain; threshold, p value, suprathreshold clusters | math: SnPM "tmax" inference (Nichols & Holmes 2002) |
| `timing_test(result, group)` | nlreg1d-style amplitude test (registered data) + timing test (displacement fields) | math; workflow of Pataky et al. (2022) |

## 8. Data and plotting

| function | description | provenance |
|---|---|---|
| `data.Dorn2012`, `data.load_dorn2012` | loader for the bundled reduced Dorn et al. (2012) GRF dataset (from the nlreg1d repository, MIT) | new loader, same data |
| `data.Dorn2012MV` | three-component forces, 18 trials (nlreg1d `Data/Dorn2021-orig.npz`) | new loader, same data |
| `data.SimulatedA`, `data.SimulatedB` | two-group simulated datasets of the nlreg1d paper (amplitude effect / timing effect) | new loader, same data |
| `plot.plot_curves`, `plot_warps`, `plot_displacement_fields`, `plot_registration` | matplotlib helpers | new |

## References

- Cheng W, Dryden IL, Huang X (2016). Bayesian registration of functions and curves. Bayesian Analysis 11: 447-475.
- Cotter SL, Roberts GO, Stuart AM, White D (2013). MCMC methods for functions: modifying old algorithms to make them faster. Statistical Science 28: 424-446.
- Gervini D, Gasser T (2004). Self-modelling warping functions. J R Stat Soc B 66: 959-971.
- Keogh EJ, Pazzani MJ (2001). Derivative dynamic time warping. SIAM International Conference on Data Mining.
- Kneip A, Gasser T (1988). Convergence and consistency results for self-modeling nonlinear regression. Ann Stat 16: 82-112.
- Lu Y, Herbei R, Kurtek S (2017). Bayesian registration of functions with a Gaussian process prior. J Comput Graph Stat 26: 894-904.
- Nichols TE, Holmes AP (2002). Nonparametric permutation tests for functional neuroimaging. Human Brain Mapping 15: 1-25.
- Pataky TC, Robinson MA, Vanrenterghem J, Donnelly CJ (2022). Simultaneously assessing amplitude and temporal effects in biomechanical trajectories using nonlinear registration and statistical nonparametric mapping. J Biomech 136: 111049.
- Tang R, Müller H-G (2008). Pairwise curve synchronization for functional data. Biometrika 95: 875-889.

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
