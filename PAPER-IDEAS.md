# Paper ideas for reg1d

Notes collected while building the package. Idea 1 is the safe, near-term
option; ideas 2-4 would each need additional work but have, in my judgement,
enough substance for a methods paper.

## 1. Software paper — Journal of Open Source Software (JOSS)

**Pitch.** A pure-Python (numpy / scipy / matplotlib only), GPLv3, curated
suite of linear and nonlinear registration methods for one-dimensional data
with a single interface, one result object, warp algebra, uncertainty
quantification (Bayesian registration) and multivariate support.

**What JOSS looks for and how reg1d meets it.**

- *Statement of need*: existing tools are either heavy (fdasrsf: Cython/numba/cffi compile chain; scikit-fda: large dependency tree, wraps fdasrsf for elastic registration), single-method (dtw-python, dtaidistance) or R-only (fdasrvf, fda). Biomechanics users typically need to try several methods on the same data and to feed warps into hypothesis tests; nothing offers that in a light package.
- *Functionality*: 10 methods (linear, shift, affine, SRSF, DTW / derivative DTW / DBA, landmark, continuous, self-modelling, pairwise synchronisation, Bayesian), all from the mathematics, with reference-implementation agreement for SRSF (fdasrsf) and DTW (dtw-python).
- *Quality*: test suite (TESTS.md), executed notebooks, GitHub Actions.
- *Length*: JOSS papers are short (1-2 pages); the notebooks become the documentation.

**Gaps to close first.** Documentation site; the reference-agreement tests as
part of the suite; a benchmark table (time, accuracy) across methods; a
`CITATION.cff`; a release on PyPI and Zenodo.

**Alternative venue of the same kind:** *SoftwareX* (longer software papers,
Elsevier, OA), or *Journal of Statistical Software* if the emphasis is put on
the statistical methodology and the comparison of methods (JSS papers are
much longer and more demanding, but carry more weight in statistics).

## 2. Methods paper — "Registration uncertainty and timing inference for biomechanical curves"

**Question.** The nlreg1d approach tests timing effects on displacement
fields obtained from point-estimate registration. How much does ignoring
registration uncertainty matter, and how can it be incorporated?

**Content.** (a) Bayesian registration (reg1d.bayes) gives per-observation
posteriors of the warps; (b) posterior-predictive propagation of the warps
into the two-sample timing test (as in the Bayesian-vs-nlreg1d notebook);
(c) a hierarchical model with group-level warp distributions on the sphere
of square-root slopes as a fully Bayesian counterpart to the nlreg1d test;
(d) simulation study: power and false-positive rate of the nlreg1d test with
and without uncertainty propagation as a function of noise, warp strength,
sample size and the flatness of the curves; (e) application to Dorn2012
(univariate and multivariate).

**Novelty.** Bayesian registration exists (Cheng et al. 2016; Lu et al. 2017;
Matuk et al. 2022 for the SRVF-based "Bayesian elastic" framework), and
the nlreg1d timing test exists, but their combination — and especially the
question of whether registration uncertainty changes inferential conclusions
in movement science — has not been studied. Fits *Journal of Biomechanics*
(methods note), *Journal of the Royal Society Interface*, or *Statistics in
Medicine* depending on the emphasis.

**Effort.** Moderate-high: the hierarchical model is new code; the
simulation study needs compute; the current sampler's calibration (credible
interval coverage, `n_eff`) must be sorted out properly (e.g. by a whitened
likelihood or a Gaussian-process error model).

## 3. Methods / comparison paper — "Which registration method for gait and running curves? A systematic comparison on a common footing"

**Question.** Papers in biomechanics use DTW, landmark registration, SRSF
alignment or none of them, largely by habit. Which methods preserve the
physical structure of force / kinematic curves (smooth derivatives,
diffeomorphic warps, amplitude invariance) and which distort downstream
statistics?

**Content.** All reg1d methods, on the same data and with the same warp
convention and centering, evaluated on (a) recovery of known warps in
simulation (with realistic warps and amplitude variation), (b) smoothness of
the registered derivatives (the DTW problem raised in this session), (c) the
effect on subsequent SPM-style amplitude and timing tests (false positives
created by registration, power gained), (d) sensitivity to parameters, (e)
run time. Includes the derivative-DTW + warp-smoothing and affine-cover
fixes as small methodological contributions, and the multivariate joint
registration versus single-component-driven registration comparison.

**Novelty.** Comparisons exist within FDA (e.g. Marron et al. 2015 review;
Wu & Srivastava; Kneip & Ramsay) but not with biomechanical constraints and
downstream inference in view, and not with an amplitude-varying dataset like
running at four speeds. Suitable for *Gait & Posture*, *Journal of
Biomechanics*, or *PeerJ*. Effort: moderate; most of the machinery exists;
the work is in the simulation design and the writing.

## 4. Short methodological note — "Derivative-continuous dynamic time warping for physiological signals"

**Question.** DTW is widely used on physiological and biomechanical signals
despite producing warps with discontinuous slopes and flat segments.

**Content.** A small, self-contained contribution: (a) show the artefact
(registered first derivatives after plain DTW are discontinuous), (b)
combine slope-constrained step patterns, derivative DTW and the
square-root-slope smoothing of the DTW warp (`smooth_warp`, which keeps the
warp diffeomorphic and preserves the total warping), (c) show on simulated
and real data that this recovers the accuracy of DTW while giving
physically plausible warps, and compare with SRSF alignment. Could also be a
section of paper 3 rather than a paper on its own; as a stand-alone note it
would fit *Biomedical Signal Processing and Control* or an *IEEE EMBC*
paper. Effort: low-moderate. Novelty: modest but real — I am not aware of
the psi-smoothing construction being used to regularise DTW paths.

## Data and reproducibility across all four

The Dorn2012 data (reduced and three-component) are already bundled;
simulated datasets A/B are bundled; every figure can be produced by a
notebook. For any of the papers, a frozen `reg1d` release with a Zenodo DOI
and the notebooks as supplementary material would make the paper fully
reproducible from `pip install reg1d`.
