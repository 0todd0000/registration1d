# Tasks towards a JOSS submission of registration1d

Status: 2026-09-27. JOSS requirements were checked against
joss.readthedocs.io on this date (Submitting a paper; Example paper).
Three of them shape the plan more than anything else and are put first.

## 0. Three gating requirements (read before anything else)

1. **Six months of public, iterative development.** JOSS desk-rejects
   submissions whose repository has not been public for more than six
   months with active, iterative development over that period (not a
   single burst of commits). The public repository should therefore be
   created as soon as the code is in a presentable state, and developed
   *in public* from then on; the earliest realistic submission date is six
   months after the first public commit.
2. **Demonstrated research impact.** JOSS now requires evidence that the
   software is being used for research beyond the authors' own work
   (preprints or papers citing it, adoption by other groups, integration
   into research workflows) and a *Research impact statement* section in
   the paper. The planned methods paper (PAPER-IDEAS.md, idea 2) and any
   collaborator analyses during the six-month window are the natural
   sources; a preprint using registration1d posted before the JOSS submission would
   satisfy this.
3. **AI usage disclosure.** JOSS papers must contain an *AI usage
   disclosure* section stating which generative tools (and versions) were
   used, for what (code generation, tests, documentation, drafting), and
   asserting that the human author reviewed, edited and validated all such
   output; JOSS treats an incomplete disclosure as an ethical breach with
   consequences up to withdrawal. This applies to the paper and to the
   code in the public repository. Deciding how to word this section is the
   author's decision and must be made before the repository goes public,
   because the disclosure has to match the repository's history.

## 1. Repository

### 1.1 Proposed structure of the new GitHub repository `registration1d`

    registration1d/                          <- repository root (public; GPL-3.0-or-later)
    ├── LICENSE                     GPL-3.0-or-later full text
    ├── README.md                   what it is, install, 10-line example, link to docs, how to cite
    ├── CITATION.cff                citation metadata (JOSS reads this; Zenodo uses it)
    ├── CONTRIBUTING.md             how to report issues, propose changes, run tests, style (spaces, 4)
    ├── CODE_OF_CONDUCT.md          Contributor Covenant (expected by JOSS reviewers)
    ├── CHANGELOG.md                one entry per tagged release
    ├── pyproject.toml              build metadata (already drafted; add urls, classifiers, optional extras)
    ├── requirements.txt            runtime deps (numpy, scipy, matplotlib); dev deps in pyproject extras
    ├── .gitignore
    ├── .github/
    │   ├── workflows/
    │   │   ├── tests.yml           GitHub Actions: pytest on Linux + macOS, Python 3.9-3.13, on push/PR
    │   │   ├── notebooks.yml       execute notebooks weekly / on release (slow tier)
    │   │   └── docs.yml            build the documentation site
    │   ├── ISSUE_TEMPLATE/         bug report, feature request (issue tracker must be open)
    │   └── PULL_REQUEST_TEMPLATE.md
    ├── src/registration1d/         the package (src layout)
    │   ├── __init__.py  warp.py  srsf.py  dtw.py  landmark.py  continuous.py  sim.py
    │   ├── pairwise.py  bayes.py  realtime.py  linear.py  stats.py  reg.py  data.py  plot.py
    │   └── data/                   Dorn2012-reduced.npz, Dorn2012-3D.npz, SimulatedA.csv, SimulatedB.csv
    ├── tests/                      pytest suite split by TESTS.md tiers
    │   ├── test_warp.py  test_methods.py  test_edge_cases.py  test_reference.py  test_slow.py  test_plot.py
    │   └── conftest.py             shared fixtures (Dorn2012, simulated warps)
    ├── docs/                       documentation source (Sphinx + MyST or MkDocs + mkdocstrings)
    │   ├── index.md  install.md  quickstart.md  methods.md  api/  notebooks/  centering.md  realtime.md
    │   └── conf.py / mkdocs.yml
    ├── examples/                   short runnable scripts (one per method), used in the docs gallery
    ├── notebooks/                  the executed notebooks (ipynb; html renderings are built by docs, not committed)
    ├── benchmarks/                 timing scripts (pure numpy vs parallel; numba experiment)
    └── paper/                      JOSS paper: paper.md, paper.bib, figures/, make_figures.py
                                    (JOSS requires paper.md and paper.bib in the software repository)

Notes on the structure:

- `paper/` is the conventional JOSS location; the JOSS build action
  (`openjournals/openjournals-draft-action`) looks for `paper/paper.md`
  by default.
- The `notebooks/html` renderings should not live in git (they are 1-2 MB
  each and change on every execution); the documentation build renders
  them instead (nbsphinx / myst-nb).
- `SUMMARY.md`, `ALGORITHMS.md`, `TESTS.md` and `GPL-COMPLIANCE.md`
  contain material worth carrying over into `docs/` (methods provenance,
  licensing, testing strategy) after editing.

### 1.2 Repository set-up tasks

- [x] Create the public repository (github.com/0todd0000/registration1d).
- [ ] Decide the AI-usage wording (0.3).
- [x] Import code (full development history carried over).
- [ ] Tag `v0.1.0` once tests and docs pass on GitHub Actions.
- [ ] Enable issues and discussions; add issue templates.
- [ ] Branch protection on `main`; develop through pull requests so that the history shows iterative work (JOSS checks this).
- [ ] Add `CITATION.cff` (name, version, DOI placeholder, authors, license, repository URL).
- [ ] Register on PyPI (`pip install registration1d`) and, optionally, conda-forge.
- [ ] Zenodo: link the GitHub repository so that each release is archived with a DOI; JOSS needs the DOI of the accepted version at the end of the review.

## 2. Package

- [ ] Freeze the public API for v0.1: names of `register_*`, `RegistrationResult` attributes, `center`/`lam`/`t` semantics; decide on the tuple-unpacking behaviour (SUMMARY 3.5).
- [ ] Type hints and NumPy-style docstrings on all public functions (the docs build depends on them).
- [ ] Input validation with clear error messages (NaN, Q < 10, mismatched shapes, non-monotone landmarks, infeasible bands).
- [ ] Non-uniform-grid handling: document that non-uniform `t` is resampled, or implement exact handling.
- [ ] Bayesian module: calibrate the credible intervals (whitened / GP error model) or label the module experimental in the docs.
- [ ] Optional numba kernel behind `try/except ImportError` (SUMMARY 2.6) — optional, only if benchmarks show a need.
- [ ] Deterministic `random_state` plumbing wherever randomness is used (Bayesian sampler, permutation tests, random warps).
- [ ] Version string in one place (`registration1d/__init__.py` read by `pyproject.toml`).

## 3. Tests and continuous integration (GitHub Actions)

- [ ] Split `tests/test_reg1d.py` into the tiers of TESTS.md; add the edge-case, plotting and reference-agreement groups (the latter `importorskip`-guarded for `fdasrsf` and `dtw-python`).
- [ ] GitHub Actions matrix: Linux and macOS × Python 3.9–3.13; fast tier on every push, slow tier and notebook execution nightly or on release.
- [ ] Coverage report (pytest-cov) with a badge; aim for > 90 % of the package.
- [ ] A test that executes every `examples/*.py`.

## 4. Documentation

- [ ] Choose the toolchain (MkDocs + Material + mkdocstrings is the lighter option; Sphinx + MyST if LaTeX-heavy method pages are wanted) and host on GitHub Pages or Read the Docs.
- [ ] Pages: installation; quick start (the nlreg1d workflow in 10 lines); method guide (one page per method with the mathematics, parameters, when to use it — adapted from ALGORITHMS.md and the notebook texts); warp centring; real-time registration; statistics helpers; API reference (auto-generated); executed notebooks; licensing of third-party work (from GPL-COMPLIANCE.md); how to cite.
- [ ] A "which method should I use" decision table (from the notebook 2 summary).
- [ ] README: badges (tests, docs, PyPI, DOI), a figure, and a minimal example that runs after `pip install registration1d`.

## 5. Paper (`paper/`)

- [ ] Fill in ORCID and affiliation details; decide the author list (JOSS: anyone with a substantive contribution).
- [ ] Write the *Research impact statement* with concrete evidence (0.2).
- [ ] Write the *AI usage disclosure* (0.3).
- [ ] Check the word count: JOSS asks for 750–1750 words excluding references; the draft is about 1,070 including the bracketed placeholders.
- [ ] Verify every reference in `paper.bib` (DOIs, years, page numbers) against the publishers' pages; add the `fdasrvf` R package reference if it is kept in the text.
- [ ] Regenerate `figures/methods.png` from the released version of the package; consider replacing it with a two-panel figure (before/after + displacement fields, `figures/srsf.png`) if the eight-panel figure is judged too dense for the JOSS page format.
- [ ] Build the PDF locally with the JOSS Docker image or the `openjournals/openjournals-draft-action` GitHub Action, and check the figure, references and YAML.
- [ ] Optional: post the paper as a preprint (allowed by JOSS) once the repository is public.

## 6. Evidence of use during the six-month window

- [ ] Use `registration1d` in the planned methods paper (PAPER-IDEAS.md, idea 2 or 3) and post it as a preprint; cite the repository/Zenodo DOI.
- [ ] Replace the `nlreg1d` dependency chain in any ongoing analyses with `registration1d` and note it in those outputs.
- [ ] Invite one or two colleagues to use it on their data and to open issues; external issues and pull requests are visible evidence of adoption.
- [ ] Teaching use (lecture notebooks) counts as integration into a workflow if it is public.

## 7. Submission checklist (at month six or later)

- [ ] Repository public ≥ 6 months, with a commit history spanning that period.
- [ ] Tagged release, PyPI package, Zenodo DOI for that release.
- [ ] Tests pass on GitHub Actions; documentation builds and is linked from the README.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, open issue tracker.
- [ ] `paper/paper.md` + `paper/paper.bib` build cleanly; all six required sections present (Summary, Statement of need, State of the field, Software design, Research impact statement, AI usage disclosure) plus References.
- [ ] Conflicts of interest and suggested reviewers prepared for the submission form.
- [ ] Submit at joss.theoj.org; expect to respond to reviewer comments within two weeks and complete requested changes within four to six weeks.
