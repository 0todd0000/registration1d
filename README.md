# registration1d

Linear and nonlinear registration of one-dimensional data.

`registration1d` aligns sets of 1D observations (e.g. biomechanical time series) in time.
It depends only on `numpy`, `scipy` and `matplotlib`; the registration
algorithms are implemented from their mathematical definitions, so no other
registration package needs to be installed.



**AI disclosure**:  this repository was co-developed with Claude Fable 5.1.  The package concept was human, but Claude did all of the algorithm coding, with only high-level human guidance. Claude's contributions can be seen in the commit list. In interest of full disclosure, a full transcript of the Claude conversation that produced this repository and its developments will be provided at a later date, when we are ready to create a release version.





## Installation (development)

    git clone https://github.com/0todd0000/registration1d.git
    cd registration1d
    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -e ".[dev,qt]" PySide6 # editable install + test/notebook tools + PyQtGraph backend (or PyQt6)
    pytest                             # runs tests/ (the PyQtGraph test is skipped if pyqtgraph is absent)

Requires Python 3.10 or later. Minimal install (numpy, scipy, matplotlib only): `pip install -e .`.
Extras: `[qt]` adds pyqtgraph for `registration1d.plotqt` (a Qt binding such
as PyQt6 or PySide6 must be installed separately); `[dev]` adds pytest and
the notebook tools.

## Quick start

The recommended import convention is `import registration1d as reg1d`;
the short alias is used throughout the documentation and notebooks.

    import registration1d as reg1d
    
    dataset = reg1d.data.Dorn2012()                     # 8 observations of unequal length
    yi      = reg1d.register_linear(dataset.y, n=101).y # (8,101) linearly registered
    yr, wf  = reg1d.register_srsf(yi, max_iter=5)       # nonlinearly registered + warps
    
    result  = reg1d.register_srsf(yi, t=np.linspace(0, 100, 101))   # explicit time grid (% stance)
    result.plot(group=dataset.group)                    # before / after / warps
    result.apply(other_variable)                        # same warps applied to another (8,101) variable
    result.displacement_fields_t                        # displacement fields in % stance

## Methods

| function | type | description |
|---|---|---|
| `register_linear` | linear | interpolate to a common number of points |
| `register_shift` | linear | least-squares time shift |
| `register_affine` | linear | least-squares time shift and stretch |
| `register_srsf` | nonlinear | elastic (square-root slope function) registration, dynamic programming, Karcher-mean template |
| `register_dtw` | nonlinear | dynamic time warping (step patterns, window, derivative DTW, DBA, warp smoothing) |
| `register_landmark` | nonlinear | landmark registration with monotone-cubic warps |
| `register_continuous` | nonlinear | Ramsay-Li penalised least-squares smooth warps |
| `register_sim` | nonlinear | self-modelling / shape-invariant model (amplitude + smooth warp) |
| `register_pairwise` | nonlinear | pairwise synchronisation (template-free) |
| `register_bayes` | nonlinear | Bayesian registration (posterior samples and credible bands of the warps) |

All functions return a `RegistrationResult` (`islinear` tells linear from
nonlinear); multivariate (J,Q,D) input is supported by `register_srsf`.
Every nonlinear method takes `center=` ('karcher' | 'pointwise' | 'anchor' |
'none'; see the WarpCentering notebook) and `register_srsf` uses a
data-adaptive elasticity penalty (`lam='auto'`; `lam=0` reproduces fdasrsf).
Observations of different lengths can be registered in **real time**
without prior resampling by passing them as a list with their sampling
interval, e.g. `register_srsf(ylist, t='fs=1000')` (see `reg1d.realtime`
and the RealTimeRegistration notebook).
`reg1d.stats` provides permutation-based two-sample tests on registered data
and displacement fields (the nlreg1d timing analysis).

See `ALGORITHMS.md` for the provenance of each algorithm, `GPL-COMPLIANCE.md`
for the licensing status of all third-party code, `SUMMARY.md` for development
notes, `TESTS.md` for the test strategy, and the notebooks in `notebooks/`
(HTML renderings in `notebooks/html/`) for demonstrations. The package
source is in `src/registration1d/`; the draft software paper is in `paper/`.

## Tests

    python -m pytest tests

## License

GNU General Public License v3.0 or later. See `LICENSE`.
