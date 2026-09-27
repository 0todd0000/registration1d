# registration1d

Linear and nonlinear registration of one-dimensional data.

`registration1d` aligns sets of 1D observations (e.g. biomechanical time series) in time.
It depends only on `numpy`, `scipy` and `matplotlib`; the registration
algorithms are implemented from their mathematical definitions, so no other
registration package needs to be installed.

## Installation (development)

    pip install -r requirements.txt
    pip install -e .        # or add the repository folder to sys.path

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
for the licensing status of all third-party code, and the notebooks in
`notebooks/` (HTML renderings in `notebooks/html/`) for demonstrations.

## Tests

    python -m pytest tests

## License

GNU General Public License v3.0 or later. See `LICENSE`.
