# reg1d

Linear and nonlinear registration of one-dimensional data.

`reg1d` aligns sets of 1D observations (e.g. biomechanical time series) in time.
It depends only on `numpy`, `scipy` and `matplotlib`; the registration
algorithms are implemented from their mathematical definitions, so no other
registration package needs to be installed.

## Installation (development)

    pip install -r requirements.txt
    pip install -e .        # or add the repository folder to sys.path

## Quick start

    import reg1d

    dataset = reg1d.data.Dorn2012()                 # 8 observations of unequal length
    yi      = reg1d.register_linear(dataset.y, n=101)   # (8,101) linearly registered
    yr, wf  = reg1d.register_srsf(yi, max_iter=5)      # nonlinearly registered + warps

    result  = reg1d.register_srsf(yi)               # full result object
    result.plot(group=dataset.group)                # before / after / warps

## Methods

| function | type | description |
|---|---|---|
| `register_linear` | linear | interpolate to a common number of points |
| `register_shift` | linear | least-squares time shift |
| `register_affine` | linear | least-squares time shift and stretch |
| `register_srsf` | nonlinear | elastic (square-root slope function) registration, dynamic programming, Karcher-mean template |
| `register_dtw` | nonlinear | dynamic time warping with step patterns and optional window |
| `register_landmark` | nonlinear | landmark registration with monotone-cubic warps |
| `register_continuous` | nonlinear | Ramsay-Li penalised least-squares smooth warps |

See `ALGORITHMS.md` for the provenance of each algorithm, `GPL-COMPLIANCE.md`
for the licensing status of all third-party code, and the notebooks in
`notebooks/` (HTML renderings in `notebooks/html/`) for demonstrations.

## Tests

    python -m pytest tests

## License

GNU General Public License v3.0 or later. See `LICENSE`.
