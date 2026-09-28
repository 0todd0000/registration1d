# GPL compliance status of third-party code

`registration1d` is to be released under the GNU General Public License v3.0 or later
(GPL-3.0-or-later; the full text is in `LICENSE`). This document records the
licensing status of every third-party package that `registration1d` imports, every
package whose code was consulted, and every package that was *not* used and
why. Compatibility judgements follow the Free Software Foundation's list of
GPL-compatible licenses (https://www.gnu.org/licenses/license-list.html).
License data were checked on 2026-09-27 against the packages' PyPI metadata
and, where noted, their source distributions.

## 1. Packages imported by registration1d (runtime dependencies)

| package | version checked | license | GPLv3-compatible | notes |
|---|---|---|---|---|
| numpy | 2.5 | BSD-3-Clause (with 0BSD / MIT / Zlib / CC0 components) | yes | permissive |
| scipy | 1.18 | BSD-3-Clause | yes | permissive; `interpolate`, `integrate`, `optimize`, `signal` are used |
| matplotlib | 3.11 | Matplotlib license (PSF-based) | yes | permissive; used only by `reg1d.plot` and the `plot` methods |

No other package is imported by `registration1d` at runtime. The modules added in
the second session (`bayes.py`, `pairwise.py`, `sim.py`, `realtime.py`, the
multivariate SRSF, derivative DTW, DBA, warp smoothing, banded / refined
DP, the `cover` constraint) introduce no new imports: they use numpy and
scipy only (`scipy.integrate`, `scipy.optimize`, `scipy.ndimage`,
`scipy.interpolate.make_smoothing_spline`) and `concurrent.futures` from
the standard library. numba was used only to *measure* a possible
speed-up (see SUMMARY.md 2.6) and is not imported. In particular `fdasrsf`,
`scikit-fda`, `dtw-python`, `dtaidistance`, `numba`, `Cython`, `cffi`,
`pandas` and `scikit-learn` are **not** imported and do not need to be
installed.

Because all three runtime dependencies are permissively licensed, releasing
`registration1d` under GPL-3.0-or-later imposes no additional obligation beyond those
of the GPL itself (retain the license text, mark modified files, provide
source).

## 1a. Optional dependencies (imported only on demand)

| package | version checked | license | GPLv3-compatible | notes |
|---|---|---|---|---|
| pyqtgraph | 0.14 | MIT | yes | imported only by `registration1d.plotqt`; installed with the `[qt]` extra |
| PyQt6 / PySide6 (a Qt binding, chosen by the user) | 6.x | PyQt6: GPL-3.0 (or commercial); PySide6: LGPL-3.0 | yes (both) | required by pyqtgraph, not by registration1d itself; never imported directly |

## 2. Development-only dependencies (not imported by registration1d)

| package | license | GPLv3-compatible | use |
|---|---|---|---|
| pytest | MIT | yes | running `tests/` |
| jupyter / nbformat / nbconvert / ipykernel | BSD-3-Clause | yes | building the notebooks |

## 3. Third-party registration packages: consulted, not copied

`registration1d` was written from the published mathematics (see `ALGORITHMS.md`). No
source code from the packages below was copied into `registration1d`, and none of
them is imported. They were used as follows.

| package | license | GPLv3-compatible | how it was used |
|---|---|---|---|
| fdasrsf (`fdasrsf_python`, J.D. Tucker) | BSD-3-Clause | yes | **validation only**: `reg1d.register_srsf` was run side by side with `fdasrsf.fdawarp.srsf_align` on the Dorn2012 data and on simulated data (notebook 1, optional cell). The maximum absolute difference between the warps of the two implementations is about 0.01 (one observation: 0.06) on the unit interval, with identical alignment of the propulsive peak. The public API of `srsf_align` (keyword names `MaxItr`, `center`, `lam`, `smoothdata`) informed the naming of `register_srsf`'s keyword arguments. |
| dtw-python (T. Giorgino) | GPL-3.0-or-later (source distribution `COPYING`, `pyproject.toml`) | yes (same license) | **validation only**: `reg1d.dtw.dtw_path` reproduces `dtw.dtw` exactly (identical path and identical accumulated distance) for the `symmetric1`, `symmetric2` and `asymmetric` step patterns and for the Sakoe-Chiba window. Step-pattern names follow Giorgino (2009). Note that although this package *could* be imported by a GPL project, it is not, because the algorithm is short and a dependency-free implementation was preferred. |
| scikit-fda (GAA-UAM) | BSD-3-Clause | yes | not installed and not used. Its `ElasticRegistration` is itself a wrapper around `fdasrsf`; its `LeastSquaresShiftRegistration` and `landmark_elastic_registration` correspond to `reg1d.register_shift` and `reg1d.register_landmark`, which were implemented from Ramsay & Silverman (2005) instead. scikit-fda's own dependency tree (fdasrsf BSD, dcor MIT, findiff MIT, rdata MIT, scikit-datasets MIT, multimethod Apache-2.0, lazy-loader BSD, pandas BSD, scikit-learn BSD) is entirely GPL-compatible, so it could be added as an optional dependency later if desired. |
| dtaidistance (KU Leuven DTAI) | Apache-2.0 | yes (Apache-2.0 is compatible with GPLv3, not GPLv2) | not used; noted as an alternative fast DTW implementation for future work. |
| nlreg1d (T. Pataky) | MIT | yes | the reference for the target workflow. Its `Warp1D` / `Warp1DList` interface and `register_linear` / `register_srsf` names are mirrored in `registration1d` for continuity, but the implementations in `reg1d.warp` are new (nlreg1d's `random_warp` wrapped scikit-fda and its displacement fields wrapped scipy interpolation; `registration1d` computes both directly). |

If, in the future, `fdasrsf` or `dtw-python` code were to be copied into
`registration1d` rather than re-derived, both licenses would permit it: BSD-3-Clause
code may be included in a GPL work provided the BSD copyright notice and
disclaimer are retained, and GPL-3.0-or-later code may be included as is.

## 4. Data

`registration1d/data/Dorn2012-reduced.npz` and `registration1d/data/Dorn2012-3D.npz` are
copies of `Data/Dorn2021-reduced.npz` and `Data/Dorn2021-orig.npz` from the
nlreg1d repository (MIT): ground reaction force data of Dorn, Schache &
Pandy (2012), *J Exp Biol* 215:1944-1956. `SimulatedA.csv` and
`SimulatedB.csv` are the simulated two-group datasets of the nlreg1d paper,
from the same repository. They are included for demonstration and testing.
Their inclusion under the GPL is permitted by the MIT license; the source is
credited in `notebooks/util.py`. (The datasets are not part of the
installed package: they live in `notebooks/data/`.)

## 5. Summary

- Every module in `registration1d` carries a GPL-3.0-or-later header and the full
  license text is distributed in `LICENSE`.
- All runtime dependencies (numpy, scipy, matplotlib) are permissively
  licensed and GPL-compatible.
- No third-party registration code is imported or copied; `fdasrsf` and
  `dtw-python` were used only to verify numerical agreement.
- Nothing in the package or its dependencies prevents release under
  GPL-3.0-or-later.
