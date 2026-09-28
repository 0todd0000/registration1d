'''
registration1d: linear and nonlinear registration of one-dimensional data.

Quick start:

The recommended import convention is  import registration1d as reg1d

>>> import registration1d as reg1d
>>> yi      = reg1d.register_linear( y, n=101 ).y         # (J,101) linearly registered
>>> yr, wf  = reg1d.register_srsf( yi, max_iter=5 )       # nonlinearly registered + warps

where y is a (J,Q) array or a list of J observations of different lengths.

Registration methods (all return a RegistrationResult):

    register_linear      interpolate to a common grid
    register_shift       time shift (linear)
    register_affine      time shift + uniform stretch (linear)
    register_srsf        elastic / square-root slope function (nonlinear)
    register_dtw         dynamic time warping (nonlinear)
    register_landmark    landmark registration (nonlinear)
    register_continuous  penalised least-squares smooth warps (nonlinear)
    register_bayes       Bayesian registration with posterior warp samples (nonlinear)
    register_pairwise    pairwise synchronisation, template-free (nonlinear)
    register_sim         self-modelling / shape-invariant model (nonlinear)

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
'''

__version__ = "0.0.2"

from . import warp
from . import linear
from . import srsf
from . import dtw
from . import landmark
from . import continuous
from . import bayes
from . import pairwise
from . import sim
from . import realtime
from . import reg
from . import plot
# registration1d.plotqt (PyQtGraph backend) is imported on demand: it needs the optional [qt] extra

from .warp import Warp1D, Warp1DList, random_warp
from .reg import (RegistrationResult, LinearRegistrationResult, NonlinearRegistrationResult,
    register, register_linear, register_shift, register_affine, register_srsf, register_dtw,
    register_landmark, register_continuous, register_bayes, register_pairwise, register_sim)
