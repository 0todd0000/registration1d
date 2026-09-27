'''
reg1d: linear and nonlinear registration of one-dimensional data.

Quick start:

>>> import reg1d
>>> dataset = reg1d.data.Dorn2012()
>>> yi      = reg1d.register_linear( dataset.y, n=101 )   # (J,101) linearly registered
>>> yr, wf  = reg1d.register_srsf( yi, max_iter=5 )       # nonlinearly registered + warps

Registration methods (all return a RegistrationResult):

    register_linear      interpolate to a common grid (returns an array)
    register_shift       time shift (linear)
    register_affine      time shift + uniform stretch (linear)
    register_srsf        elastic / square-root slope function (nonlinear)
    register_dtw         dynamic time warping (nonlinear)
    register_landmark    landmark registration (nonlinear)
    register_continuous  penalised least-squares smooth warps (nonlinear)

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

__version__ = '0.0.1'

from . import warp
from . import linear
from . import srsf
from . import dtw
from . import landmark
from . import continuous
from . import reg
from . import data
from . import plot

from .warp import Warp1D, Warp1DList, random_warp
from .reg import (RegistrationResult, register, register_linear, register_shift,
	register_affine, register_srsf, register_dtw, register_landmark, register_continuous)
