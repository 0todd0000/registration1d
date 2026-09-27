'''
Example datasets.

Dorn2012: anteroposterior ground reaction forces (N) during running at
four speeds (coded 0-3, increasing), two trials per speed, sampled at
the original (unequal) number of frames. The data are a reduced subset
of:

    Dorn TW, Schache AG, Pandy MG (2012). Muscular strategy shift in
        human running: dependence of running speed on hip and ankle
        muscle performance. Journal of Experimental Biology 215: 1944-1956.

as distributed (MIT licence) with the nlreg1d repository:

    https://github.com/0todd0000/nlreg1d   (Data/Dorn2021-reduced.npz)

Dorn2012MV: the three-component (anteroposterior, vertical, mediolateral)
forces for 18 trials (Data/Dorn2021-orig.npz in the same repository).

SimulatedA / SimulatedB: the two-group simulated datasets of the nlreg1d
paper (Data/SimulatedA.csv, SimulatedB.csv): A has a pure amplitude effect,
B a pure timing effect.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import os, pathlib
import numpy as np


dirDATA = os.path.join( str(pathlib.Path(__file__).parent), 'data' )



class Dorn2012(object):
    '''
    Dorn et al. (2012) anteroposterior GRF dataset.

    >>> dataset = reg1d.data.Dorn2012()
    >>> y       = dataset.y        # object array of 8 arrays with 185-383 frames each
    >>> speed   = dataset.group    # speed codes 0,1,2,3

    Because the observations have different lengths they must be linearly
    registered (dataset.resample(n)) before nonlinear registration.
    '''
    fpath = os.path.join(dirDATA, 'Dorn2012-reduced.npz')
    name  = 'Dorn2012'

    def __init__(self):
        with np.load(self.fpath, allow_pickle=True) as z:
            self.y     = z['y']
            self.group = np.asarray(z['speed'], dtype=int)

    def __repr__(self):
        s  = f'Dataset: {self.name}\n'
        s += f'    J       = {self.J}\n'
        s += f'    lengths = {self.lengths}\n'
        s += f'    groups  = {np.unique(self.group).tolist()}\n'
        return s

    @property
    def J(self):
        return len(self.y)

    @property
    def dv(self):   # nlreg1d-compatible alias
        return self.y

    @property
    def lengths(self):
        return [yy.size  for yy in self.y]

    def resample(self, n=101):
        from .linear import resample
        return resample(self.y, n)



class Dorn2012MV(object):
    '''
    Dorn et al. (2012) three-component GRF dataset (18 trials: four speeds,
    left and right feet). Components: 0 = anteroposterior, 1 = vertical,
    2 = mediolateral (N).

    >>> dataset = reg1d.data.Dorn2012MV()
    >>> Y       = dataset.resample(101)     # (18,101,3) array
    >>> speed   = dataset.speed             # m/s
    '''
    fpath = os.path.join(dirDATA, 'Dorn2012-3D.npz')
    name  = 'Dorn2012MV'
    components = ('anteroposterior', 'vertical', 'mediolateral')

    def __init__(self):
        with np.load(self.fpath, allow_pickle=True) as z:
            self.y     = z['Y']
            self.foot  = np.asarray(z['FOOT'], dtype=int)
            self.speed = np.asarray(z['SPEED'], dtype=float)
        self.group = np.unique(self.speed, return_inverse=True)[1]

    def __repr__(self):
        return f'Dataset: {self.name}\n    J = {self.J}\n    speeds = {np.unique(self.speed).tolist()}\n'

    @property
    def J(self):
        return len(self.y)

    def resample(self, n=101):
        from .linear import resample
        return np.array([resample(np.asarray(a, dtype=float).T, n).T  for a in self.y])



class _CSVDataset(object):
    '''Two-group simulated datasets from the nlreg1d repository (first column = group).'''
    fname = None

    def __init__(self):
        a          = np.loadtxt(os.path.join(dirDATA, self.fname), delimiter=',')
        self.group = np.asarray(a[:, 0], dtype=int)
        self.y     = a[:, 1:]

    @property
    def dv(self):
        return self.y

    @property
    def J(self):
        return self.y.shape[0]

    @property
    def Q(self):
        return self.y.shape[1]


class SimulatedA(_CSVDataset):
    '''Simulated two-group dataset with a pure AMPLITUDE effect (nlreg1d Data/SimulatedA.csv).'''
    fname = 'SimulatedA.csv'
    name  = 'SimulatedA'


class SimulatedB(_CSVDataset):
    '''Simulated two-group dataset with a pure TIMING effect (nlreg1d Data/SimulatedB.csv).'''
    fname = 'SimulatedB.csv'
    name  = 'SimulatedB'



def load_dorn2012():
    '''Return (y, group) for the Dorn2012 dataset.'''
    d = Dorn2012()
    return d.y, d.group
