'''
Helpers for the registration1d notebooks and the paper figures: example
datasets and the permutation tests used in the demonstrations. None of
this is part of the registration1d package, which contains registration
algorithms only.

Datasets
--------
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

Statistics
----------
Two-sample t statistics with permutation-based inference on registered
data and on displacement fields (`ttest2`, `permutation_ttest2`,
`timing_test`), so that the timing analysis of nlreg1d (registered
amplitude test + displacement-field test) can be reproduced without
further dependencies. For random field theory inference use spm1d.

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

    >>> dataset = util.Dorn2012()
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
        from registration1d.linear import resample
        return resample(self.y, n)



class Dorn2012MV(object):
    '''
    Dorn et al. (2012) three-component GRF dataset (18 trials: four speeds,
    left and right feet). Components: 0 = anteroposterior, 1 = vertical,
    2 = mediolateral (N).

    >>> dataset = util.Dorn2012MV()
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
        from registration1d.linear import resample
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



# ---------------------------------------------------------------- statistics





def ttest2(yA, yB):
    '''Pointwise two-sample t statistic (equal variances), (Q,) array.'''
    yA, yB = np.asarray(yA, dtype=float), np.asarray(yB, dtype=float)
    nA, nB = yA.shape[0], yB.shape[0]
    mA, mB = yA.mean(axis=0), yB.mean(axis=0)
    sp2    = (((yA - mA)**2).sum(axis=0) + ((yB - mB)**2).sum(axis=0)) / (nA + nB - 2)
    return (mA - mB) / np.sqrt(sp2 * (1.0/nA + 1.0/nB) + 1e-300)


def permutation_ttest2(yA, yB, n_perm=1000, alpha=0.05, two_tailed=True, random_state=None):
    '''
    Nonparametric two-sample test on 1D data: the critical threshold is the
    (1-alpha) quantile of the permutation distribution of max |t| (or max t)
    over the domain (SnPM-style "tmax" inference).

    Returns a dict with 't' (Q,), 'threshold', 'p' (p value of the observed
    max |t|), 'clusters' (list of (start, end) index ranges exceeding the
    threshold), 'tmax_perm' (n_perm,).
    '''
    rng    = np.random.default_rng(random_state)
    yA, yB = np.asarray(yA, dtype=float), np.asarray(yB, dtype=float)
    nA     = yA.shape[0]
    y      = np.vstack([yA, yB])
    t      = ttest2(yA, yB)
    stat   = (lambda a: np.abs(a).max()) if two_tailed else (lambda a: a.max())
    tmax   = np.empty(n_perm)
    for k in range(n_perm):
        ind = rng.permutation(y.shape[0])
        tmax[k] = stat(ttest2(y[ind[:nA]], y[ind[nA:]]))
    thr    = float(np.percentile(tmax, 100*(1-alpha)))
    p      = float((tmax >= stat(t)).mean())
    excess = (np.abs(t) if two_tailed else t) > thr
    return dict(t=t, threshold=thr, p=p, clusters=_clusters(excess), tmax_perm=tmax)


def _clusters(mask):
    '''(start, end) index ranges of runs of True.'''
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m) and start is not None:
            out.append((start, i-1)); start = None
    if start is not None:
        out.append((start, len(mask)-1))
    return out


def timing_test(result, group, n_perm=1000, alpha=0.05, random_state=None):
    '''
    nlreg1d-style two-group timing analysis of a RegistrationResult: a
    permutation two-sample test on the displacement fields (timing effect)
    and on the registered observations (amplitude effect).

    *group* : (J,) array with exactly two unique labels

    Returns (test_amplitude, test_timing), each a dict from permutation_ttest2.
    '''
    group = np.asarray(group)
    ug    = np.unique(group)
    if ug.size != 2:
        raise ValueError('group must contain exactly two labels')
    d     = result.displacement_fields
    y     = result.y
    A, B  = group == ug[0], group == ug[1]
    ta    = permutation_ttest2(y[A], y[B], n_perm=n_perm, alpha=alpha, random_state=random_state)
    tt    = permutation_ttest2(d[A], d[B], n_perm=n_perm, alpha=alpha, random_state=random_state)
    return ta, tt
