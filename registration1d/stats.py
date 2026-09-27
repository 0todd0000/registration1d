'''
Small statistics helpers for hypothesis tests on registered data and on
displacement fields (two-sample t statistics with nonparametric,
permutation-based inference), so that the timing analysis of nlreg1d
(registered amplitude test + displacement-field test) can be reproduced
without further dependencies. For full random field theory inference use
spm1d.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np



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
