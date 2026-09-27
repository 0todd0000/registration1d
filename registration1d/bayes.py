'''
Bayesian registration.

A simplified implementation of the Bayesian pairwise registration model of

    Cheng W, Dryden IL, Huang X (2016). Bayesian registration of functions
        and curves. Bayesian Analysis 11: 447-475.
    Lu Y, Herbei R, Kurtek S (2017). Bayesian registration of functions
        with a Gaussian process prior. Journal of Computational and
        Graphical Statistics 26: 894-904.

Model
-----
Let q1 be the SRSF of the template and q2 the SRSF of the observation to be
registered. The warp gamma is parameterised through a tangent vector v at
the identity of the unit sphere of square-root slopes,

    v(t)   = sum_k c_k phi_k(t),    phi_k = sqrt(2) cos(k pi t),  k = 1..K
    psi    = cos(|v|) + sin(|v|) v / |v|            (exponential map at psi = 1)
    gamma  = int_0^t psi(s)^2 ds

so that every coefficient vector c gives a valid warp. The likelihood is a
Gaussian error model in SRSF space,

    q1  =  (q2 o gamma) sqrt(gamma')  +  epsilon,    epsilon ~ N(0, sigma^2)

with prior c_k ~ N(0, tau^2 / k^2) (smooth warps a priori; Cheng et al. use
a Z-mixture prior, Lu et al. a Gaussian-process prior of which this is a
finite-basis special case) and sigma^2 ~ InverseGamma(a, b).

Inference
---------
c is sampled with the preconditioned Crank-Nicolson (pCN) Metropolis
proposal  c' = sqrt(1 - beta^2) c + beta xi,  xi ~ prior,  which is
reversible with respect to the prior so that the acceptance probability
depends only on the likelihood; sigma^2 is updated by Gibbs sampling. The
output is a set of posterior samples of the warp, from which posterior
means and pointwise credible intervals of the warp and the displacement
field can be computed. This quantifies the *uncertainty of the
registration itself*, which point-estimate methods (dynamic programming,
landmarks, ...) do not provide.

Copyright (C) 2026 Todd Pataky

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See the LICENSE file for details.
'''

import numpy as np
from scipy import integrate
from . import warp as _warp
from . import srsf as _srsf



def _basis(t, K):
    k = np.arange(1, K+1)[:, None]
    return np.sqrt(2) * np.cos(k * np.pi * t[None, :])


def coef_to_warp(c, B, t):
    '''Warp from tangent-space coefficients c via the exponential map at the identity.'''
    v   = c @ B
    nv  = np.sqrt(max(integrate.trapezoid(v*v, t), 0.0))
    if nv < 1e-12:
        return t.copy()
    psi = np.cos(nv) + np.sin(nv) * v / nv
    g   = integrate.cumulative_trapezoid(psi**2, t, initial=0)
    return g / g[-1]


def _loglik_kernel(q1, q2, gam, t):
    '''||q1 - (q2, gamma)||^2 (integral form).'''
    d = q1 - _srsf.warp_srsf(q2, gam)
    d2 = d**2 if d.ndim == 1 else (d**2).sum(axis=-1)
    return integrate.trapezoid(d2, t)


def sample_pair(y_template, y, n_samples=2000, burn=500, thin=1, K=8, tau=0.3, beta=0.05,
    adapt=True, a0=1.0, b0=1e-3, c0=None, n_eff=None, smooth=0, random_state=None, return_coef=False):
    '''
    Posterior samples of the warp aligning *y* to *y_template*.

    *n_samples*, *burn*, *thin* : MCMC settings (total iterations = burn + n_samples*thin)
    *K*     : number of cosine basis functions for the tangent vector
    *tau*   : prior scale of the coefficients (c_k ~ N(0, tau^2/k^2));
              larger tau allows stronger warps a priori
    *beta*  : initial pCN step size in (0,1]
    *adapt* : adapt beta during burn-in towards an acceptance rate of ~25 %
    *a0*, *b0* : InverseGamma hyper-parameters of the noise variance
    *n_eff* : effective number of independent observations in the likelihood
              (default Q, the number of grid points). SRSF residuals of
              smooth data are strongly autocorrelated, so Q overstates the
              information and the credible intervals are then optimistic;
              a smaller n_eff (e.g. Q / autocorrelation length) tempers the
              likelihood accordingly.
    *c0*    : initial coefficients (None = identity warp; a good initial value
              is the projection of a dynamic-programming warp, see initial_coef)

    Returns a dict with 'warps' (S,Q) posterior warp samples, 'sigma2' (S,),
    'accept' (acceptance rate after burn-in), 'beta' (final step size),
    'coef' (S,K) if return_coef.
    '''
    rng   = np.random.default_rng(random_state)
    yt    = np.asarray(y_template, dtype=float)
    yy    = np.asarray(y, dtype=float)
    Q     = yt.shape[0]
    t     = _warp.grid(Q)
    B     = _basis(t, K)
    f     = _srsf.srsf_mv if yy.ndim == 2 else _srsf.srsf
    q1, q2 = f(yt, smooth), f(yy, smooth)
    prior_sd = tau / np.arange(1, K+1)
    c     = np.zeros(K) if c0 is None else np.asarray(c0, dtype=float).copy()
    gam   = coef_to_warp(c, B, t)
    E     = _loglik_kernel(q1, q2, gam, t)
    sig2  = max(E, 1e-6)
    n_eff = Q if n_eff is None else float(n_eff)
    keep_w, keep_s, keep_c = [], [], []
    acc   = 0
    acc_w = 0                          # acceptances in the current adaptation window
    total = burn + n_samples*thin
    for it in range(total):
        # --- pCN update of c
        xi    = rng.normal(0, prior_sd)
        cp    = np.sqrt(1 - beta**2) * c + beta * xi
        gp    = coef_to_warp(cp, B, t)
        Ep    = _loglik_kernel(q1, q2, gp, t)
        logr  = -(Ep - E) * n_eff / (2*sig2)
        if np.log(rng.uniform()) < logr:
            c, gam, E = cp, gp, Ep
            acc_w += 1
            if it >= burn:
                acc += 1
        # --- adapt the step size during burn-in towards ~25 % acceptance
        if adapt and it < burn and (it + 1) % 50 == 0:
            rate  = acc_w / 50.0
            beta  = float(np.clip(beta * np.exp(0.5 * (rate - 0.25)), 1e-3, 1.0))
            acc_w = 0
        # --- Gibbs update of sigma^2 ~ InvGamma(a0 + n/2, b0 + n E / 2)
        sig2  = 1.0 / rng.gamma(a0 + n_eff/2.0, 1.0 / (b0 + n_eff*E/2.0))
        if it >= burn and (it - burn) % thin == 0:
            keep_w.append(gam)
            keep_s.append(sig2)
            if return_coef:
                keep_c.append(c.copy())
    out = dict(warps=np.array(keep_w), sigma2=np.array(keep_s), accept=acc/(n_samples*thin), beta=beta)
    if return_coef:
        out['coef'] = np.array(keep_c)
    return out


def initial_coef(gam, K):
    '''
    Tangent-space coefficients approximating a given warp (e.g. a dynamic
    programming solution), obtained by the inverse exponential map at the
    identity and projection onto the cosine basis. Useful as *c0* so that
    the chain starts near the mode rather than at the identity.
    '''
    Q   = gam.size
    t   = _warp.grid(Q)
    B   = _basis(t, K)
    psi = _warp.warp_to_psi(gam)
    one = np.ones(Q)
    v   = _warp._sphere_log(one, psi, t)
    return np.array([integrate.trapezoid(v*b, t)  for b in B])


def summarize(samples, alpha=0.05):
    '''
    Posterior summaries of warp samples (S,Q): Karcher-mean warp, pointwise
    posterior mean and (1-alpha) credible interval of the warp and of the
    displacement field.
    '''
    w    = np.asarray(samples, dtype=float)
    lo, hi = 100*alpha/2, 100*(1-alpha/2)
    d    = _warp.displacement_field(w)
    return dict(
        warp_mean   = _warp.karcher_mean_warp(w),
        warp_ci     = np.percentile(w, [lo, hi], axis=0),
        disp_mean   = d.mean(axis=0),
        disp_ci     = np.percentile(d, [lo, hi], axis=0),
        disp_samples= d,
    )


def align_group(y, template='srsf', n_samples=1000, burn=500, thin=1, K=8, tau=0.3, beta=0.05,
    init='dp', n_eff=None, smooth=0, center=True, anchor=None, random_state=None, verbose=False, **srsf_kwargs):
    '''
    Bayesian registration of a set of observations to a common template.

    *template* : 'srsf' (the Karcher-mean template of register_srsf, computed
                 with **srsf_kwargs), an integer index, or a (Q,) array. The
                 template is held fixed while the warps are sampled (no
                 hierarchical template update).
    *init*     : 'dp' (start each chain at the projection of the dynamic
                 programming warp) or 'identity'
    *center*   : 'karcher' (default, = True) | 'pointwise' | 'anchor' | 'none'.
                 The reference warp is computed from the posterior-MEAN warps
                 and applied to the means and to every posterior sample, so
                 the credible bands are reported on the same common axis.
    Other arguments as in sample_pair.

    Returns a dict with keys 'y' (registered with the posterior-mean warps),
    'warps' (J,Q) posterior-mean warps, 'template', 'samples' (J,S,Q),
    'sigma2' (J,S), 'accept' (J,), 'disp_ci' (J,2,Q).
    '''
    y     = np.asarray(y, dtype=float)
    J, Q  = y.shape[:2]
    rng   = np.random.default_rng(random_state)
    if isinstance(template, str) and template == 'srsf':
        r    = _srsf.align_group(y, smooth=smooth, **srsf_kwargs)
        tmpl = r['template']
        gdp  = r['warps']
    elif isinstance(template, (int, np.integer)):
        tmpl = y[int(template)].copy()
        gdp  = None
    else:
        tmpl = np.asarray(template, dtype=float)
        gdp  = None
    if gdp is None and init == 'dp':
        gdp = np.array([_srsf.align_pair(tmpl, yi, smooth=smooth)[1]  for yi in y])
    samples, sig2, acc = [], [], []
    for i in range(J):
        c0 = initial_coef(gdp[i], K) if init == 'dp' else None
        s  = sample_pair(tmpl, y[i], n_samples=n_samples, burn=burn, thin=thin, K=K, tau=tau,
            beta=beta, c0=c0, n_eff=n_eff, smooth=smooth, random_state=rng.integers(2**31))
        samples.append(s['warps']); sig2.append(s['sigma2']); acc.append(s['accept'])
        if verbose:
            print(f'observation {i+1}/{J}: acceptance rate = {s["accept"]:.2f}')
    samples = np.array(samples)
    wmean   = np.array([_warp.karcher_mean_warp(s)  for s in samples])
    center  = _warp.resolve_center(center, 'karcher')
    if center != 'none' and J > 1:
        wmean, gref = _warp.center_warps(wmean, method=center, anchor=anchor)
        gi          = _warp.invert(gref)
        samples     = np.array([[_warp.compose(w, gi)  for w in s]  for s in samples])
        tmpl        = _srsf._apply_warp_any(tmpl, gi)
    yr      = np.array([_srsf._apply_warp_any(y[i], wmean[i])  for i in range(J)])
    disp_ci = np.array([summarize(s)['disp_ci']  for s in samples])
    return dict(y=yr, warps=wmean, template=tmpl, samples=samples, sigma2=np.array(sig2),
        accept=np.array(acc), disp_ci=disp_ci, center=center)
