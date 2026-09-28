'''
Generate, execute and render notebooks/power_simulated_datasets.ipynb.

    python make_power_notebook.py --simulate   # run the simulations (hours) and cache the results
    python make_power_notebook.py              # build the notebook from the cached results

Requires power1d (pip install power1d). The simulation code below (SIM_CODE)
is the code that appears in the notebook; --simulate executes the same code
outside the notebook so that the long run does not sit inside a kernel. The
notebook itself runs the simulations only when no cache file is present.
'''

import os, sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = 'power_simulated_datasets_results.pkl'

def md(s):   return nbf.v4.new_markdown_cell(s.strip('\n'))
def code(s): return nbf.v4.new_code_cell(s.strip('\n'))


INTRO = md('''
# Registration and statistical power on simulated data

The jiku-data survey (`jiku-data-datasets.ipynb`) showed that elastic registration introduces
geometric regularity into data that are essentially noise, SRSF registration in particular.
This notebook studies that behaviour under controlled conditions using the
[power1d](https://spm1d.org/power1d) framework: signal geometry from `power1d.geom`, smooth
Gaussian noise from `power1d.noise.SmoothGaussian` (FWHM = 25 % of the domain throughout), and
power1d's definition of numerical power.

**Design.** A one-sample experiment on J observations of Q = 101 nodes. The null model is pure
noise; the alternative model adds a `GaussianPulse` of amplitude `amp` (in units of the noise
standard deviation) and full width at half maximum `fwhm` (% of the domain), centred at 50 % of
the domain, with the centre of each observation's pulse displaced by an integer number of nodes
drawn from N(0, `jsd`) (`jsd` in % of the domain) — the "position variability" of the signal.
Each simulated dataset is registered by each method (or left unregistered), a one-sample
t continuum is computed with `power1d.stats.t_1sample`, and its maximum is recorded.

**Power, as defined by power1d.** The critical threshold `zstar` is the (1 − α) quantile of the
maximum t statistic under the null model, and power is the probability that the maximum t
statistic under the alternative model exceeds `zstar`. Because registration changes the null
distribution, `zstar` is computed *per method* from registered null data ("calibrated"
thresholds); in addition the false-positive rate of each method at the *unregistered* threshold
is reported, which measures how much regularity the method introduces into noise (the price of
registering without recalibrating the test).

**Other quantities recorded per simulated dataset**: the residual sum of squares about the mean
after registration relative to before (`SSE ratio`); the peak of the cross-sectional mean of
the registered data relative to the true amplitude (`peak / amp`, amplitude recovery); and the
squared correlation between the true pulse displacements and the displacements estimated by the
warps at the pulse centre, γᵢ(0.5) − 0.5 (`timing r²`, timing recovery; undefined when `jsd` = 0).

**Methods.** `none` (linearly registered data as generated), `SRSF lam=0` (unpenalised
Fisher–Rao alignment), `SRSF` (`lam='auto'`), `DDTW` (derivative DTW, strict step pattern,
smoothed warps), `continuous` (Ramsay–Li, 4 basis functions), `self-modelling` (shape-invariant
model, 4 basis functions). Iteration counts are reduced (SRSF 3, others 2) to keep the run time
of roughly 2 000 registrations per method manageable; all other settings are the defaults.

**Simulations.** Part A: pure noise, J ∈ {5, 10, 20, 50}, 100 datasets per J. Part B: signal
present, one factor varied at a time about the base condition (J = 10, amp = 1, fwhm = 20,
jsd = 5): amp ∈ {0.5, 1, 2, 4}, fwhm ∈ {10, 20, 40}, jsd ∈ {0, 2.5, 5, 10}, J ∈ {5, 10, 20, 50};
60 datasets per condition. The null simulations of Part A provide the thresholds for Part B.
''')


SIM_CODE = '''
import sys, os, time, pickle, warnings
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')      # one BLAS thread per worker process
sys.path.insert(0, os.path.abspath('../src'))   # so that this notebook finds registration1d without installation
import numpy as np
from matplotlib import pyplot as plt
import registration1d as reg1d
import power1d
warnings.filterwarnings('ignore')

Q, NOISE_FWHM, ALPHA = 101, 25, 0.05
CACHE   = 'power_simulated_datasets_results.pkl'
METHODS = ['none', 'SRSF lam=0', 'SRSF', 'DDTW', 'continuous', 'self-modelling']
COLORS  = dict(zip(METHODS, ['k', 'r', 'orange', 'b', 'g', 'm']))

def make_dataset(J, amp, fwhm, jsd, rng):
    """Noise (power1d SmoothGaussian) plus, if amp > 0, a GaussianPulse per observation whose
    centre is displaced by an integer number of nodes drawn from N(0, jsd). Returns (y, shifts)
    with shifts in units of the normalised domain."""
    noise = power1d.noise.SmoothGaussian(J=J, Q=Q, mu=0, sigma=1, fwhm=NOISE_FWHM)
    noise.random()
    y      = noise.value.copy()
    shifts = np.zeros(J)
    if amp > 0:
        dq = np.clip(np.round(rng.normal(0, jsd, J)), -30, 30).astype(int)
        for i in range(J):
            y[i] += power1d.geom.GaussianPulse(Q=Q, q=int(50 + dq[i]), fwhm=fwhm, amp=amp).value
        shifts = dq / (Q - 1)
    return y, shifts

def register(y, method):
    if method == 'none':
        return y, None
    if method == 'SRSF lam=0':
        r = reg1d.register_srsf(y, lam=0, max_iter=3)
    elif method == 'SRSF':
        r = reg1d.register_srsf(y, max_iter=3)
    elif method == 'DDTW':
        r = reg1d.register_dtw(y, derivative=True, step_pattern='strict', smooth=0.03, max_iter=2)
    elif method == 'continuous':
        r = reg1d.register_continuous(y, n_basis=4, max_iter=2)
    elif method == 'self-modelling':
        r = reg1d.register_sim(y, n_basis=4, max_iter=2)
    return r.y, r

def one_dataset(args):
    """Simulate one dataset and register it with every method; returns per-method metrics."""
    J, amp, fwhm, jsd, seed = args
    rng      = np.random.default_rng(seed)
    np.random.seed(seed)          # power1d draws from numpy's global generator
    y, s     = make_dataset(J, amp, fwhm, jsd, rng)
    sse0     = ((y - y.mean(axis=0))**2).sum()
    out      = {}
    for m in METHODS:
        try:
            yr, r = register(y, m)
        except Exception:
            yr, r = y, None
        t     = power1d.stats.t_1sample(yr)
        est   = np.zeros(J) if r is None else r.warps.asarray()[:, (Q - 1) // 2] - 0.5
        r2    = np.nan
        if s.std() > 0 and est.std() > 0:
            r2 = np.corrcoef(s, est)[0, 1]**2
        out[m] = dict(tmax=float(t.max()), t=t.astype(np.float32),
                      sse=float(((yr - yr.mean(axis=0))**2).sum() / sse0),
                      peak=float(yr.mean(axis=0).max()), r2=r2, y=yr.astype(np.float32))
    return out

def simulate(J, amp, fwhm, jsd, n, seed0, workers=2):
    """n datasets of one condition; returns dict method -> dict of stacked metrics."""
    from multiprocessing import Pool
    args = [(J, amp, fwhm, jsd, seed0 + i)  for i in range(n)]
    t0   = time.time()
    with Pool(workers) as pool:
        res = pool.map(one_dataset, args, chunksize=1)
    out = {}
    for m in METHODS:
        out[m] = dict(tmax=np.array([r[m]['tmax'] for r in res]), t=np.array([r[m]['t'] for r in res]),
                      sse=np.array([r[m]['sse'] for r in res]), peak=np.array([r[m]['peak'] for r in res]),
                      r2=np.array([r[m]['r2'] for r in res]), y_example=res[0][m]['y'])
    print(f'J={J:3d} amp={amp:<4g} fwhm={fwhm:<3g} jsd={jsd:<4g}  n={n}  ({time.time()-t0:6.0f} s)', flush=True)
    return out

BASE   = dict(J=10, amp=1.0, fwhm=20, jsd=5.0)
NULLS  = [dict(J=J, amp=0.0, fwhm=20, jsd=0.0)  for J in (5, 10, 20, 50)]
ALTS   = []
for amp in (0.5, 1.0, 2.0, 4.0):    ALTS.append(dict(BASE, amp=amp))
for fwhm in (10, 40):               ALTS.append(dict(BASE, fwhm=fwhm))
for jsd in (0.0, 2.5, 10.0):        ALTS.append(dict(BASE, jsd=jsd))
for J in (5, 20, 50):               ALTS.append(dict(BASE, J=J))

def key(c):  return (c['J'], c['amp'], c['fwhm'], c['jsd'])

def run_all(n_null=100, n_alt=60):
    results = {}
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            results = pickle.load(f)
    todo = [(c, n_null) for c in NULLS] + [(c, n_alt) for c in ALTS]
    for k, (c, n) in enumerate(todo):
        if key(c) in results:
            continue
        results[key(c)] = simulate(n=n, seed0=1000 * k, **c)
        with open(CACHE, 'wb') as f:
            pickle.dump(results, f)
    return results

if __name__ == '__main__' and '--simulate' in sys.argv:
    run_all()
'''


SETUP_RUN = code('''
results = run_all()          # loads the cached results; runs the simulations (hours) if the cache is absent
print(len(results), 'conditions')
''')


def build():
    cells = [INTRO, code(SIM_CODE), SETUP_RUN]

    cells.append(md('''
## Part A: pure noise

### What registration does to noise

One simulated null dataset (J = 10) before and after each method. The unregistered curves are
smooth Gaussian random fields with no common structure; the registered panels show what each
method makes of them.
'''))
    cells.append(code('''
res = results[(10, 0.0, 20, 0.0)]
fig, AX = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True, sharey=True)
t = np.linspace(0, 100, Q)
for ax, m in zip(AX.ravel(), METHODS):
    y = res[m]['y_example']
    ax.plot(t, y.T, color=COLORS[m], lw=0.8, alpha=0.7)
    ax.plot(t, y.mean(axis=0), color='k', lw=2.5)
    ax.set_title(f"{m}   (SSE ratio {res[m]['sse'].mean():.2f}, max t {res[m]['tmax'][0]:.1f})", size=10)
for ax in AX[1]: ax.set_xlabel('domain (%)')
plt.suptitle('One null dataset (J = 10, noise FWHM = 25): thin lines observations, thick line cross-sectional mean', size=11)
plt.tight_layout(); plt.show()
'''))
    cells.append(md('''
### Regularity introduced into noise, as a function of sample size

Left: residual reduction (a descriptive measure of how much the warps "explain"). Middle: the
false-positive rate of the one-sample test at the *unregistered* critical threshold — the
probability of declaring a significant mean signal in pure noise when the data are registered
but the test is not recalibrated (nominal rate α = 0.05, dotted line). Right: the calibrated
critical threshold `zstar` for each method, i.e. the threshold the test would need in order to
keep α = 0.05 after registration.
'''))
    cells.append(code('''
Js = [5, 10, 20, 50]
fig, AX = plt.subplots(1, 3, figsize=(15, 4.2))
zstar = {}
for m in METHODS:
    sse, fpr, zs = [], [], []
    for J in Js:
        res  = results[(J, 0.0, 20, 0.0)]
        z0   = np.percentile(results[(J, 0.0, 20, 0.0)]['none']['tmax'], 100 * (1 - ALPHA))
        zs.append(np.percentile(res[m]['tmax'], 100 * (1 - ALPHA)))
        fpr.append((res[m]['tmax'] > z0).mean())
        sse.append(res[m]['sse'].mean())
        zstar[(J, m)] = zs[-1]
    AX[0].plot(Js, sse, 'o-', color=COLORS[m], label=m)
    AX[1].plot(Js, fpr, 'o-', color=COLORS[m], label=m)
    AX[2].plot(Js, zs,  'o-', color=COLORS[m], label=m)
AX[1].axhline(ALPHA, color='k', ls=':')
AX[0].set_ylabel('SSE ratio (after / before)');            AX[0].set_ylim(0, 1.3)
AX[1].set_ylabel('false-positive rate at the unregistered threshold'); AX[1].set_ylim(0, 1.02)
AX[2].set_ylabel('calibrated critical threshold  zstar')
for ax in AX:
    ax.set_xscale('log'); ax.set_xticks(Js); ax.set_xticklabels(Js); ax.minorticks_off(); ax.set_xlabel('sample size J')
AX[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"{'J':>3s} " + ' '.join(f'{m:>15s}' for m in METHODS))
print('false-positive rate at the unregistered threshold:')
for J in Js:
    z0 = np.percentile(results[(J, 0.0, 20, 0.0)]['none']['tmax'], 100 * (1 - ALPHA))
    print(f'{J:3d} ' + ' '.join(f"{(results[(J, 0.0, 20, 0.0)][m]['tmax'] > z0).mean():15.2f}" for m in METHODS))
print('calibrated threshold zstar:')
for J in Js:
    print(f'{J:3d} ' + ' '.join(f'{zstar[(J, m)]:15.2f}' for m in METHODS))
'''))
    cells.append(md('''
### The distribution of the maximum t statistic under the null

The whole null distribution, not only its 95th percentile: the unregistered maximum t statistic
(black) against the registered ones, for J = 10 and J = 50.
'''))
    cells.append(code('''
fig, AX = plt.subplots(1, 2, figsize=(13, 4))
for ax, J in zip(AX, (10, 50)):
    res  = results[(J, 0.0, 20, 0.0)]
    bins = np.linspace(0, max(res[m]['tmax'].max() for m in METHODS) * 1.05, 40)
    for m in METHODS:
        ax.hist(res[m]['tmax'], bins=bins, histtype='step', color=COLORS[m], lw=1.5, label=m)
    ax.axvline(np.percentile(res['none']['tmax'], 95), color='k', ls=':')
    ax.set_title(f'J = {J}: maximum one-sample t statistic in pure noise', size=10)
    ax.set_xlabel('max t'); ax.set_ylabel('count')
AX[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''))
    cells.append(md('''
### Mean t continua of registered noise

Registration that aligns noise extrema across observations concentrates the spurious effect
where the aligned extrema land. The average t continuum over the null datasets shows whether
the introduced regularity has a preferred location (it should be flat, near zero, for
unregistered noise).
'''))
    cells.append(code('''
fig, AX = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, J in zip(AX, (10, 50)):
    res = results[(J, 0.0, 20, 0.0)]
    for m in METHODS:
        ax.plot(t, res[m]['t'].mean(axis=0), color=COLORS[m], label=m)
    ax.axhline(0, color='k', ls=':'); ax.set_title(f'J = {J}: mean t continuum over null datasets', size=10)
    ax.set_xlabel('domain (%)')
AX[0].set_ylabel('mean t'); AX[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''))

    cells.append(md('''
## Part B: signal plus noise

### The base condition

One dataset of the base condition (J = 10, amp = 1, fwhm = 20, jsd = 5): a unit-amplitude pulse
of the same width as the noise's correlation length, with its centre jittered by about five
nodes. The true signal (mean pulse shape at the nominal centre) is shown dashed.
'''))
    cells.append(code('''
res = results[key(BASE)]
sig = power1d.geom.GaussianPulse(Q=Q, q=50, fwhm=BASE['fwhm'], amp=BASE['amp']).value
fig, AX = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, m in zip(AX.ravel(), METHODS):
    y = res[m]['y_example']
    ax.plot(t, y.T, color=COLORS[m], lw=0.8, alpha=0.7)
    ax.plot(t, y.mean(axis=0), color='k', lw=2.5)
    ax.plot(t, sig, 'k--', lw=1.5)
    ax.set_title(f"{m}   (peak/amp {res[m]['peak'].mean():.2f}, timing r² {np.nanmean(res[m]['r2']):.2f})", size=10)
for ax in AX[1]: ax.set_xlabel('domain (%)')
plt.suptitle('Base condition: observations, cross-sectional mean (thick) and true pulse (dashed)', size=11)
plt.tight_layout(); plt.show()
'''))
    cells.append(md('''
### Power, amplitude recovery and timing recovery

Each row varies one factor about the base condition. Left: calibrated power (the threshold of
each method comes from its own registered null distribution at the same J). Middle: the peak
of the cross-sectional mean relative to the true amplitude (1 = the amplitude is recovered;
below 1 = smeared by position variability or noise; above 1 = inflated by aligned noise).
Right: squared correlation between true and estimated pulse displacements (how much of the
imposed timing variability the warps recover; blank where jsd = 0).
'''))
    cells.append(code('''
def power(c, m):
    z = zstar[(c['J'], m)]
    return (results[key(c)][m]['tmax'] > z).mean()

sweeps = [('amp',  [0.5, 1.0, 2.0, 4.0], 'signal amplitude (noise SD units)'),
          ('fwhm', [10, 20, 40],          'signal breadth  fwhm (% of domain)'),
          ('jsd',  [0.0, 2.5, 5.0, 10.0], 'position variability  jsd (% of domain)'),
          ('J',    [5, 10, 20, 50],       'sample size J')]
fig, AX = plt.subplots(4, 3, figsize=(15, 15))
for row, (name, values, label) in zip(AX, sweeps):
    for m in METHODS:
        conds = [dict(BASE, **{name: v}) for v in values]
        row[0].plot(values, [power(c, m) for c in conds], 'o-', color=COLORS[m], label=m)
        row[1].plot(values, [results[key(c)][m]['peak'].mean() / c['amp'] for c in conds], 'o-', color=COLORS[m])
        row[2].plot(values, [np.nanmean(results[key(c)][m]['r2']) for c in conds], 'o-', color=COLORS[m])
    row[0].set_ylabel('power (calibrated)'); row[0].set_ylim(0, 1.02)
    row[1].set_ylabel('peak of mean / amp');  row[1].axhline(1, color='k', ls=':')
    row[2].set_ylabel('timing r²');           row[2].set_ylim(0, 1.02)
    for ax in row:
        ax.set_xlabel(label)
        if name in ('J', 'amp'):
            ax.set_xscale('log'); ax.set_xticks(values); ax.set_xticklabels(values); ax.minorticks_off()
AX[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.show()

print('calibrated power')
print(f"{'condition':>28s} " + ' '.join(f'{m:>15s}' for m in METHODS))
for name, values, _ in sweeps:
    for v in values:
        c = dict(BASE, **{name: v})
        print(f"{name+'='+str(v):>28s} " + ' '.join(f'{power(c, m):15.2f}' for m in METHODS))
'''))
    cells.append(md('''
### Power at the unregistered threshold

The same power curves when the test is *not* recalibrated after registration (the threshold of
the unregistered data is used for every method). This is what a naive pipeline —
register, then test as usual — would report, and it must be read together with the false-positive
rates of Part A: a method whose false-positive rate is 0.5 will show high "power" here for any
signal, including none.
'''))
    cells.append(code('''
fig, AX = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, (name, values, label) in zip(AX, sweeps):
    for m in METHODS:
        conds = [dict(BASE, **{name: v}) for v in values]
        p = [(results[key(c)][m]['tmax'] > zstar[(c['J'], 'none')]).mean() for c in conds]
        ax.plot(values, p, 'o-', color=COLORS[m], label=m)
    ax.set_xlabel(label); ax.set_ylim(0, 1.02)
    if name in ('J', 'amp'):
        ax.set_xscale('log'); ax.set_xticks(values); ax.set_xticklabels(values); ax.minorticks_off()
AX[0].set_ylabel('power at the unregistered threshold'); AX[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''))
    cells.append(md('''
### Mean t continua with the signal present

Where the evidence for the signal sits after registration, for the amplitude sweep (J = 10,
fwhm = 20, jsd = 5). The dotted horizontal line is the unregistered threshold; the coloured
dotted lines are each method's calibrated threshold.
'''))
    cells.append(code('''
fig, AX = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for ax, amp in zip(AX, (0.5, 1.0, 2.0, 4.0)):
    c = dict(BASE, amp=amp)
    for m in METHODS:
        ax.plot(t, results[key(c)][m]['t'].mean(axis=0), color=COLORS[m], label=m)
        ax.axhline(zstar[(c['J'], m)], color=COLORS[m], ls=':', lw=0.8)
    ax.set_title(f'amp = {amp}', size=10); ax.set_xlabel('domain (%)')
AX[0].set_ylabel('mean t continuum'); AX[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
'''))

    cells.append(md('''
## Summary

**False-positive rate (Part A).** Registration introduces geometric regularity into pure noise,
and the effect grows with sample size rather than shrinking. At the unregistered α = 0.05
threshold, the one-sample test on SRSF-registered noise rejects the null in 15-17 % of datasets
at J = 5, about 75 % at J = 10, 85-90 % at J = 20 and 100 % at J = 50 (`lam='auto'` makes
almost no difference: the median-total-variation penalty is small relative to the noise's own
variation). Derivative DTW behaves the same way with slightly smaller rates (12 %, 55 %, 70 %,
99 %). Continuous registration is different in kind: its low-dimensional warps pinch all curves
through common nodes, so the maximum t statistic explodes (calibrated threshold 34 at J = 5) but
the rate levels off near 50 % because the pinch points are where the mean is close to zero. The
self-modelling method leaves the false-positive rate at its nominal value at every J because its
amplitude parameters absorb what its warps would otherwise align. The mean t continua show that
the spurious evidence produced by SRSF and DTW has no preferred location: registration
concentrates whatever extrema the noise has, wherever they are. The consequence for practice is
that a test applied after registration must be recalibrated (the critical threshold for SRSF is
2-3.5 times the unregistered one), for instance by a permutation scheme that re-registers each
permuted dataset, or the registration must be shown to be justified before the test.

**Power (Part B).** With thresholds calibrated per method, no registration method exceeded the
power of the unregistered test in any condition studied: at the base condition (J = 10, unit
amplitude, signal as wide as the noise correlation length, five-node position jitter) the
unregistered power is 0.62 against 0.20-0.27 for SRSF and DTW and 0.50 for continuous
registration, and the ordering is the same for narrower and wider pulses, for no jitter, and for
twice the jitter. The methods converge as amplitude grows (at four noise SDs every method has
power 1, recovers the amplitude within 1 % and recovers 90 % of the imposed timing variance) and
as J grows (J = 50: 0.83-0.97 against 1.00), i.e. registration approaches, but does not beat, the
unregistered test in this design. Convergence with decreasing position variability does not
occur: with no jitter at all the registered methods still lose power (0.17 against 0.63),
because calibration for the regularity they introduce costs more than the alignment gains.
Read without calibration, the picture inverts — registered "power" is 0.9-1.0 at unit amplitude —
which is exactly the inflation seen in Part A and not evidence for the signal.

**Signal recovery.** Amplitude recovery separates the methods more clearly than power does.
SRSF and DTW inflate the peak of the mean by 30-40 % at unit amplitude and by more than a factor
of two at half the noise SD, by aligning noise extrema onto the pulse; the inflation vanishes at
four noise SDs. The unregistered mean is attenuated by position jitter (to 0.80 at ten nodes of
jitter) but never inflated; continuous registration is roughly unbiased at unit amplitude and
self-modelling under-estimates the amplitude (its scale parameters shrink observations towards
the template). Timing recovery (r² between the true and estimated pulse displacements) is low
at unit amplitude (0.1-0.2), best for derivative DTW when the pulse is narrow (0.65 at
fwhm = 10), and rises to about 0.9 for SRSF and DTW at four noise SDs: the warps recover the
timing structure only once the signal dominates the noise, and at realistic amplitudes most of
the estimated displacement is noise alignment.

**Implications for registration1d.** (1) A registrability check is not optional: the
false-positive rate at J ≥ 10 makes an uncalibrated test after SRSF or DTW registration
uninterpretable. The null distributions computed here are the reference for such a check on
smooth noise; a data-driven version would compare the observed reduction with that obtained on
surrogate noise of the same smoothness. (2) The default `lam='auto'` does not protect against
noise alignment; a penalty strong enough to do so would have to scale with the noise, not with
the signal's total variation, which argues for a penalty chosen by a null-based criterion
(e.g. the smallest `lam` at which the false-positive rate on surrogate data returns to nominal).
(3) `reg1d.stats` should offer a permutation test in which registration is inside the
permutation loop, since only that keeps α when the analysis includes registration. (4) These
simulations cover the one-sample, single-pulse case at one noise smoothness; the two-sample case
with a *group difference in timing*, where registration converts a timing effect into a warp
effect, is the case in which registration can gain power and is the natural next simulation.
'''))
    return cells


def write_notebook(execute=True):
    cells = build()
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata']['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    if execute:
        ep = ExecutePreprocessor(timeout=36000, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': HERE}})
    fpath = os.path.join(HERE, 'power_simulated_datasets.ipynb')
    nbf.write(nb, fpath)
    html, _ = HTMLExporter().from_notebook_node(nb)
    with open(os.path.join(HERE, 'html', 'power_simulated_datasets.html'), 'w') as f:
        f.write(html)
    print('wrote', fpath)


if __name__ == '__main__':
    if '--simulate' in sys.argv:
        os.chdir(HERE)
        exec(compile(SIM_CODE, 'SIM_CODE', 'exec'))
    else:
        write_notebook(execute='--no-execute' not in sys.argv)
