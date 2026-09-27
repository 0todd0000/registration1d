'''
Figures for the JOSS paper.  Run from this folder:  python make_figures.py
'''
import os, sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
from matplotlib import pyplot as plt
import registration1d as reg1d

HERE    = os.path.dirname(os.path.abspath(__file__))
dataset = reg1d.data.Dorn2012()
speed   = dataset.group
yi      = reg1d.register_linear(dataset.y, n=101).y
colors  = ['k', 'b', 'g', 'r']
labels  = [f'speed {i}'  for i in range(4)]

results = [
    ('linear only',                 None),
    ('affine (cover)',              reg1d.register_affine(yi, cover=True, fill_value='zero')),
    ('landmark',                    reg1d.register_landmark(yi, kinds=('zero', 'max'))),
    ('derivative DTW, smoothed',    reg1d.register_dtw(yi, derivative=True, step_pattern='strict', smooth=0.03)),
    ('continuous (Ramsay-Li)',      reg1d.register_continuous(yi, n_basis=8, lam=1e-3)),
    ('self-modelling',              reg1d.register_sim(yi, n_basis=6)),
    ('pairwise synchronisation',    reg1d.register_pairwise(yi)),
    ('SRSF (elastic)',              reg1d.register_srsf(yi, max_iter=5)),
]

fig, AX = plt.subplots(2, 4, figsize=(14, 6.2))
for ax, (name, r) in zip(AX.ravel(), results):
    y = yi if r is None else r.y
    reg1d.plot.plot_curves(y, group=speed, ax=ax, colors=colors, x='percent', legend=(r is None), labels=labels, lw=1)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_title(name, size=11)
    ax.set_xlabel('time (% stance)')
for ax in AX[:, 0]:
    ax.set_ylabel('anteroposterior GRF (N)')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'figures', 'methods.png'), dpi=200)

r = reg1d.register_srsf(yi, max_iter=5)
fig, AX = plt.subplots(1, 3, figsize=(14, 3.8))
reg1d.plot.plot_curves(yi, group=speed, ax=AX[0], colors=colors, x='percent', labels=labels, lw=1); AX[0].set_title('linearly registered'); AX[0].set_xlabel('time (% stance)'); AX[0].set_ylabel('anteroposterior GRF (N)')
reg1d.plot.plot_curves(r.y, group=speed, ax=AX[1], colors=colors, x='percent', legend=False, lw=1); AX[1].set_title('SRSF-registered'); AX[1].set_xlabel('time (% stance)')
reg1d.plot.plot_displacement_fields(r.warps.asarray(), group=speed, ax=AX[2], colors=colors, legend=False, lw=1); AX[2].set_title('displacement fields'); AX[2].set_xlabel('time (normalised)')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'figures', 'srsf.png'), dpi=200)
print('wrote figures/methods.png and figures/srsf.png')
