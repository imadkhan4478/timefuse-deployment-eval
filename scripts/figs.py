import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('results/results.json'))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
TXT, MUTED, GRID, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#f3f2ee'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': TXT,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True})
ms = pd.DataFrame(R['monthly_static']); rb = pd.DataFrame(R['protocol_B'])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, ds in zip(axes, ['ETTh1', 'ETTh2']):
    a = ms[ms.dataset == ds].sort_values('month'); b = rb[rb.dataset == ds].sort_values('month'); x = a.months_after_val.values
    g = lambda s: (a.val_best.values - s) / a.val_best.values * 100
    ax.axvspan(4.5, 8.5, color=BAND, zorder=0); ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(x, g(a.timefuse_static.values), '-o', color=C[0], lw=2, ms=4.5, label='TimeFuse, fusor fit once')
    ax.plot(x, g(b.tf_rolling.values), '-s', color=C[1], lw=2, ms=4.5, label='TimeFuse, fusor refit monthly')
    ax.plot(x, g(a.mean_ens.values), '--^', color=C[2], lw=1.6, ms=4.5, label='Equal-weight average')
    ax.set_title(ds, loc='left', fontsize=9, color=TXT); ax.set_xticks(range(1, 9)); ax.set_xlim(0.5, 8.5)
    ax.set_xlabel('Months after the validation window')
axes[0].set_ylabel('MSE reduction vs model\nselected on validation (%)')
for ax in axes:
    lo, hi = ax.get_ylim(); ax.set_ylim(min(lo, -1.5), hi * 1.12); lo, hi = ax.get_ylim()
    ax.text(2.5, hi - (hi - lo) * 0.05, 'benchmark test months', ha='center', va='top', color=MUTED, fontsize=7.3)
    ax.text(6.5, hi - (hi - lo) * 0.05, 'never-used months', ha='center', va='top', color=MUTED, fontsize=7.3)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('figures/fig1.png', dpi=250, bbox_inches='tight'); plt.close(fig)

st = pd.DataFrame(R['protocol_C'])
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for ds, col, mk in [('ETTh1', C[0], 'o'), ('ETTh2', C[1], 's')]:
    s = st[st.dataset == ds]; base = s[s.lag_months == 0].set_index('month').timefuse
    rel = s.groupby('lag_months').apply(lambda d: (d.set_index('month').timefuse / base - 1).mean() * 100)
    ax.plot(rel.index, rel.values, '-' + mk, color=col, lw=2, ms=4.5, label=ds)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(0, 5)); ax.set_ylim(-1, 3)
ax.set_xlabel("Age of the fusor's 4-month training window (months)"); ax.set_ylabel('MSE vs freshest fusor (%)')
ax.legend(frameon=False, loc='upper left'); fig.tight_layout(); fig.savefig('figures/fig2.png', dpi=250, bbox_inches='tight'); plt.close(fig)

mods = ['DLinear', 'PatchTST', 'TimesNet', 'PAttn', 'TimeMixer', 'TimeXer']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, ds in zip(axes, ['ETTh1', 'ETTh2']):
    a = ms[ms.dataset == ds].sort_values('month'); x = a.months_after_val.values
    W = np.array([a[f'w_{m}'].values for m in mods])
    ax.bar(x[:, None].repeat(6, 1).T.flatten() * 0 + 0, 0)  # no-op keeps axes consistent
    bottom = np.zeros(len(x))
    for i, m in enumerate(mods):
        ax.bar(x, W[i], bottom=bottom, color=C[i], width=0.72, edgecolor='white', linewidth=1.0, label=m); bottom += W[i]
    ax.axhline(1/6, color=TXT, lw=0.8, ls=':')
    for xi, bm in zip(x, a.oracle_month_best_model): ax.text(xi, 1.03, bm, rotation=50, fontsize=6.4, color=MUTED, ha='left', va='bottom')
    ax.set_ylim(0, 1); ax.set_xticks(range(1, 9)); ax.set_xlim(0.4, 8.6); ax.grid(False)
    ax.set_title(ds, loc='left', fontsize=9, color=TXT, pad=36); ax.set_xlabel('Months after the validation window')
axes[0].set_ylabel('Mean fusor weight\n(fusor fit on validation)')
axes[0].text(8.65, 1/6, '1/6', fontsize=6.5, color=TXT, va='center')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('figures/fig3.png', dpi=250, bbox_inches='tight'); plt.close(fig)
print('ok')
