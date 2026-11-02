# Protocols A, A+, B, C + statistics for the TimeFuse deployment-faithful study
import numpy as np, pandas as pd, torch, torch.nn as nn, json, math
torch.set_num_threads(2)
DS = ['ETTh1', 'ETTh2']; MONTH = 720; PRED = 96
SEEDS = [2021, 2022, 2023, 2024, 2025]
D = {}
for ds in DS:
    z = np.load(f'/home/claude/tf_work/preds_{ds}.npz', allow_pickle=True)
    M = pd.read_parquet(f'/home/claude/tf_work/meta_{ds}.parquet').set_index('origin')
    failed = M.pop('__failed__').notna().sum() if '__failed__' in M else 0
    M = M.loc[z['origins']].fillna(0.0)
    D[ds] = dict(P=z['preds'], Y=z['trues'], o=z['origins'], X=M.values.astype(np.float32),
                 models=list(z['models']), failed=int(failed), feat_names=list(M.columns))
MODELS = D['ETTh1']['models']; K = len(MODELS)
def month(o): return o // MONTH + 1
VAL = lambda o: (o >= 8640) & (o + PRED <= 11520)
TEST = lambda o: (o >= 11520) & (o + PRED <= 14400)
TAIL = lambda o: (o >= 14400) & (o < 17280)   # months 21-24 (the data TSLib never uses)

def fit_fusor(sets, seed, epochs=5, bs=64, lr=5e-4):
    """sets: list of index masks per dataset (same order as DS). Mirrors run_timefuse_exp.ipynb."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xs = [D[ds]['X'][m] for ds, m in zip(DS, sets)]
    allx = np.concatenate(Xs); mu, sd = allx.mean(0), allx.std(0); sd[sd == 0] = 1
    fc = nn.Linear(Xs[0].shape[1], K); nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
    opt = torch.optim.Adam(fc.parameters(), lr=lr); crit = nn.SmoothL1Loss(beta=0.01)
    data = [(torch.tensor((D[ds]['X'][m] - mu) / sd), torch.tensor(D[ds]['P'][m]), torch.tensor(D[ds]['Y'][m]))
            for ds, m in zip(DS, sets)]
    L = max(len(d[0]) for d in data)
    for ep in range(epochs):
        orders = []
        for d in data:  # length alignment by oversampling, as in get_length_aligned_loaders
            n = len(d[0]); idx = np.concatenate([np.tile(np.arange(n), L // n), rng.choice(n, L % n, replace=False)])
            orders.append(rng.permutation(idx))
        for b in range(0, L, bs):
            for d, od in zip(data, orders):
                ii = od[b:b+bs]
                w = torch.softmax(fc(d[0][ii]), -1)[:, :, None, None]
                loss = crit((w * d[1][ii]).sum(1), d[2][ii])
                opt.zero_grad(); loss.backward(); opt.step()
    return fc, mu, sd

def apply(fc, mu, sd, ds, m):
    with torch.no_grad():
        w = torch.softmax(fc(torch.tensor((D[ds]['X'][m] - mu) / sd)), -1).numpy()
    return (w[:, :, None, None] * D[ds]['P'][m]).sum(1), w

def per_origin_mse(pred, y): return ((pred - y) ** 2).mean((1, 2))
def per_origin_mae(pred, y): return np.abs(pred - y).mean((1, 2))

# ---------- statistics ----------
def block_boot(d, block=168, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(d); nb = int(np.ceil(n / block))
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        means[b] = np.concatenate([d[s:s+block] for s in starts])[:n].mean()
    return means
def dm_test(d, h=PRED):
    from scipy import stats
    n = len(d); dbar = d.mean(); dc = d - dbar
    lag = h - 1; gam = [np.dot(dc[k:], dc[:n-k]) / n for k in range(lag + 1)]
    lrv = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))
    stat = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)
def compare(loss_tf, loss_base, name):
    d = loss_tf - loss_base             # negative = TimeFuse better
    bm = block_boot(d)
    s, p = dm_test(d)
    return dict(comparison=name, mean_diff=float(d.mean()), rel_gain_pct=float(-d.mean() / loss_base.mean() * 100),
                ci_lo=float(np.percentile(bm, 2.5)), ci_hi=float(np.percentile(bm, 97.5)),
                p_not_better=float((bm >= 0).mean()), dm_stat=s, dm_p=p, n=len(d))

R = {'failed_meta_windows': {ds: D[ds]['failed'] for ds in DS}, 'n_meta_features': len(D['ETTh1']['feat_names'])}

# ---------- Protocol A and A+ : fusor fit on validation (paper protocol) ----------
static = [fit_fusor([VAL(D[ds]['o']) for ds in DS], s) for s in SEEDS]
rows, stats_rows, monthly, weights_month = [], [], [], []
for ds in DS:
    o, P, Y = D[ds]['o'], D[ds]['P'], D[ds]['Y']
    v = VAL(o); val_mse = [per_origin_mse(P[v][:, k], Y[v]).mean() for k in range(K)]
    k_val = int(np.argmin(val_mse))
    for split, msk in [('test', TEST(o)), ('tail', TAIL(o))]:
        y = Y[msk]
        base_mse = {MODELS[k]: per_origin_mse(P[msk][:, k], y) for k in range(K)}
        base_mae = {MODELS[k]: per_origin_mae(P[msk][:, k], y) for k in range(K)}
        k_orc = min(base_mse, key=lambda k: base_mse[k].mean())
        mean_pred = P[msk].mean(1)
        tf = [apply(f, mu, sd, ds, msk) for f, mu, sd in static]
        tf_mse = np.mean([per_origin_mse(p, y) for p, _ in tf], 0)
        tf_mae = np.mean([per_origin_mae(p, y) for p, _ in tf], 0)
        seed_mse = [per_origin_mse(p, y).mean() for p, _ in tf]
        row = dict(dataset=ds, split=split, timefuse_mse=float(tf_mse.mean()), timefuse_mse_sd=float(np.std(seed_mse)),
                   timefuse_mae=float(tf_mae.mean()), mean_ens_mse=float(per_origin_mse(mean_pred, y).mean()),
                   val_best=MODELS[k_val], val_best_mse=float(base_mse[MODELS[k_val]].mean()),
                   val_best_mae=float(base_mae[MODELS[k_val]].mean()),
                   oracle_best=k_orc, oracle_best_mse=float(base_mse[k_orc].mean()))
        row.update({f'{m}_mse': float(base_mse[m].mean()) for m in MODELS})
        row.update({f'{m}_mae': float(base_mae[m].mean()) for m in MODELS})
        rows.append(row)
        for nm, base in [('vs val-selected best single', base_mse[MODELS[k_val]]), ('vs oracle best single', base_mse[k_orc]),
                         ('vs simple mean', per_origin_mse(mean_pred, y))]:
            stats_rows.append(dict(dataset=ds, split=split, **compare(tf_mse, base, nm)))
    # monthly breakdown 17..24
    for mth in range(17, 25):
        msk = month(o) == mth
        y = Y[msk]
        base = np.array([per_origin_mse(P[msk][:, k], y).mean() for k in range(K)])
        tfp = [apply(f, mu, sd, ds, msk) for f, mu, sd in static]
        tfm = np.mean([per_origin_mse(p, y).mean() for p, _ in tfp])
        w = np.mean([ww.mean(0) for _, ww in tfp], 0)
        monthly.append(dict(dataset=ds, month=mth, months_after_val=mth - 16, timefuse_static=float(tfm),
                            val_best=float(base[k_val]), oracle_month_best=float(base.min()),
                            oracle_month_best_model=MODELS[int(base.argmin())], mean_ens=float(per_origin_mse(P[msk].mean(1), y).mean()),
                            **{f'w_{m}': float(w[i]) for i, m in enumerate(MODELS)},
                            **{f'mse_{m}': float(base[i]) for i, m in enumerate(MODELS)}))
R['protocol_A'] = rows; R['stats_A'] = stats_rows; R['monthly_static'] = monthly

# ---------- Protocol B: rolling monthly refit (most recent 4 months of labelled windows) ----------
WIN = 4 * MONTH
roll = []
B_loss = {ds: {'tf_roll': [], 'tf_static': [], 'val_best': [], 'roll_best': []} for ds in DS}
for mth in range(17, 25):
    T = (mth - 1) * MONTH
    trmask = lambda o: (o + PRED <= T) & (o + PRED > T - WIN)
    fits = [fit_fusor([trmask(D[ds]['o']) for ds in DS], s) for s in SEEDS]
    for ds in DS:
        o, P, Y = D[ds]['o'], D[ds]['P'], D[ds]['Y']
        tm, em = trmask(o), month(o) == mth
        y = Y[em]
        k_roll = int(np.argmin([per_origin_mse(P[tm][:, k], Y[tm]).mean() for k in range(K)]))
        v = VAL(o); k_val = int(np.argmin([per_origin_mse(P[v][:, k], Y[v]).mean() for k in range(K)]))
        tr = np.mean([per_origin_mse(apply(f, mu, sd, ds, em)[0], y) for f, mu, sd in fits], 0)
        ts = np.mean([per_origin_mse(apply(f, mu, sd, ds, em)[0], y) for f, mu, sd in static], 0)
        vb = per_origin_mse(P[em][:, k_val], y); rb = per_origin_mse(P[em][:, k_roll], y)
        for kk, vv in [('tf_roll', tr), ('tf_static', ts), ('val_best', vb), ('roll_best', rb)]: B_loss[ds][kk].append(vv)
        wr = np.mean([apply(f, mu, sd, ds, em)[1].mean(0) for f, mu, sd in fits], 0)
        roll.append(dict(dataset=ds, month=mth, tf_rolling=float(tr.mean()), tf_static=float(ts.mean()),
                         val_best=float(vb.mean()), rolling_best_single=float(rb.mean()), rolling_best_model=MODELS[k_roll],
                         **{f'wroll_{m}': float(wr[i]) for i, m in enumerate(MODELS)}))
R['protocol_B'] = roll
statsB = []
for ds in DS:
    for split, sl in [('test', slice(0, 4)), ('tail', slice(4, 8))]:
        L = {k: np.concatenate(v[sl]) for k, v in B_loss[ds].items()}
        statsB.append(dict(dataset=ds, split=split, **compare(L['tf_roll'], L['tf_static'], 'rolling TF vs static TF')))
        statsB.append(dict(dataset=ds, split=split, **compare(L['tf_roll'], L['roll_best'], 'rolling TF vs rolling best single')))
        statsB.append(dict(dataset=ds, split=split, **compare(L['tf_static'], L['val_best'], 'static TF vs val-selected best single')))
R['stats_B'] = statsB

# ---------- Protocol C: staleness curve (fusor window ends k months before eval month; months 21-24) ----------
stale = []
for mth in range(21, 25):
    T = (mth - 1) * MONTH
    for k in range(0, 5):
        end = T - k * MONTH
        trm = lambda o: (o + PRED <= end) & (o + PRED > end - WIN) & (o >= 8640)
        fits = [fit_fusor([trm(D[ds]['o']) for ds in DS], s) for s in SEEDS]
        for ds in DS:
            o, P, Y = D[ds]['o'], D[ds]['P'], D[ds]['Y']; em = month(o) == mth; y = Y[em]
            tfm = np.mean([per_origin_mse(apply(f, mu, sd, ds, em)[0], y).mean() for f, mu, sd in fits])
            tmk = trm(o); kb = int(np.argmin([per_origin_mse(P[tmk][:, j], Y[tmk]).mean() for j in range(K)]))
            stale.append(dict(dataset=ds, month=mth, lag_months=k, timefuse=float(tfm),
                              sel_best_single=float(per_origin_mse(P[em][:, kb], y).mean()), n_fit=int(tmk.sum())))
R['protocol_C'] = stale
json.dump(R, open('/home/claude/tf_work/results.json', 'w'), indent=1)
print('done')
