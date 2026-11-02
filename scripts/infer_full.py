# Run each trained base model on every forecast origin from validation start to the end of the data
import sys, os, numpy as np, pandas as pd, torch, time
sys.path.insert(0, '/home/claude/ext/TimeFuse'); os.chdir('/home/claude/ext/TimeFuse')
from load_configs import get_forecast_exp_args
from exp.exp_fuse_forecasting import Exp_Fuse_Forecasting
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(2)
SEQ, LAB, PRED = 96, 48, 96
TRAIN_END = 12*30*24
MODELS = ['DLinear', 'PatchTST', 'TimesNet', 'PAttn', 'TimeMixer', 'TimeXer']
ds = sys.argv[1]
df = pd.read_csv(f'/home/claude/tfdata/{ds}.csv')
X = df.iloc[:, 1:].values
Z = StandardScaler().fit(X[:TRAIN_END]).transform(X).astype(np.float32)
stamp = time_features(pd.to_datetime(df['date'].values), freq='h').transpose(1, 0).astype(np.float32)
origins = np.arange(TRAIN_END, len(Z) - PRED + 1)
trues = np.stack([Z[o:o+PRED] for o in origins])
allp = []
for m in MODELS:
    args = get_forecast_exp_args(ds, m, SEQ, LAB, PRED)
    args.root_path = '/home/claude/tfdata/'; args.use_gpu = False; args.use_multi_gpu = False; args.num_workers = 0
    exp = Exp_Fuse_Forecasting(args)
    ck = f"/home/claude/tf_work/checkpoints/96_48_96/{ds}_{m}_dmodel{args.d_model}_epoch{args.train_epochs}/checkpoint.pth"
    exp.model.load_state_dict(torch.load(ck, map_location='cpu')); exp.model.eval()
    preds = []; t = time.time()
    with torch.no_grad():
        for i in range(0, len(origins), 256):
            ob = origins[i:i+256]
            bx = torch.from_numpy(np.stack([Z[o-SEQ:o] for o in ob]))
            bxm = torch.from_numpy(np.stack([stamp[o-SEQ:o] for o in ob]))
            by = torch.from_numpy(np.stack([Z[o-LAB:o+PRED] for o in ob]))
            bym = torch.from_numpy(np.stack([stamp[o-LAB:o+PRED] for o in ob]))
            dec = torch.cat([by[:, :LAB], torch.zeros_like(by[:, -PRED:])], 1)
            out = exp.model(bx, bxm, dec, bym)[:, -PRED:, :]
            preds.append(out.numpy())
    preds = np.concatenate(preds)
    print(ds, m, preds.shape, 'mse_all', float(((preds-trues)**2).mean()), f'{time.time()-t:.0f}s', flush=True)
    allp.append(preds)
P = np.stack(allp, 1).astype(np.float32)
np.savez_compressed(f'/home/claude/tf_work/preds_{ds}.npz', preds=P, trues=trues, origins=origins, models=np.array(MODELS))
