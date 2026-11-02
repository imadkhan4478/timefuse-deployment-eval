import sys, os, time, json
sys.path.insert(0, '/home/claude/ext/TimeFuse'); os.chdir('/home/claude/ext/TimeFuse')
import torch, numpy as np, random
from load_configs import get_forecast_exp_args
from exp.exp_fuse_forecasting import Exp_Fuse_Forecasting
torch.set_num_threads(2)
ds, model = sys.argv[1], sys.argv[2]
seed = 2021
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
args = get_forecast_exp_args(ds, model, 96, 48, 96)
args.root_path = '/home/claude/tfdata/'
args.use_gpu = False; args.use_multi_gpu = False; args.num_workers = 0
args.checkpoints = '/home/claude/tf_work/checkpoints/'
setting = f"96_48_96/{ds}_{model}_dmodel{args.d_model}_epoch{args.train_epochs}"
t = time.time()
exp = Exp_Fuse_Forecasting(args)
m, vl, tl = exp.train(setting=setting, verbose=False, tqdm_disable=True, save_model=True, raise_fwd_error=True)
out = dict(ds=ds, model=model, val_mse=float(vl), test_mse=float(tl), secs=time.time()-t,
           d_model=args.d_model, epochs=args.train_epochs, lr=args.learning_rate, bs=args.batch_size)
print(json.dumps(out))
with open('/home/claude/tf_work/train_log.jsonl', 'a') as f: f.write(json.dumps(out) + '\n')
