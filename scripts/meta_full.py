# Meta-features for every forecast origin from validation start to end of data (TimeFuse's own extractor)
import os, sys, numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/ext/TimeFuse')
from meta_feature import extract_meta_feature
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS']='ignore'
SEQ, PRED = 96, 96
TRAIN_END = 12*30*24
ds = sys.argv[1]
df = pd.read_csv(f'/home/claude/tfdata/{ds}.csv')
X = df.iloc[:, 1:].values
sc = StandardScaler().fit(X[:TRAIN_END]); Z = sc.transform(X)
b1 = TRAIN_END - SEQ
origins = np.arange(TRAIN_END, len(Z) - PRED + 1)  # first target row
def f(o):
    try:
        return extract_meta_feature(Z[o-SEQ:o])
    except Exception:
        return {'__failed__': 1.0}  # constant window; features filled with 0 later, as TimeFuse's Dataset_Meta does for NaN
t = time.time()
with Pool(2) as p: feats = p.map(f, origins, chunksize=64)
M = pd.DataFrame(feats); M.insert(0, 'origin', origins)
M.to_parquet(f'/home/claude/tf_work/meta_{ds}.parquet')
print(ds, M.shape, time.time()-t)
