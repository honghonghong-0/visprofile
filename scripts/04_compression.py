"""RQ3 프로필 압축: PCA 차원 x 저장 정밀도별 성능 (그림 2) -> results/compression_<encoder>.csv"""
import numpy as np
import pandas as pd

from _common import parse
from visprofile.data import NEVER, load_dataset
from visprofile.evaluate import evaluate, pick_gpu, summarize
from visprofile.stream import build_events
from visprofile.user_profile import pca_basis, profile_bytes, quantize

args, paths, cfg = parse(__doc__)
device = pick_gpu(cfg.min_free_gpu_mib)
ds = load_dataset(paths, cfg)
cut = cfg.to_day(cfg.test_cut)
ev = build_events(ds, cfg, cut, NEVER)
full_profiles = ev.profiles
rows = []
for dim in [768, 256, 128, 64, 32, 16]:
    W = pca_basis(ds.X, dim)
    Xp = ds.X @ W
    Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
    for prec in ["fp32", "fp16", "int8"]:
        ev.profiles = {m: quantize(full_profiles[m] @ W, prec) for m in ["mean", "ema0.05"]}
        res = summarize(ev, evaluate(ev, Xp, ds.first_day, ["mean", "ema0.05"], device, new_cut=cut))
        for m in res.index:
            rows.append(dict(profile=m, dim=dim, precision=prec, bytes=profile_bytes(dim, prec),
                             recall=res.loc[m, "recall"], ndcg=res.loc[m, "ndcg"]))
        print(f"dim={dim:>3} {prec}: {profile_bytes(dim, prec):>5} B  "
              f"mean {res.loc['mean', 'recall']:.4f}  ema {res.loc['ema0.05', 'recall']:.4f}", flush=True)
pd.DataFrame(rows).to_csv(f"{paths.results}/compression_{cfg.encoder}.csv", index=False)
