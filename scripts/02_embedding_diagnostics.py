"""중심화 전/후 임베딩 분포 진단 (3.1절): 포스터 쌍 유사도, 사용자 프로필 간 유사도, 사용자별 점수 폭."""
import numpy as np
import pandas as pd

from _common import parse
from visprofile.data import center_normalize, load_dataset

args, paths, cfg = parse(__doc__)
ds = load_dataset(paths, cfg)
T = cfg.to_day(cfg.test_cut)
raw = np.load(paths.embedding(cfg.encoder))
rng = np.random.default_rng(0)
liked = ds.users[(ds.users.rating >= cfg.like_threshold) & (ds.users.day < T)]
groups = [g.values for _, g in liked.groupby("userId").item if len(g) >= cfg.min_pre_liked]
cand = np.where(ds.first_day < T)[0]

for name, X in [("raw", raw), ("centered", center_normalize(raw))]:
    i, j = rng.integers(0, len(X), (2, 200_000))
    pair = (X[i] * X[j]).sum(1)[i != j]
    P = np.stack([X[g].mean(0) for g in groups])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    a, b = rng.integers(0, len(P), (2, 50_000))
    upair = (P[a] * P[b]).sum(1)[a != b]
    S = P[:2000] @ X[cand].T
    spread = S.max(1) - np.median(S, 1)
    print(f"[{name}] poster-pair cos {pair.mean():.3f}  user-profile cos {upair.mean():.3f}  "
          f"score range(max-median) {spread.mean():.3f}")
