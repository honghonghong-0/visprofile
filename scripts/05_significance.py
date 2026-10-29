"""RQ별 사전 지정 비교의 사용자 단위 Wilcoxon 검정 + Holm 보정 (03 의 test 결과 사용)."""
import numpy as np
import pandas as pd

from _common import parse
from visprofile.stats import holm, paired_user_test

COMPARISONS = [  # (방법 A, 방법 B, 지표, RQ)
    ("mean", "global", "recall", "RQ1"),
    ("mean", "pop_cum", "recall", "RQ1"),
    ("hybrid|ema0.05|0.5", "pop_recent", "recall", "RQ1"),
    ("mean", "static", "recall", "RQ2"),
    ("ema0.05", "mean", "recall", "RQ2"),
    ("ema0.05", "global_recent", "recall", "RQ2"),
    ("ema0.05", "global_recent", "recall_new", "RQ2"),
]
args, paths, cfg = parse(__doc__)
z = np.load(f"{paths.results}/stream_test_{cfg.encoder}_events.npz")
rows = [dict(rq=rq, a=a, b=b, metric=k, **paired_user_test(z["user"], z[f"{a}__{k}"], z[f"{b}__{k}"]))
        for a, b, k, rq in COMPARISONS]
df = pd.DataFrame(rows)
df["p_holm"] = holm(df.p)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
df.to_csv(f"{paths.results}/significance_{cfg.encoder}.csv", index=False)
