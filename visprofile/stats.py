"""사용자 단위 짝비교 검정 (Wilcoxon 부호순위 + Holm 보정)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_user_test(users, a, b):
    df = pd.DataFrame({"u": users, "a": a, "b": b}).groupby("u").mean().dropna()
    d = df.a - df.b
    return dict(mean_a=df.a.mean(), mean_b=df.b.mean(), rel=df.a.mean() / df.b.mean() - 1,
                win=int((d > 0).sum()), tie=int((d == 0).sum()), loss=int((d < 0).sum()),
                p=wilcoxon(df.a, df.b).pvalue, n_users=len(df))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
