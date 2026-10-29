"""포스터 커버리지와 기준 시점 T 후보별 평가 가능 사용자 수 (T 선택 근거)."""
import os

import numpy as np
import pandas as pd

from _common import parse
from visprofile.config import DAY
from visprofile.data import filter_users

args, paths, cfg = parse(__doc__)
r = pd.read_csv(paths.ratings)
posters = {int(f[:-4]) for f in os.listdir(paths.posters) if f.endswith(".jpg")}
r["has_poster"] = r.movieId.isin(posters)
rated = set(r.movieId.unique())
print(f"rated movies with poster: {len(rated & posters) / len(rated):.1%}   ratings covered: {r.has_poster.mean():.2%}")
print(f"users rating everything on a single day: {(r.groupby('userId').timestamp.agg(lambda t: (t // DAY).nunique()) == 1).mean():.1%}")

r["day"] = r.timestamp // DAY
first_seen = r.groupby("movieId").day.min()
f = filter_users(r, cfg)
print(f"filtered users: {f.userId.nunique():,}  ratings: {len(f):,}")
p = f[(f.rating >= cfg.like_threshold) & f.has_poster]
print(f"{'T':>6} {'eval_users':>10}")
for year in range(2003, 2015):
    T = cfg.to_day(f"{year}-01-01")
    cand = set(first_seen[first_seen < T].index) & posters
    pre = p[p.day < T].groupby("userId").size()
    post = p[(p.day >= T) & p.movieId.isin(cand)].groupby("userId").day.nunique()
    users = pre[pre >= cfg.min_pre_liked].index.intersection(post[post >= cfg.min_eval_days].index)
    print(f"{year:>6} {len(users):>10,}")
