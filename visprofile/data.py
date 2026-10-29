"""데이터 적재: 평점, 포스터 임베딩, 영화 첫 등장일, 사용자 필터."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DAY, Config, Paths

NEVER = np.iinfo(np.int64).max


@dataclass
class Dataset:
    X: np.ndarray          # (n_items, d) 중심화·정규화된 포스터 임베딩
    movie_ids: np.ndarray  # X 의 행 -> MovieLens movieId
    first_day: np.ndarray  # 영화별 첫 평가일 (전체 사용자 기준), 후보 존재 판정용
    all_ratings: pd.DataFrame  # 포스터가 있는 전체 평점 [item, day, rating] (인기도 계산용)
    users: pd.DataFrame        # 필터를 통과한 사용자 평점 [userId, item, day, rating, timestamp]


def center_normalize(E):
    """평균 벡터를 빼고 L2 정규화 (CLIP 임베딩의 비등방성 완화)."""
    X = E - E.mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load_dataset(paths: Paths, cfg: Config) -> Dataset:
    E = np.load(paths.embedding(cfg.encoder))
    ids = np.load(paths.movie_ids)
    X = center_normalize(E)

    r = pd.read_csv(paths.ratings)
    r["day"] = (r.timestamp // DAY).astype(np.int64)
    r["item"] = r.movieId.map(pd.Series(np.arange(len(ids)), index=ids)).fillna(-1).astype(np.int64)
    all_ratings = r[r.item >= 0][["item", "day", "rating"]]

    first_day = np.full(len(ids), NEVER, dtype=np.int64)
    fd = all_ratings.groupby("item").day.min()
    first_day[fd.index.values] = fd.values

    users = filter_users(r, cfg)
    return Dataset(X, ids, first_day, all_ratings, users[users.item >= 0])


def filter_users(r, cfg: Config):
    """가입 초기 평점 제거 후, 충분한 기간에 걸쳐 평가한 사용자만 남긴다."""
    r = r.sort_values(["userId", "timestamp"], kind="stable")
    r = r[r.groupby("userId").cumcount() >= cfg.drop_first]
    g = r.groupby("userId").agg(n=("movieId", "size"), days=("day", "nunique"))
    keep = g[(g.n >= cfg.min_ratings) & (g.days >= cfg.min_days)].index
    return r[r.userId.isin(keep)]
