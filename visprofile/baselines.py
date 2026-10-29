"""서버 측 전역 신호: 인기도와 비개인화 최근 프로필. 모두 예측일 이전 기록만 사용한다."""
import numpy as np


def popularity_table(all_ratings, n_items, days, window=None):
    """pop[j, i] = 영화 i 의 평점 수, 기간 [days[j] - window, days[j]) (window=None 이면 누적)."""
    f = all_ratings.sort_values("day")
    fd, fi = f.day.values, f.item.values

    def count_before(qdays):
        out = np.zeros((len(qdays), n_items), dtype=np.float32)
        acc = np.zeros(n_items, dtype=np.float64)
        p = 0
        for j, d in enumerate(qdays):
            q = np.searchsorted(fd, d, "left")
            np.add.at(acc, fi[p:q], 1)
            p = q
            out[j] = acc
        return out

    c = count_before(days)
    if window:
        c -= count_before(days - window)
    return c


def global_recent_profiles(X, liked_ratings, days, window):
    """각 예측일 d 에 대해, 전체 사용자가 [d - window, d) 에 선호한 영화 포스터의 평균."""
    f = liked_ratings.sort_values("day")
    fd, fi = f.day.values, f.item.values
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X[fi], axis=0, dtype=np.float64)])
    hi = np.searchsorted(fd, days, "left")
    lo = np.searchsorted(fd, days - window, "left")
    return ((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)[:, None]).astype(np.float32)
