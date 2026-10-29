"""일 단위 연속(prequential) 평가 스트림 구성.

기준 시점 cut 이전 기록으로 초기 프로필을 만들고, cut 이후 매일
  1) 전날까지의 프로필로 후보를 점수화(예측) -> 2) 그날 선호작 관측 -> 3) 프로필 갱신
을 반복한다. 평가 이벤트 = (사용자, 날짜) 쌍, 정답 = 그날 선호한 영화들(복수).
후보(u, d) = d 이전에 누군가 평가해 존재가 확인된 영화 - u 가 d 이전에 평가한 영화.
d 당일 처음 등장한 영화는 후보·정답 모두에서 제외된다.
"""
from dataclasses import dataclass

import numpy as np

from .config import Config
from .user_profile import ProfileState


@dataclass
class EventSet:
    user: np.ndarray
    day: np.ndarray
    k: np.ndarray            # 사용자 내 평가 순번(0부터) = 갱신 횟수
    targets: list            # 이벤트별 정답 item 인덱스 리스트
    excl_start: np.ndarray   # rated 배열에서 이 사용자의 시작 위치
    excl_n: np.ndarray       # 예측 시점에 이미 평가한 영화 수
    rated: np.ndarray        # 사용자별 (스트림 순서) 평가 영화들을 이어 붙인 배열
    profiles: dict           # 방법 이름 -> (n_events, d) 예측 시점 프로필

    def __len__(self):
        return len(self.user)


def build_events(ds, cfg: Config, cut, end, shuffle_seed=None, shuffle_all=False):
    """shuffle_seed 를 주면 순서 대조군: cut 이후 날짜만(기본) 또는 전체 날짜(shuffle_all)를 섞는다."""
    X, first_day = ds.X, ds.first_day
    ev = dict(user=[], day=[], k=[], targets=[], excl_start=[], excl_n=[])
    prof = {"static": [], "mean": [], **{f"ema{a}": [] for a in cfg.alphas}}
    rated, offset = [], 0
    rng = np.random.default_rng(shuffle_seed)
    glob_sum, glob_n = np.zeros(X.shape[1]), 0

    for u, df in ds.users.groupby("userId", sort=True):
        items, days, liked = df.item.values, df.day.values, df.rating.values >= cfg.like_threshold
        pre = liked & (days < cut)
        if pre.sum() < cfg.min_pre_liked:
            continue
        glob_sum += X[items[pre]].sum(0)
        glob_n += pre.sum()

        order = np.unique(days)
        if shuffle_seed is not None:
            if shuffle_all:
                order = rng.permutation(order)
            else:
                order[order >= cut] = rng.permutation(order[order >= cut])
        # 스트림 순서대로 평가 영화를 정렬 -> 예측 시점의 "이미 평가한 영화" 범위를 앞부분으로 표현
        pos = {d: i for i, d in enumerate(order)}
        rank = np.array([pos[d] for d in days])
        srt = np.argsort(rank, kind="stable")
        s_items, s_rank = items[srt], rank[srt]

        static = X[items[pre]].mean(0)
        state = ProfileState(X.shape[1], cfg.alphas)
        u_events = []
        for i, d in enumerate(order):
            today = days == d
            liked_today = items[today & liked]
            if cut <= d < end and state.ready:
                tg = [t for t in liked_today if first_day[t] < d]  # 존재 판정은 실제 달력 날짜 기준
                if tg:
                    u_events.append((d, tg, int(np.searchsorted(s_rank, i, "left")), *state.snapshot()))
            state.update(X[liked_today])

        if len(u_events) < cfg.min_eval_days:
            continue
        for k, (d, tg, n_ex, mean, ema) in enumerate(u_events):
            ev["user"].append(u); ev["day"].append(d); ev["k"].append(k); ev["targets"].append(tg)
            ev["excl_start"].append(offset); ev["excl_n"].append(n_ex)
            prof["static"].append(static); prof["mean"].append(mean)
            for a in cfg.alphas:
                prof[f"ema{a}"].append(ema[a])
        rated.append(s_items)
        offset += len(s_items)

    prof = {m: np.stack(v).astype(np.float32) for m, v in prof.items()}
    n = len(ev["user"])
    prof["global"] = np.repeat((glob_sum / glob_n)[None].astype(np.float32), n, 0)  # 비개인화 대조군
    return EventSet(user=np.array(ev["user"]), day=np.array(ev["day"]), k=np.array(ev["k"]),
                    targets=ev["targets"], excl_start=np.array(ev["excl_start"]),
                    excl_n=np.array(ev["excl_n"]), rated=np.concatenate(rated), profiles=prof)
