"""후보 점수화와 Recall@K / NDCG@K / 신규 영화 Recall@K 계산 (GPU 배치)."""
import os
import sys

import numpy as np
import pandas as pd
import torch

from .config import TOP_K


def pick_gpu(min_free_mib):
    """공용 서버 보호: 여유 메모리가 가장 큰 GPU 를 고르고, 임계치 미만이면 중단.
    환경변수 VISPROFILE_DEVICE=cpu 이면 GPU 를 쓰지 않는다."""
    if os.environ.get("VISPROFILE_DEVICE") == "cpu" or not torch.cuda.is_available():
        print("device: cpu", flush=True)
        return torch.device("cpu")
    free =[torch.cuda.mem_get_info(i)[0] // 2**20 for i in range(torch.cuda.device_count())]
    g = int(np.argmax(free))
    if free[g] < min_free_mib:
        sys.exit(f"ABORT: max free GPU memory {free[g]} MiB < {min_free_mib}")
    print(f"GPU free MiB {free} -> cuda:{g}", flush=True)
    return torch.device(f"cuda:{g}")


def zscore(A, valid):
    """유효 후보 안에서 행별 표준화 (서로 다른 척도의 점수를 결합하기 위함)."""
    n = valid.sum(1, keepdim=True).clamp(min=2)
    mu = (A * valid).sum(1, keepdim=True) / n
    sd = (((A - mu) ** 2) * valid).sum(1, keepdim=True).div(n - 1).sqrt().clamp(min=1e-8)
    return (A - mu) / sd


def evaluate(ev, X, first_day, methods, device, new_cut, pop=None, day_idx=None, bs=1024):
    """methods: 이름 리스트.
      'random' | 'pop_cum' | 'pop_recent' | 프로필 이름(ev.profiles 키) |
      'hybrid|<프로필>|<w>'  = (1-w) z(log(1+최근 인기도)) + w z(시각 유사도)
    반환: {method: {'recall','ndcg','recall_new'}: (n_events,)}
    """
    Xg = torch.from_numpy(X).to(device)
    fdg = torch.from_numpy(first_day).to(device)
    n_ev, n_items = len(ev), X.shape[0]
    disc = 1.0 / torch.log2(torch.arange(TOP_K, device=device, dtype=torch.float32) + 2)
    gen = torch.Generator(device=device).manual_seed(0)
    out = {m: {k: np.zeros(n_ev) for k in ("recall", "ndcg", "recall_new")} for m in methods}

    def visual(name, b):
        return torch.from_numpy(ev.profiles[name][b]).to(device) @ Xg.T

    for s in range(0, n_ev, bs):
        b = np.arange(s, min(s + bs, n_ev))
        B = len(b)
        # 무효 후보: 아직 존재하지 않는 영화 + 이미 평가한 영화
        invalid = fdg[None, :] >= torch.from_numpy(ev.day[b]).to(device)[:, None]
        rows = np.concatenate([np.full(ev.excl_n[i], j) for j, i in enumerate(b)])
        cols = np.concatenate([ev.rated[ev.excl_start[i]: ev.excl_start[i] + ev.excl_n[i]] for i in b])
        invalid[torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)] = True
        trow = np.concatenate([np.full(len(ev.targets[i]), j) for j, i in enumerate(b)])
        tcol = np.concatenate([ev.targets[i] for i in b])
        tmask = torch.zeros(B, n_items, dtype=torch.bool, device=device)
        tmask[torch.from_numpy(trow).to(device), torch.from_numpy(tcol).to(device)] = True
        newmask = tmask & (fdg >= new_cut)[None, :]
        nt, nn = tmask.sum(1).float(), newmask.sum(1).float()

        for m in methods:
            if m == "random":
                S = torch.rand(B, n_items, device=device, generator=gen)
            elif m.startswith("pop"):
                S = torch.from_numpy(pop[m][day_idx[b]]).to(device)
                S = S + 1e-6 * torch.rand(B, n_items, device=device, generator=gen)  # 동점 무작위 처리
            elif m.startswith("hybrid"):
                _, name, w = m.split("|")
                w = float(w)
                A = torch.log1p(torch.from_numpy(pop["pop_recent"][day_idx[b]]).to(device))
                S = (1 - w) * zscore(A, ~invalid) + w * zscore(visual(name, b), ~invalid)
            else:
                S = visual(m, b)
            top = S.masked_fill(invalid, float("-inf")).topk(TOP_K, dim=1).indices
            hit = tmask.gather(1, top).float()
            idcg = torch.stack([disc[: int(min(n, TOP_K))].sum() for n in nt.tolist()])
            rn = newmask.gather(1, top).float().sum(1) / nn
            out[m]["recall"][b] = (hit.sum(1) / nt).cpu().numpy()
            out[m]["ndcg"][b] = ((hit * disc).sum(1) / idcg).cpu().numpy()
            out[m]["recall_new"][b] = torch.where(nn > 0, rn, torch.full_like(rn, float("nan"))).cpu().numpy()
    return out


def summarize(ev, out):
    """사용자별 평균 후 전체 평균 (recall_new 는 신규 정답이 있는 이벤트만)."""
    rows = {m: pd.DataFrame({"u": ev.user, **d}).groupby("u").mean().mean() for m, d in out.items()}
    return pd.DataFrame(rows).T
