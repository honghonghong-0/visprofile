"""RQ1·RQ2 연속 평가 (표 1).

splits
  val                검증 구간 [2008, 2009): EMA α, 결합 가중치 선택
  test               테스트 구간 [2009, ...)
  test_shuffled      순서 대조군: 사용자별 T 이후 날짜 순서만 섞음
  test_shuffled_all  순서 대조군: 전체 날짜 순서를 섞음
결과: results/stream_<split>_<encoder>.csv (요약), _events.npz (이벤트별 지표; 05·06 스크립트 입력)
"""
import numpy as np

from _common import parse
from visprofile.baselines import global_recent_profiles, popularity_table
from visprofile.data import NEVER, load_dataset
from visprofile.evaluate import evaluate, pick_gpu, summarize
from visprofile.stream import build_events

SPLITS = ["val", "test", "test_shuffled", "test_shuffled_all"]
args, paths, cfg = parse(__doc__, lambda ap: ap.add_argument("splits", nargs="*", default=["val", "test"],
                                                             help=f"{SPLITS}"))
device = pick_gpu(cfg.min_free_gpu_mib)
ds = load_dataset(paths, cfg)
val_cut, test_cut = cfg.to_day(cfg.val_cut), cfg.to_day(cfg.test_cut)
liked_all = ds.all_ratings[ds.all_ratings.rating >= cfg.like_threshold]

for split in args.splits:
    cut, end = (val_cut, test_cut) if split == "val" else (test_cut, NEVER)
    shuffle = dict(shuffle_seed=0, shuffle_all=split.endswith("_all")) if "shuffled" in split else {}
    ev = build_events(ds, cfg, cut, end, **shuffle)
    ev.profiles["global_recent"] = global_recent_profiles(ds.X, liked_all, ev.day, cfg.recent_window_days)

    udays = np.unique(ev.day)
    day_idx = np.searchsorted(udays, ev.day)
    n_items = len(ds.movie_ids)
    pop = {"pop_cum": popularity_table(ds.all_ratings, n_items, udays),
           "pop_recent": popularity_table(ds.all_ratings, n_items, udays, cfg.recent_window_days)}

    methods = ["random", "pop_cum", "pop_recent", "global", "global_recent", "static", "mean"]
    methods += [f"ema{a}" for a in cfg.alphas]
    methods += [f"hybrid|ema0.05|{w}" for w in cfg.hybrid_weights]
    out = evaluate(ev, ds.X, ds.first_day, methods, device, new_cut=cut, pop=pop, day_idx=day_idx)

    res = summarize(ev, out)
    print(f"\n== {split}: users {len(np.unique(ev.user)):,}  events {len(ev):,}")
    print(res.round(4).to_string(), flush=True)
    stem = f"{paths.results}/stream_{split}_{cfg.encoder}"
    res.to_csv(f"{stem}.csv")
    np.savez_compressed(f"{stem}_events.npz", user=ev.user, day=ev.day, k=ev.k,
                        **{f"{m}__{k}": v for m, d in out.items() for k, v in d.items()})
