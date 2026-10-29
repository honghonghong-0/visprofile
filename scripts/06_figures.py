"""논문 그림 생성: 그림 1(갱신 횟수별 성능), 그림 2(압축), 그림 3(정성 사례)."""
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from _common import parse
from visprofile.data import NEVER, load_dataset
from visprofile.stream import build_events

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

args, paths, cfg = parse(__doc__)
R = paths.results

# ---- 그림 1: 같은 사용자 집단(T 이후 평가일 >= 8일)에서 갱신 횟수별 Recall@10
z = np.load(f"{R}/stream_test_{cfg.encoder}_events.npz")
df = pd.DataFrame({"u": z["user"], "k": z["k"]})
n = df.groupby("u").k.max() + 1
N = 8
m = (df.u.isin(n[n >= N].index) & (df.k < N)).values
fig, ax = plt.subplots(figsize=(4.2, 2.8))
for meth, lab, c, mk in [("static", "Static (no update)", "#8a8f98", "o"),
                         ("mean", "Cumulative mean (long-term)", "#2a9d8f", "s"),
                         ("ema0.05", "EMA, a=0.05 (short-term weighted)", "#2f6fdf", "^")]:
    s = pd.DataFrame({"k": df.k[m].values + 1, "v": z[f"{meth}__recall"][m]}).groupby("k").v
    mu, se = s.mean(), s.std() / np.sqrt(s.count())
    ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.12, lw=0)
    ax.plot(mu.index, mu.values, marker=mk, color=c, label=lab, lw=1.4, ms=4)
ax.set_xlabel("Update count (k-th evaluated day after T)"); ax.set_ylabel("Recall@10")
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{R}/fig1_update_curve.png", dpi=200)

# ---- 그림 2: 프로필 크기별 Recall@10
d = pd.read_csv(f"{R}/compression_{cfg.encoder}.csv")
t = pd.read_csv(f"{R}/stream_test_{cfg.encoder}.csv", index_col=0)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for prof, c, lab in [("ema0.05", "#2f6fdf", "Personal EMA"), ("mean", "#8a8f98", "Personal mean")]:
    for prec, mk in [("fp32", "o"), ("fp16", "s"), ("int8", "^")]:
        y = d[(d.profile == prof) & (d.precision == prec)].sort_values("bytes")
        ax.plot(y.bytes, y.recall, marker=mk, color=c, ls="-" if prec == "int8" else ":", ms=5, lw=1.2,
                label=f"{lab}, {prec}")
        if prof == "ema0.05" and prec == "int8":
            for _, r in y.iterrows():
                ax.annotate(f"D={r.dim}", (r.bytes, r.recall), textcoords="offset points", xytext=(0, 6),
                            fontsize=7, ha="center")
for name, lab, c in [("global_recent", "Global recent visual profile", "#d9822b"),
                     ("pop_cum", "Cumulative popularity", "#555"), ("global", "Global-mean profile", "#aaa")]:
    ax.axhline(t.loc[name, "recall"], color=c, lw=0.9, ls="--")
    ax.text(20, t.loc[name, "recall"] + 0.0005, lab, fontsize=7, color=c)
ax.set_xscale("log"); ax.set_xlabel("Profile size (bytes, log scale)"); ax.set_ylabel("Recall@10")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{R}/fig2_compression.png", dpi=170)

# ---- 그림 3: 첫 평가일 상위 5개에 정답이 있는 사용자 중, 최근 선호작의 시각적 일관성이 가장 높은 사례
ds = load_dataset(paths, cfg)
cut = cfg.to_day(cfg.test_cut)
ev = build_events(ds, cfg, cut, NEVER)
titles = pd.read_csv(paths.movies).set_index("movieId").title
best = None
first = np.r_[True, ev.user[1:] != ev.user[:-1]]
for i in np.where(first)[0]:
    valid = ds.first_day < ev.day[i]
    valid[ev.rated[ev.excl_start[i]: ev.excl_start[i] + ev.excl_n[i]]] = False
    s = np.where(valid, ds.X @ ev.profiles["ema0.05"][i], -np.inf)
    top = np.argsort(-s)[:5]
    hits = set(top) & set(ev.targets[i])
    if not hits:
        continue
    u = ds.users
    h = u[(u.userId == ev.user[i]) & (u.rating >= cfg.like_threshold) & (u.day < ev.day[i])] \
        .sort_values("timestamp").item.values[-5:]
    coh = (ds.X[h] @ ds.X[h].T)[np.triu_indices(len(h), 1)].mean()
    if best is None or coh > best[0]:
        best = (coh, h, top, hits)
_, hist, top, hits = best
W, H, GAP, LAB, TXT = 220, 330, 14, 34, 46
font, small = ImageFont.load_default(size=19), ImageFont.load_default(size=16)
canvas = Image.new("RGB", (5 * (W + GAP) + GAP, 2 * (LAB + H + TXT) + 10), "white")
dr = ImageDraw.Draw(canvas)
for row, (items, label) in enumerate([(hist, "(a) Recently liked"),
                                      (top, "(b) Top-5 by personal EMA profile (green = liked on that day)")]):
    y0 = 10 + row * (LAB + H + TXT)
    dr.text((GAP, y0), label, fill="black", font=font)
    for col, it in enumerate(items):
        x, y = GAP + col * (W + GAP), y0 + LAB
        canvas.paste(Image.open(f"{paths.posters}/{ds.movie_ids[it]}.jpg").convert("RGB").resize((W, H)), (x, y))
        if row == 1 and it in hits:
            dr.rectangle([x - 5, y - 5, x + W + 4, y + H + 4], outline=(20, 160, 60), width=7)
        t_ = titles.get(ds.movie_ids[it], "")
        dr.text((x, y + H + 6), t_[:24], fill="black", font=small)
        dr.text((x, y + H + 25), t_[24:48], fill="black", font=small)
canvas.save(f"{R}/fig3_qualitative.png", dpi=(300, 300))
print("saved fig1_update_curve.png, fig2_compression.png, fig3_qualitative.png")
