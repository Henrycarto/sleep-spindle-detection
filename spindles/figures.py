import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PURPLE, GREY, LIGHT = "#53057F", "#9A9A9A", "#D9C7E6"


def make_figures(res, X, F, dets, out):
    Path(out).mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # F1 comparison
    rows = [("2023 detector\n(as reported)", None),
            ("2023 detector\n(scored correctly)", res["morlet_2023"]["vs_expert1"]["f1"]),
            ("Reworked wavelet\n(LOEO-tuned)", res["morlet_v2_loeo"]["vs_expert1"]["f1"]),
            ("Wavelet + GBM\n(LOEO)", res["wavelet_plus_gbm_loeo"]["vs_expert1"]["f1"])]
    reported = res.get("reported_2023_test_f1_e1")
    vals = [reported if v is None else v for _, v in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [GREY, LIGHT, PURPLE, PURPLE]
    b = ax.bar([r[0] for r in rows], vals, color=colors, width=0.6)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center")
    ir = res["inter_rater_e2_vs_e1"]["f1"]
    ax.axhline(ir, ls="--", color="black", lw=1)
    ax.text(-0.3, ir + 0.02, f"expert 2 vs expert 1: {ir:.2f}", ha="left", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Event-level F1 vs expert 1")
    ax.set_title("DREAMS spindles, 8 excerpts, C3-A2", loc="left")
    fig.tight_layout(); fig.savefig(f"{out}/f1_comparison.png", dpi=160); plt.close(fig)

    # example trace
    i = next(k for k, ex in enumerate(X) if ex.name == "excerpt2")
    ex, f = X[i], F[i]
    t0, t1 = 0.0, 1000.0
    ref = ex.scorers[0].events
    # pick a 20 s window with several expert spindles
    best = max(np.arange(t0, t1 - 20, 5), key=lambda s: ((ref[:, 0] >= s) & (ref[:, 1] <= s + 20)).sum())
    s, e = int(best * ex.fs), int((best + 20) * ex.fs)
    t = np.arange(s, e) / ex.fs
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, f["broad"][s:e], color="black", lw=0.6)
    ax[0].set_ylabel("C3-A2 (uV)")
    for a, b_ in ref:
        if a < t[-1] and b_ > t[0]:
            ax[0].axvspan(a, b_, color=LIGHT, alpha=0.9, lw=0)
    for a, b_ in dets["v2"][i]:
        if a < t[-1] and b_ > t[0]:
            ax[0].hlines(ax[0].get_ylim()[1] * 0.9, a, b_, color=PURPLE, lw=4)
    ax[0].set_title(f"{ex.name}: shaded = expert 1, bar = reworked detector (held-out fold)", loc="left")
    ax[1].plot(t, f["z"][s:e], color=PURPLE, lw=0.8)
    ax[1].set_ylabel("sigma z"); ax[1].set_xlabel("time (s)")
    ax[1].set_xlim(t[0], t[-1])
    fig.tight_layout(); fig.savefig(f"{out}/example_trace.png", dpi=160); plt.close(fig)
