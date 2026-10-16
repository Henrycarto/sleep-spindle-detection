"""Reproduce every number in the README.

    python -m spindles.evaluate --data data

Protocol: leave-one-excerpt-out (LOEO). Anything tuned (thresholds, the
classifier) is fit on 7 excerpts and scored on the 8th, so no excerpt is ever
scored by parameters that saw it. Expert 1 is the primary reference (all 8
excerpts, first 1000 s). Expert 2 (excerpts 1-6, full 1800 s) is a second check.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from .detect import (FEATURE_NAMES, candidate_features, detect_morlet_2023,
                     detect_morlet_v2, sigma_features)
from .io import load_all
from .metrics import Counts, clip_to_window, match_events, score

GRID = dict(k_hi=[4, 5, 6, 8, 10, 12, 15], lo=[0.4, 0.5, 0.6], rel_min=[0.0, 0.1, 0.2])
CAND = dict(k_hi=3.0, k_lo=1.5, rel_min=0.0)  # liberal candidate generator for the ML stage
SEED = 0


def _ref_counts(detections, X, expert):
    c, per = Counts(), {}
    for ex, d in zip(X, detections):
        if len(ex.scorers) > expert:
            k = score(d, ex.scorers[expert])
            c += k
            per[ex.name] = round(k.f1, 3)
    return c, per


def _summary(c, per=None):
    out = dict(precision=round(c.precision, 3), recall=round(c.recall, 3),
               f1=round(c.f1, 3), tp=c.tp, fp=c.fp, fn=c.fn)
    if per is not None:
        out["per_excerpt_f1"] = per
    return out


def tune_v2(X, F, idx):
    best = None
    for k, lo, rel in itertools.product(GRID["k_hi"], GRID["lo"], GRID["rel_min"]):
        c = Counts()
        for i in idx:
            d = detect_morlet_v2(X[i].eeg, X[i].fs, k_hi=k, k_lo=k * lo, rel_min=rel, feats=F[i])
            c += score(d, X[i].scorers[0])
        if best is None or c.f1 > best[0]:
            best = (c.f1, dict(k_hi=k, k_lo=k * lo, rel_min=rel))
    return best[1]


def candidates(X, F):
    out = []
    for ex, f in zip(X, F):
        ev = detect_morlet_v2(ex.eeg, ex.fs, feats=f, **CAND)
        feats = candidate_features(ex.eeg, ex.fs, ev, f)
        ref = clip_to_window(ex.scorers[0].events, ex.scorers[0].scored_until)
        y = np.array([match_events(e[None], ref).tp for e in ev], int)
        inside = ev[:, 0] < ex.scorers[0].scored_until if len(ev) else np.zeros(0, bool)
        out.append(dict(ev=ev, X=feats, y=y, labelled=inside))
    return out


def fit_threshold(clf, Xtr, ytr, gtr, C, train_idx, X):
    """Pick the probability cut on out-of-fold predictions inside the training set."""
    p = cross_val_predict(clf, Xtr, ytr, groups=gtr, cv=GroupKFold(n_splits=len(set(gtr))),
                          method="predict_proba")[:, 1]
    best, start = (0, 0.5), 0
    offsets = []
    for i in train_idx:
        n = int(C[i]["labelled"].sum()); offsets.append((i, start, start + n)); start += n
    for t in np.arange(0.2, 0.81, 0.05):
        c = Counts()
        for i, a, b in offsets:
            ev = C[i]["ev"][C[i]["labelled"]][p[a:b] >= t]
            c += score(ev, X[i].scorers[0])
        if c.f1 > best[0]:
            best = (c.f1, float(t))
    return best[1]


def run(data_dir):
    X = load_all(data_dir)
    F = [sigma_features(ex.eeg, ex.fs) for ex in X]
    n = len(X)
    res = {"excerpts": [ex.name for ex in X]}

    # 1. inter-rater ceiling (expert 2 vs expert 1, where both scored)
    c = Counts()
    for ex in X:
        if len(ex.scorers) > 1:
            t = min(s.scored_until for s in ex.scorers)
            c += match_events(clip_to_window(ex.scorers[1].events, t),
                              clip_to_window(ex.scorers[0].events, t))
    res["inter_rater_e2_vs_e1"] = _summary(c)

    # 2. original 2023 detector, unchanged, scored correctly
    d23 = [detect_morlet_2023(ex.eeg, ex.fs) for ex in X]
    res["morlet_2023"] = {"vs_expert1": _summary(*_ref_counts(d23, X, 0)),
                          "vs_expert2": _summary(*_ref_counts(d23, X, 1))}

    # 3. reworked wavelet detector, thresholds tuned LOEO
    dv2, params = [], {}
    for i in range(n):
        p = tune_v2(X, F, [j for j in range(n) if j != i])
        params[X[i].name] = p
        dv2.append(detect_morlet_v2(X[i].eeg, X[i].fs, feats=F[i], **p))
    res["morlet_v2_loeo"] = {"vs_expert1": _summary(*_ref_counts(dv2, X, 0)),
                             "vs_expert2": _summary(*_ref_counts(dv2, X, 1)),
                             "chosen_params": params}

    # 4. candidates + gradient-boosted re-scoring, LOEO
    C = candidates(X, F)
    dml, cuts = [], {}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr = np.vstack([C[j]["X"][C[j]["labelled"]] for j in tr])
        ytr = np.concatenate([C[j]["y"][C[j]["labelled"]] for j in tr])
        gtr = np.concatenate([[j] * int(C[j]["labelled"].sum()) for j in tr])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                             l2_regularization=1.0, random_state=SEED)
        cut = fit_threshold(clf, Xtr, ytr, gtr, C, tr, X)
        clf.fit(Xtr, ytr)
        cuts[X[i].name] = round(cut, 2)
        keep = clf.predict_proba(C[i]["X"])[:, 1] >= cut if len(C[i]["X"]) else np.zeros(0, bool)
        dml.append(C[i]["ev"][keep])
    res["wavelet_plus_gbm_loeo"] = {"vs_expert1": _summary(*_ref_counts(dml, X, 0)),
                                    "vs_expert2": _summary(*_ref_counts(dml, X, 1)),
                                    "prob_cut": cuts, "features": FEATURE_NAMES,
                                    "n_candidates": int(sum(len(c["ev"]) for c in C))}
    return res, X, F, {"2023": d23, "v2": dv2, "gbm": dml}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res, X, F, dets = run(a.data)
    # Output of original/spindle_detection_2023.py run unchanged (test split,
    # scorer 1): "Avg. Test F1 Score Scorer 1: 25.26%". Kept for the comparison chart.
    res["reported_2023_test_f1_e1"] = 0.253
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v.get("vs_expert1", v) if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=2))
    try:
        from .figures import make_figures
        make_figures(res, X, F, dets, "figures")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
