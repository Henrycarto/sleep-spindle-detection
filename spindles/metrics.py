"""Event-level matching between detected and expert-scored spindles.

A detection matches a reference event if their overlap divided by their union
(IoU) is at least `min_iou`. With min_iou=0 any overlap counts. Matching is
one-to-one and greedy on IoU, so one long detection cannot claim three spindles.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, o):
        return Counts(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def clip_to_window(events, t_end):
    ev = np.asarray(events, float).reshape(-1, 2)
    return ev[ev[:, 0] < t_end]


def match_events(detected, reference, min_iou=0.0) -> Counts:
    det = np.asarray(detected, float).reshape(-1, 2)
    ref = np.asarray(reference, float).reshape(-1, 2)
    if len(det) == 0 or len(ref) == 0:
        return Counts(0, len(det), len(ref))
    inter = np.clip(np.minimum(det[:, None, 1], ref[None, :, 1])
                    - np.maximum(det[:, None, 0], ref[None, :, 0]), 0, None)
    union = (np.maximum(det[:, None, 1], ref[None, :, 1])
             - np.minimum(det[:, None, 0], ref[None, :, 0]))
    iou = np.where(inter > 0, inter / union, 0.0)
    ok = (iou > min_iou) if min_iou == 0 else (iou >= min_iou)
    pairs = sorted(zip(*np.nonzero(ok)), key=lambda ij: -iou[ij])
    used_d, used_r = set(), set()
    for i, j in pairs:
        if i not in used_d and j not in used_r:
            used_d.add(i); used_r.add(j)
    tp = len(used_d)
    return Counts(tp, len(det) - tp, len(ref) - tp)


def score(detected, scorer, min_iou=0.0) -> Counts:
    """Compare only inside the window the expert actually scored."""
    return match_events(clip_to_window(detected, scorer.scored_until),
                        clip_to_window(scorer.events, scorer.scored_until), min_iou)
