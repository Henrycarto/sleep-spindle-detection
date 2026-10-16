import numpy as np

from spindles.detect import detect_morlet_v2
from spindles.io import Scorer, mask_to_events
from spindles.metrics import match_events, score


def test_mask_to_events_roundtrip():
    m = np.zeros(100); m[10:20] = 1; m[50:55] = 1
    np.testing.assert_allclose(mask_to_events(m, 10), [[1.0, 2.0], [5.0, 5.5]])


def test_matching_is_one_to_one():
    det = [[0.0, 10.0]]                       # one long detection
    ref = [[1, 2], [4, 5], [7, 8]]            # three spindles
    c = match_events(det, ref)
    assert (c.tp, c.fp, c.fn) == (1, 0, 2)


def test_units_are_seconds_on_both_sides():
    # The 2023 metric compared samples to seconds when counting misses.
    det = [[100.0, 101.0]]
    ref = [[100.2, 100.9]]
    c = match_events(det, ref)
    assert c.tp == 1 and c.fn == 0


def test_only_scored_window_counts():
    s = Scorer(np.array([[10.0, 11.0]]), scored_until=1000.0)
    c = score([[10.1, 10.8], [1200.0, 1201.0]], s)   # second detection is unscored time
    assert (c.tp, c.fp, c.fn) == (1, 0, 0)


def test_detector_finds_synthetic_spindle():
    fs = 200.0
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 1 / fs)
    x = rng.normal(0, 5, t.size)
    for c in (15.0, 40.0):
        w = (t > c) & (t < c + 1.0)
        x[w] += 40 * np.sin(2 * np.pi * 13 * t[w]) * np.hanning(w.sum())
    ev = detect_morlet_v2(x, fs, k_hi=10, rel_min=0.1)
    c = match_events(ev, [[15, 16], [40, 41]])
    assert c.tp == 2 and c.fp == 0
