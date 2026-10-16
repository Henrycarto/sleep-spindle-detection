"""Spindle detectors. All return an (n, 2) array of [start, end] in seconds."""
import numpy as np
from scipy.signal import butter, detrend, sosfilt, sosfiltfilt, fftconvolve


# --------------------------------------------------------------------------- #
# 2023 course detector, ported verbatim (logic unchanged, only output units
# normalised to [start_s, end_s] so it can be scored with the fixed metric).
# --------------------------------------------------------------------------- #
def detect_morlet_2023(eeg, fs):
    from mne.time_frequency import morlet, psd_array_multitaper

    sos = butter(8, [11, 16], btype="band", output="sos", fs=fs)
    filtered = sosfilt(sos, eeg)
    psd, freqs = psd_array_multitaper(filtered, fs, fmin=11, fmax=16, verbose=0)
    cf = freqs[np.argmax(psd)]
    wlt = morlet(fs, [cf], n_cycles=20)[0]
    analytic = np.convolve(eeg, wlt, mode="same")
    phase = np.angle(analytic)
    power = np.abs(analytic) ** 2
    norm_power = (power - power.min()) / (power.max() - power.min())
    supra = np.where(norm_power >= 0.05)[0]
    if len(supra) == 0:
        return np.empty((0, 2))
    sp = np.split(supra, np.where(np.diff(supra) != 1)[0] + 1)
    idx = np.array([[k[0], k[-1]] for k in sp])
    dur_ms = (np.diff(idx, axis=1) / fs * 1000).flatten()
    freq = np.array([np.median(fs / (2 * np.pi) * np.diff(phase[k])) for k in sp])

    kept = [(idx[i][0], dur_ms[i] / 1000) for i in range(len(idx))
            if 11 <= freq[i] <= 16 and 500 < dur_ms[i] < 3000]
    # original merge step (start in samples, duration in seconds, as in 2023)
    out, last, last_end = [], None, None
    for start, dur in sorted(kept):
        if last_end is not None and start - last_end < 2:
            if dur > last[1]:
                last = (start, dur)
        else:
            if last is not None:
                out.append(last)
            last = (start, dur)
        last_end = start + dur
    if last is not None:
        out.append(last)
    out = [s for s in out if s[0] != 0]
    return np.array([[s / fs, s / fs + d] for s, d in out]).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Reworked detector: same wavelet idea, fixed and made robust.
# --------------------------------------------------------------------------- #
def complex_morlet(fs, fc, n_cycles):
    sigma_t = n_cycles / (2 * np.pi * fc)
    t = np.arange(-4 * sigma_t, 4 * sigma_t, 1 / fs)
    w = np.exp(2j * np.pi * fc * t) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
    return w / np.sqrt(0.5 * np.sum(np.abs(w) ** 2))


def _moving_mean(x, n):
    n = max(1, int(n))
    return np.convolve(x, np.ones(n) / n, mode="same")


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.c_[np.where(d == 1)[0], np.where(d == -1)[0]]


def sigma_features(eeg, fs, fc=13.5, n_cycles=7):
    """Per-sample sigma envelope (robust z) and relative sigma power."""
    broad = sosfiltfilt(butter(4, [0.5, min(30.0, 0.45 * fs)], btype="band", output="sos", fs=fs), eeg)
    analytic = fftconvolve(broad, complex_morlet(fs, fc, n_cycles), mode="same")
    power = _moving_mean(np.abs(analytic) ** 2, 0.2 * fs)
    z = power / np.median(power)          # robust: one artefact cannot rescale everything
    sigma = sosfiltfilt(butter(4, [11, 16], btype="band", output="sos", fs=fs), eeg)
    win = int(fs)                          # 1 s windows
    rel = _moving_mean(sigma ** 2, win) / (_moving_mean(broad ** 2, win) + 1e-12)
    return dict(broad=broad, sigma=sigma, z=z, rel=rel, phase=np.unwrap(np.angle(analytic)))


def detect_morlet_v2(eeg, fs, k_hi=4.5, k_lo=None, rel_min=0.2,
                     min_dur=0.5, max_dur=3.0, merge_gap=0.3, feats=None):
    """Dual-threshold on the sigma envelope, gated by relative sigma power."""
    f = feats or sigma_features(eeg, fs)
    k_lo = k_hi / 2 if k_lo is None else k_lo
    core = f["z"] >= k_hi
    ext = _runs(f["z"] >= k_lo)
    ev = [r for r in ext if core[r[0]:r[1]].any()]
    # merge close events
    merged = []
    for s, e in ev:
        if merged and s - merged[-1][1] < merge_gap * fs:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        dur = (e - s) / fs
        if not (min_dur <= dur <= max_dur):
            continue
        if f["rel"][s:e].mean() < rel_min:
            continue
        out.append([s / fs, e / fs])
    return np.array(out).reshape(-1, 2)


def candidate_features(eeg, fs, events, feats):
    """Features per candidate event for the learned re-scoring stage."""
    rows = []
    for s_t, e_t in events:
        s, e = int(s_t * fs), max(int(e_t * fs), int(s_t * fs) + 2)
        seg_sig, seg_b = feats["sigma"][s:e], feats["broad"][s:e]
        inst_f = np.diff(feats["phase"][s:e]) * fs / (2 * np.pi)
        ctx = feats["broad"][max(0, s - int(2 * fs)):e + int(2 * fs)]
        rows.append([
            e_t - s_t,
            feats["z"][s:e].max(),
            feats["z"][s:e].mean(),
            feats["rel"][s:e].mean(),
            np.ptp(seg_sig),
            np.ptp(detrend(seg_b)),
            np.ptp(seg_sig) / (np.ptp(detrend(seg_b)) + 1e-9),
            np.median(inst_f) if len(inst_f) else 0.0,
            np.std(ctx) / (np.median(np.abs(feats["broad"])) + 1e-9),
        ])
    return np.array(rows).reshape(-1, len(FEATURE_NAMES))


FEATURE_NAMES = ["duration_s", "z_max", "z_mean", "rel_sigma", "sigma_ptp_uv",
                 "broad_ptp_uv", "sigma_to_broad_ptp", "inst_freq_hz", "context_std_ratio"]
