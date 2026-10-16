"""Load DREAMS spindle excerpts (EDF with embedded expert annotation channels)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyedflib


@dataclass
class Scorer:
    events: np.ndarray      # (n, 2) start/end in seconds
    scored_until: float     # seconds; expert 1 only annotated the first ~1000 s


@dataclass
class Excerpt:
    name: str
    eeg: np.ndarray         # C3-A2, microvolts
    fs: float
    duration: float
    scorers: list = field(default_factory=list)


def mask_to_events(mask: np.ndarray, fs: float) -> np.ndarray:
    """Binary annotation channel -> (n, 2) array of [start, end] in seconds."""
    m = (np.asarray(mask) > 0.5).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return np.c_[starts / fs, ends / fs].astype(float)


# Scoring windows of the DREAMS spindle database: expert 1 annotated the first
# 1000 s of every excerpt, expert 2 annotated the full 1800 s of excerpts 1-6.
SCORED_UNTIL = {1: 1000.0, 2: 1800.0}


def load_excerpt(path) -> Excerpt:
    path = Path(path)
    r = pyedflib.EdfReader(str(path))
    try:
        eeg = r.readSignal(0)
        fs = r.getSampleFrequency(0)
        ex = Excerpt(path.name.split(".")[0], eeg, fs, r.getFileDuration())
        for k in range(1, r.signals_in_file):
            ev = mask_to_events(r.readSignal(k), r.getSampleFrequency(k))
            ex.scorers.append(Scorer(ev, SCORED_UNTIL.get(k, ex.duration)))
    finally:
        r.close()
    return ex


def load_all(data_dir) -> list:
    files = sorted(Path(data_dir).glob("excerpt*.edf"),
                   key=lambda p: int("".join(c for c in p.name.split(".")[0] if c.isdigit())))
    if not files:
        raise FileNotFoundError(f"No excerpt*.edf files in {data_dir}. See data/README.md.")
    return [load_excerpt(f) for f in files]
