"""
Timbre-descriptor handling at inference (Nov 2026).

Aperiodicity (raw channel 4) and spectral centroid (raw channel 5) describe the SOURCE recording's
timbre (noisiness, brightness). In cross-instrument transfer the model copies them, so the output
keeps the source's character (flute -> "trumpet" still sounds like a flute). Three modes:

  "source"     : pass them through (training default; best for like-to-like, e.g. violin -> violin)
  "instrument" : drop them: set to 0 in normalized space, exactly what channel dropout did in
                 training (p = 0.4), so v3 knows this state. The chosen instrument decides timbre.
  "retarget"   : keep their MOVEMENT (a note getting brighter as it swells) but move and scale them
                 to the target instrument's typical range, computed from Slakh training contours.

Only meaningful for models trained with channel dropout (v3). Per-instrument stats:
  .venv/bin/python -m data.timbre_features      # writes artifacts/slakh_v3/timbre_feature_stats.npz
"""
import os
import numpy as np

TIMBRE_RAW_CHANNELS = (4, 5)          # aperiodicity, centroid in the raw 9-channel contour
STATS_PATH = "artifacts/slakh_v3/timbre_feature_stats.npz"
TIMBRE_MODES = ("instrument", "retarget", "source")
MIN_FRAMES = 4


def _active(raw):
    """Frames that carry a note: voiced, else (unpitched / chordal clips) the louder half."""
    v = raw[1] > 0.5
    if v.sum() >= MIN_FRAMES:
        return v
    return raw[2] >= np.median(raw[2])


def compute_stats(n_per_instrument=300, seed=0, out_path=STATS_PATH):
    """Mean / std of aperiodicity and centroid over active frames, per Slakh class (train split)."""
    import pandas as pd
    from data.dataset import get_split
    df = pd.read_csv("data/manifest.csv").dropna(subset=["uuid4"])
    df = df[df["uuid4"].apply(get_split) == "train"]
    mean = np.zeros((18, 2), "float32"); std = np.zeros((18, 2), "float32"); count = np.zeros(18, int)
    for inst in range(18):
        rows = df[df["instrument_id"].astype(int) == inst].sample(frac=1, random_state=seed)
        vals = []
        for u in rows["uuid4"]:
            raw = np.load(f"artifacts/contours/{u}.npy")
            if (raw[1] > 0.5).sum() < 8:          # skip near-silent chunks
                continue
            vals.append(raw[list(TIMBRE_RAW_CHANNELS)][:, _active(raw)])
            if len(vals) == n_per_instrument:
                break
        x = np.concatenate(vals, axis=1)
        mean[inst], std[inst], count[inst] = x.mean(1), x.std(1), len(vals)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, mean=mean, std=std, count=count, channels=np.array(TIMBRE_RAW_CHANNELS))
    return mean, std, count


_STATS = None


def load_stats(path=STATS_PATH):
    global _STATS
    if _STATS is None:
        s = np.load(path)
        _STATS = (s["mean"], s["std"])
    return _STATS


def retarget_raw(raw, instrument_id, stats=None):
    """Affine map of the clip's aperiodicity/centroid onto the target instrument's distribution:
    z-score within the clip (active frames), then target mean + z * target std. Movement kept,
    level and spread replaced. Aperiodicity is clipped to [0, 1], centroid to >= 0."""
    mean, std = stats if stats is not None else load_stats()
    out = raw.copy()
    act = _active(raw)
    if act.sum() < MIN_FRAMES:
        return out
    for j, ch in enumerate(TIMBRE_RAW_CHANNELS):
        x = raw[ch]
        mu, sd = x[act].mean(), x[act].std()
        z = (x - mu) / max(sd, 1e-6) if sd > 1e-6 else np.zeros_like(x)
        out[ch] = mean[instrument_id, j] + np.clip(z, -3, 3) * std[instrument_id, j]
    out[4] = np.clip(out[4], 0.0, 1.0)
    out[5] = np.maximum(out[5], 0.0)
    return out


def apply_mode_normalized(c, mode, positions=(4, 5)):
    """'instrument' mode on an already-normalized (C, T) input: zero the two channels
    (positions = their index within the model's selected channels; 4, 5 for v3)."""
    if mode == "instrument":
        c = c.copy()
        c[list(positions)] = 0.0
    return c


if __name__ == "__main__":
    m, s, n = compute_stats()
    names = ["Accordion", "Grand Piano", "Ac. Guitar", "Alto Sax", "Brass Section", "Cello", "Choir Aahs",
             "Clarinet", "Dist. Guitar", "El. Bass", "El. Guitar", "El. Piano", "Flute", "Rock Organ",
             "String Ens.", "Trombone", "Trumpet", "Violin"]
    print(f"{'instrument':14s} {'aper mean':>9s} {'aper std':>8s} {'centroid Hz':>11s} {'cent std':>8s} {'clips':>5s}")
    for i in range(18):
        print(f"{names[i]:14s} {m[i,0]:9.3f} {s[i,0]:8.3f} {m[i,1]:11.0f} {s[i,1]:8.0f} {n[i]:5d}")
