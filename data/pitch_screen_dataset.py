"""
Pitch-fix screening dataset: fold-based, violin-only Medley-solos-DB.

- Split by the manifest's `fold` column (song-level, no leakage):
  train = 0,1,2 / val = 3 / test = 4. Same SPLIT_FOLDS as compute_pitch_screen_stats.py.
- stats_path and fill_unvoiced are constructor arguments, so cell A0 (legacy) and
  cells A-D (voiced) never share module-level constants.
- fill_unvoiced=True: after normalizing, log_f0 (channel 0) is set to 0.0 (the stats mean)
  on frames where the RAW voiced flag is <= 0.5. Keyed on voiced, not log_f0 == 0, because
  ~35% of unvoiced frames still carry an unreliable log_f0 value.
  Use True for cells A-D, False for A0.
- Everything is preloaded into RAM (4,075 train clips is small).
- Returns all 9 contour channels (normalized); the training script slices what it needs.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEDLEY_ROOT = "archive/Medley_solos_DB"
MANIFEST = f"{MEDLEY_ROOT}/data/manifest.csv"
LATENT_DIR = f"{MEDLEY_ROOT}/artifacts/latents"
CONTOUR_DIR = f"{MEDLEY_ROOT}/artifacts/contours"

SPLIT_FOLDS = {"train": [0, 1, 2], "val": [3], "test": [4]}
VOICED_THRESH = 0.5


# Named instrument sets. "all8" = every Medley-solos-DB instrument (ids 0-7 in the manifest).
INSTRUMENT_SETS = {"violin": ["violin"], "all8": None}
N_MEDLEY_INSTRUMENTS = 8


def get_split_rows(df, split, instrument="violin"):
    """instrument: one name ("violin"), a list of names, a set name from INSTRUMENT_SETS, or None (= all)."""
    if isinstance(instrument, str) and instrument in INSTRUMENT_SETS:
        instrument = INSTRUMENT_SETS[instrument]
    rows = df[df["fold"].isin(SPLIT_FOLDS[split])]
    if instrument is not None:
        names = [instrument] if isinstance(instrument, str) else list(instrument)
        rows = rows[rows["instrument"].isin(names)]
    return rows.reset_index(drop=True)


# Default periods (semitones): octave, minor third, semitone. Used by cells B and D, and as
# the fallback for any checkpoint saved before periods became configurable.
FOURIER_PERIODS = (12.0, 3.0, 1.0)


def fourier_pitch_features(raw_contour, periods=FOURIER_PERIODS):
    """Raw (9, T) contour -> (2*len(periods), T) sin/cos pitch features.
    m = 12*log2(f0/440) in semitones; sin/cos(2*pi*m/P) for P in periods.
    Unvoiced frames are 0 on every Fourier channel (= "unknown", same as CFG dropout)."""
    log_f0 = raw_contour[0]
    voiced = raw_contour[1] > VOICED_THRESH
    m = 12.0 * (log_f0 - np.log(440.0)) / np.log(2.0)
    feats = []
    for P in periods:
        phase = 2.0 * np.pi * m / P
        feats.append(np.sin(phase))
        feats.append(np.cos(phase))
    feats = np.stack(feats).astype("float32")
    feats[:, ~voiced] = 0.0
    return feats


def normalize_contour(raw_contour, contour_mean, contour_std, fill_unvoiced, fourier=False,
                      periods=FOURIER_PERIODS):
    """Raw (9, T) contour -> normalized (9, T), or (9 + 2*len(periods), T) with fourier=True
    (channels 9+ = Fourier pitch features).
    Use this same function at inference so the silence rule stays identical."""
    c = (raw_contour - contour_mean) / contour_std
    if fill_unvoiced:
        unvoiced = raw_contour[1] <= VOICED_THRESH   # raw voiced flag, before normalizing
        c[0, unvoiced] = 0.0
    if fourier:
        c = np.concatenate([c, fourier_pitch_features(raw_contour, periods)], axis=0)
    return c.astype("float32")


class MedleyViolinDataset(Dataset):
    def __init__(self, split, stats_path, fill_unvoiced, fourier=False, fourier_periods=FOURIER_PERIODS,
                 instrument="violin", verbose=True):
        if split not in SPLIT_FOLDS:
            raise ValueError(f"Unknown split: {split}")

        df = pd.read_csv(MANIFEST).dropna(subset=["uuid4"])
        self.rows = get_split_rows(df, split, instrument)
        self.split = split
        self.stats_path = stats_path
        self.fill_unvoiced = fill_unvoiced
        self.fourier = fourier
        self.fourier_periods = tuple(float(p) for p in fourier_periods)

        stats = np.load(stats_path)
        self.latent_mean = stats["latent_mean"][:, None].astype("float32")
        self.latent_std = stats["latent_std"][:, None].astype("float32")
        self.contour_mean = stats["contour_mean"][:, None].astype("float32")
        self.contour_std = stats["contour_std"][:, None].astype("float32")
        self.recipe = str(stats["recipe"]) if "recipe" in stats else "unknown"

        if verbose:
            counts = self.rows["instrument"].value_counts().to_dict()
            print(f"[MedleyViolinDataset] {split}: {len(self.rows)} clips {counts} | "
                  f"stats={stats_path} ({self.recipe}) | fill_unvoiced={fill_unvoiced} | fourier={fourier}")

        latents, contours = [], []
        for uuid in self.rows["uuid4"]:
            latent = np.load(os.path.join(LATENT_DIR, f"{uuid}.npy")).astype("float32")
            raw_contour = np.load(os.path.join(CONTOUR_DIR, f"{uuid}.npy")).astype("float32")
            latents.append((latent - self.latent_mean) / self.latent_std)
            contours.append(normalize_contour(raw_contour, self.contour_mean,
                                              self.contour_std, fill_unvoiced, fourier,
                                              self.fourier_periods))
        self.latents = np.stack(latents)      # (N, 256, 32)
        self.contours = np.stack(contours)    # (N, 9, 32), or (N, 9 + 2*len(periods), 32) with fourier
        self.instrument_ids = self.rows["instrument_id"].to_numpy().astype("int64")

    def stats_dict(self):
        """Everything a checkpoint needs to reproduce this normalization (Step 1d)."""
        return {
            "stats_path": self.stats_path,
            "stats_recipe": self.recipe,
            "fill_unvoiced": self.fill_unvoiced,
            "fourier": self.fourier,
            "fourier_periods": self.fourier_periods,
            "contour_mean": self.contour_mean[:, 0].copy(),
            "contour_std": self.contour_std[:, 0].copy(),
            "latent_mean": self.latent_mean[:, 0].copy(),
            "latent_std": self.latent_std[:, 0].copy(),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {
            "latent": torch.from_numpy(self.latents[i]),
            "contour": torch.from_numpy(self.contours[i]),
            "instrument_id": torch.tensor(self.instrument_ids[i], dtype=torch.long),
        }


# SMOKE TEST
if __name__ == "__main__":
    LEGACY = "artifacts/pitch_screen/norm_stats_violin_legacy.npz"
    VOICED = "artifacts/pitch_screen/norm_stats_violin_voiced.npz"

    for split in ["train", "val", "test"]:
        ds = MedleyViolinDataset(split, VOICED, fill_unvoiced=True, verbose=False)
        print(f"{split}: {len(ds)} clips")

    configs = [("A0: legacy, no fill", LEGACY, False),
               ("voiced, NO fill (the cliff)", VOICED, False),
               ("A-D: voiced + fill", VOICED, True)]
    for name, path, fill in configs:
        ds = MedleyViolinDataset("train", path, fill_unvoiced=fill, verbose=False)
        raw_voiced = np.stack([np.load(os.path.join(CONTOUR_DIR, f"{u}.npy"))[1]
                               for u in ds.rows["uuid4"]]) > VOICED_THRESH
        f0 = ds.contours[:, 0, :]
        print(f"\n--- {name} ---")
        print(f"ch0 all       min {f0.min():8.3f}  max {f0.max():7.3f}")
        print(f"ch0 voiced    mean {f0[raw_voiced].mean():7.3f}  std {f0[raw_voiced].std():6.3f}")
        print(f"ch0 unvoiced  min {f0[~raw_voiced].min():8.3f}  max {f0[~raw_voiced].max():7.3f}")
        print(f"latent        mean {ds.latents.mean():7.3f}  std {ds.latents.std():6.3f}")

    b = ds[0]
    print("\nitem:", {k: tuple(v.shape) for k, v in b.items()}, "| instrument_id", int(b["instrument_id"]))
