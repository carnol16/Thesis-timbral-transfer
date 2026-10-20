"""
pyTorch Dataset joining manifest + latents + contours,
with hashed splits and lazy-loading for massive datasets.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MANIFEST = "data/manifest.csv"
LATENT_DIR = "artifacts/latents"
CONTOUR_DIR = "artifacts/contours"
STATS_PATH = "artifacts/norm_stats.npz"

def get_split(uuid_str):
    # Deterministic 90/5/5 split
    h = int(hashlib.md5(str(uuid_str).encode()).hexdigest(), 16)
    if h % 10 < 9: 
        return "train"
    elif (h // 10) % 2 == 0: 
        return "val"
    else: 
        return "test"

class AudioLatentDataset(Dataset):
    def __init__(self, split="train"):
        df = pd.read_csv(MANIFEST)
        df = df.dropna(subset=['uuid4'])
        df['split'] = df['uuid4'].apply(get_split)
        
        self.rows = df[df["split"] == split].sort_values("instrument_id").reset_index(drop=True)
        print(f"Initialized Dataset [{split}]: {len(self.rows)} chunks")

        stats = np.load(STATS_PATH)
        self.latent_mean = stats["latent_mean"][:, None].astype("float32")
        self.contour_mean = stats["contour_mean"][:, None].astype("float32")
        self.latent_std = stats["latent_std"][:, None].astype("float32")
        self.contour_std = stats["contour_std"][:, None].astype("float32")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        uuid = row["uuid4"]
        
        # Lazy load from SSD (prevents 10GB RAM crash on initialization)
        latent = np.load(os.path.join(LATENT_DIR, f"{uuid}.npy")).astype("float32")
        contour = np.load(os.path.join(CONTOUR_DIR, f"{uuid}.npy")).astype("float32")
        
        # Apply precomputed normalization
        latent = (latent - self.latent_mean) / self.latent_std
        contour = (contour - self.contour_mean) / self.contour_std
        
        return {
            "latent":        torch.from_numpy(latent),
            "contour":       torch.from_numpy(contour),
            "instrument_id": torch.tensor(int(row["instrument_id"]), dtype=torch.long),
        }

# SMOKE TEST
if __name__ == "__main__":
    for split in ["train", "val"]:
        ds = AudioLatentDataset(split)
    b = ds[0]
    print("latent", tuple(b["latent"].shape), 
          "| contour,", tuple(b["contour"].shape),
          "| instrument", int(b["instrument_id"]))
            

