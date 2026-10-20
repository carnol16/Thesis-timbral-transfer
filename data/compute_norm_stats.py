
import os, sys, hashlib
import numpy as np
import pandas as pd

MANIFEST = "data/manifest.csv"
LATENT_DIR = "artifacts/latents"
CONTOUR_DIR = "artifacts/contours"
OUT_PATH = "artifacts/norm_stats.npz"

def get_split(uuid_str):
    # Deterministic 90/10 split
    h = int(hashlib.md5(str(uuid_str).encode()).hexdigest(), 16)
    if h % 10 < 9: return "train"
    else: return "val"

df = pd.read_csv(MANIFEST)
df = df.dropna(subset=['uuid4'])
df['split'] = df['uuid4'].apply(get_split)
train = df[df["split"] == "train"].reset_index(drop=True)
print(f"{len(train)} train clips (of {len(df)})")

# Since 300k items is too large for RAM, we'll sample 10,000 to compute stats.
# This gives a highly accurate estimate of mean/std while using ~10GB of RAM.
sample_size = min(len(train), 10000)
train_sample = train.sample(n=sample_size, random_state=42)
print(f"Sampling {sample_size} chunks to compute normalization stats without OOMing...")

latents, contours = [], []
for uuid in train_sample["uuid4"]:
    latent_path = os.path.join(LATENT_DIR, f"{uuid}.npy")
    contour_path = os.path.join(CONTOUR_DIR, f"{uuid}.npy")
    if os.path.exists(latent_path) and os.path.exists(contour_path):
        latents.append(np.load(latent_path))
        contours.append(np.load(contour_path))
    else:
        print(f"Warning: Missing files for {uuid}")

latents = np.stack(latents)
contours = np.stack(contours)
print("latents (sample):", latents.shape, "| contours (sample):", contours.shape)

# compute and freeze
# per-channel stats: collapse over clips (axis 0) and time (axis 2)
latent_mean  = latents.mean(axis=(0, 2))    # (256,)
latent_std   = latents.std(axis=(0, 2))     # (256,)
contour_mean = contours.mean(axis=(0, 2))   # (4,)
contour_std  = contours.std(axis=(0, 2))    # (4,)

# never divide by ~0
latent_std[latent_std < 1e-6] = 1.0
contour_std[contour_std < 1e-6] = 1.0

np.savez(OUT_PATH,
         latent_mean=latent_mean,   latent_std=latent_std,
         contour_mean=contour_mean, contour_std=contour_std,
         n_clips_sampled=len(latents))

print("saved", OUT_PATH)
print("latent std range: %.4f - %.4f" % (latent_std.min(), latent_std.max()))
print("contour mean:", np.round(contour_mean, 3))
print("contour std :", np.round(contour_std, 3))
