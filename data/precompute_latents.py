"""
Encode all manifest clip to a SAME latent [256, 32], 
one .npy per clip in artifacts/latents/.
This is resumable
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "data":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

os.environ["OMP_NUM_THREADS"] = "1" # prevent thread thrashing
import csv
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from stable_audio_3 import AutoencoderModel

MANIFEST   = "data/manifest.csv"
OUT_DIR    = "artifacts/latents"
MODEL_NAME = "same-l"
SR         = 44100
BATCH_SIZE = 16  # Process 16 clips simultaneously on the GPU
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

def load_audio(row):
    if not row or not row.get("uuid4"):
        return None, None
    out_path = os.path.join(OUT_DIR, f"{row['uuid4']}.npy")
    if os.path.exists(out_path):
        return None, None
        
    start_val = row.get("start_sample", 0)
    start_sample = int(start_val) if start_val else 0
    data, _ = sf.read(row["filepath"], start=start_sample, frames=131072, dtype="float32", always_2d=True)
    return row['uuid4'], data.mean(axis=1)

if __name__ == "__main__":
    # 1. Load model to GPU
    model = AutoencoderModel.from_pretrained(MODEL_NAME)
    model.autoencoder = model.autoencoder.to(DEVICE)
    model.device = DEVICE
    model.autoencoder.eval()
    print(f"Model loaded and pushed to {DEVICE}")

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    # 2. Filter out clips we've already done and ignore blank CSV lines
    todo = [r for r in rows if r.get("uuid4") and not os.path.exists(os.path.join(OUT_DIR, f"{r['uuid4']}.npy"))]
    print(f"Latents left to extract: {len(todo)}")

    # 3. Process in batches using multi-threading for disk I/O
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Extracting Latents"):
            batch_rows = todo[i : i + BATCH_SIZE]
            
            # Parallel disk load
            results = list(pool.map(load_audio, batch_rows))
            
            uuids, audios = [], []
            for uuid, audio in results:
                if uuid is not None:
                    uuids.append(uuid)
                    audios.append(audio)
                    
            if not audios:
                continue
                
            # Move batch to GPU and add channel dimension: (Batch, Channels, Time)
            audio_batch = torch.tensor(np.stack(audios)).unsqueeze(1).to(DEVICE)
            
            with torch.no_grad():
                latents = model.encode(audio_batch, SR)
                
            latents_np = latents.cpu().numpy().astype("float32")
            
            # Save to disk
            for idx, uuid in enumerate(uuids):
                np.save(os.path.join(OUT_DIR, f"{uuid}.npy"), latents_np[idx])

