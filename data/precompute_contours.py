"""
Extracts (4, 32) contors for every manifest clip, 1 npy per clip
in folder artifacts/contours/. Resumables (will skip files that exist)
Run in root enviroment
"""

import os,sys
# Restrict numpy internal threading so our 6 python workers don't thrash the CPU cache
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))   # make `contour` importable
from contour import load_mono, extract_contours

MANIFEST = "data/manifest.csv"
OUT_DIR = "artifacts/contours"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(MANIFEST)

from concurrent.futures import ThreadPoolExecutor, as_completed

LIMIT = None
if LIMIT is not None:
    df = df.head(LIMIT)
rows = list(df.itertuples())

def process_row(row):
    out_path = os.path.join(OUT_DIR, f"{row.uuid4}.npy")
    if os.path.exists(out_path):
        return
    audio = load_mono(row.filepath, start=int(getattr(row, "start_sample", 0)))
    contour = extract_contours(audio)
    np.save(out_path, contour.astype("float32"))

# Use 6 threads to parallelize disk IO and librosa, keeping the RTX 3080 constantly fed
WORKERS = 6

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [executor.submit(process_row, row) for row in rows]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting Contours"):
        pass
