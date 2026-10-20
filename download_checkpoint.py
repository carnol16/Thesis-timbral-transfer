"""
download_checkpoint.py - Helper to check and download diT_best_val.pt from Hugging Face.
"""
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "diT_best_val.pt")
REPO_ID = "carnol16/timbral-transfer"
FILENAME = "diT_best_val.pt"

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 100_000_000:
        print(f"[OK] Checkpoint already exists at: {CKPT_PATH}")
        return 0

    print(f"Downloading {FILENAME} (~690 MB) from Hugging Face ({REPO_ID})...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=CKPT_DIR,
        )
        cache_dir = os.path.join(CKPT_DIR, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 100_000_000:
            print(f"[OK] Checkpoint downloaded successfully to: {CKPT_PATH}")
            return 0
        else:
            raise RuntimeError("Download completed but file is missing or incomplete.")
    except Exception as e:
        print(f"\n[WARNING] Automatic download failed: {e}")
        print("Please manually download 'diT_best_val.pt' from:")
        print(f"  https://huggingface.co/{REPO_ID}")
        print(f"and place it directly into: {CKPT_DIR}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

