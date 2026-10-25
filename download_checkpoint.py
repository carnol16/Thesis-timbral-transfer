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

LORA_DIR = os.path.join(CKPT_DIR, "lora")
LORA_FILES = [
    "violin_song121_r8_best.pt",
    "flute_song56_r8_best.pt",
    "tenor_saxophone_song85_r8_best.pt",
    "trumpet_song94_r8_best.pt",
    "clarinet_song201_r8_best.pt",
]

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LORA_DIR, exist_ok=True)

    # 1. Download Base Checkpoint
    if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 100_000_000:
        print(f"[OK] Base model checkpoint already exists at: {CKPT_PATH}")
    else:
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
            else:
                raise RuntimeError("Download completed but base checkpoint is missing.")
        except Exception as e:
            print(f"\n[WARNING] Base checkpoint download failed: {e}")
            print(f"Please manually download '{FILENAME}' from https://huggingface.co/{REPO_ID}")

    # 2. Download Curated Studio LoRA Adapters
    print(f"\nChecking Studio LoRA adapters in {LORA_DIR}...")
    from huggingface_hub import hf_hub_download
    for lora_file in LORA_FILES:
        target_path = os.path.join(LORA_DIR, lora_file)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 100_000:
            print(f"  [OK] LoRA adapter present: {lora_file}")
            continue

        print(f"  Downloading LoRA {lora_file} (~2.3 MB) from Hugging Face...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                subfolder="lora",
                filename=lora_file,
                local_dir=CKPT_DIR,
            )
            cache_dir = os.path.join(CKPT_DIR, ".cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"  [OK] Successfully downloaded {lora_file}")
        except Exception as e:
            print(f"  [WARNING] Could not download {lora_file}: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

