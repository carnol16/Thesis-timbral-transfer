"""
download_checkpoint.py - Download the v3 base model and v3 Studio LoRA adapters from Hugging Face.

Every file lives at the same relative path in the HF repo as under checkpoints/ locally, so
hf_hub_download(local_dir="checkpoints") places each one where the GUI expects it.
The GUI also downloads any missing file on first use, so running this is optional but avoids
a long pause the first time you click Generate.
"""
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REPO_ID = "carnol16/timbral-transfer"

# (path in HF repo == path under checkpoints/, minimum valid size in bytes, label)
FILES = [
    ("v3/v3_final_ep57.pt", 100_000_000, "v3 base model (~690 MB)"),
] + [
    (f"lora_v3/{inst}_r8_all/adapter_best.pt", 100_000, f"{inst} Studio LoRA (~2.4 MB)")
    for inst in ("violin", "flute", "clarinet", "trumpet", "tenor_saxophone")
]


def main():
    from huggingface_hub import hf_hub_download
    os.makedirs(CKPT_DIR, exist_ok=True)
    failed = []

    for repo_path, min_size, label in FILES:
        target = os.path.join(CKPT_DIR, *repo_path.split("/"))
        if os.path.exists(target) and os.path.getsize(target) > min_size:
            print(f"  [OK] {label} present: checkpoints/{repo_path}")
            continue
        print(f"  Downloading {label} from Hugging Face ({REPO_ID}/{repo_path})...")
        try:
            hf_hub_download(repo_id=REPO_ID, filename=repo_path, local_dir=CKPT_DIR)
            if not (os.path.exists(target) and os.path.getsize(target) > min_size):
                raise RuntimeError("download finished but the file is missing or too small")
            print(f"  [OK] Downloaded checkpoints/{repo_path}")
        except Exception as e:
            print(f"  [WARNING] Could not download {repo_path}: {e}")
            failed.append(repo_path)

    cache_dir = os.path.join(CKPT_DIR, ".cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    if failed:
        print(f"\n[WARNING] {len(failed)} file(s) failed. Download them manually from "
              f"https://huggingface.co/{REPO_ID} into checkpoints/ (same folder layout):")
        for f in failed:
            print(f"  - {f}")
        print("(The GUI will also retry missing files when it starts.)")
        return 0   # don't abort the installers; the GUI downloads on demand
    print("\n[OK] All v3 checkpoints are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
