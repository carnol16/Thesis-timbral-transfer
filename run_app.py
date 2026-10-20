"""
run_app.py  –  Launch the Timbre Transfer Gradio app.

Usage (from project root, venv active):
    python run_app.py
    python run_app.py --share      # public link for remote demos
    python run_app.py --port 8080  # custom port
"""
import argparse, os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from misc_tools.gui.app import demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
