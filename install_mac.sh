#!/bin/bash
set -e

echo "=========================================="
echo " Starting Mac Timbre Transfer Setup...    "
echo "=========================================="

# Find a compatible python version
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Could not find Python 3.10, 3.11, or 3.12 installed on this Mac."
    echo "PyTorch does not support Python 3.13/3.14 yet. Please install Python 3.12 from python.org."
    exit 1
fi
echo "Using Python executable: $PYTHON_CMD"

echo "1/4: Setting up the main Virtual Environment (.venv)..."
$PYTHON_CMD -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_mac.txt
deactivate

echo "2/4: Setting up the Stable Audio Venv (sa_venv)..."
$PYTHON_CMD -m venv sa_venv
source sa_venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio torchcodec
pip install numpy scipy soundfile librosa einops einops-exts safetensors transformers huggingface-hub tqdm
deactivate

echo "3/4: Setting up stable-audio-3 for the Autoencoder..."
if [ ! -d "stable-audio-3" ]; then
    echo "Downloading stable-audio-3 repository into root folder..."
    git clone https://github.com/Stability-AI/stable-audio-3.git || {
        echo "Git clone failed. Trying Python fallback..."
        python -c "import urllib.request, zipfile, os; urllib.request.urlretrieve('https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip', 'sa3.zip'); zipfile.ZipFile('sa3.zip').extractall('.'); os.rename('stable-audio-3-main', 'stable-audio-3'); os.remove('sa3.zip')"
    }
fi

if [ -d "stable-audio-3" ]; then
    source sa_venv/bin/activate
    pip install --no-deps -e stable-audio-3
    deactivate
else
    echo "Error: Failed to find or download stable-audio-3."
    exit 1
fi

echo "4/4: Checking and downloading model checkpoint (diT_best_val.pt)..."
source .venv/bin/activate
python download_checkpoint.py
deactivate

echo "=========================================="
echo " Setup Complete! You are ready to run!"
echo " "
echo " To start the GUI:"
echo "    source .venv/bin/activate"
echo "    python run_app.py"
echo "=========================================="
