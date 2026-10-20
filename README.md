# Slakh18 Timbre Transfer — Universal Deployment

Hi everyone, this is a current work in progress for my Masters Thesis at CalArts. There is still optimization and bugs present.

The project is a standalone, cross-platform application for neural musical instrument timbre transfer powered by a Latent Diffusion Transformer (DiT) conditioned on pitch contours and musical instrument classes.

---

## Features

- **18 Musical Instrument Classes**: Transfer arbitrary melodic audio to acoustic piano, violin, cello, electric guitar, alto sax, clarinet, flute, brass, organ, and more.
- **Intelligent Register Alignment**: Automatically transposes input pitch contours by integer octaves into the target instrument's natural playable register while proportionally scaling spectral centroids.
- **Universal Hardware Support**:
  - **Windows & Linux**: Fully accelerated with NVIDIA CUDA (Flash / SDPA support).
  - **macOS**: Native acceleration via Apple Silicon Metal Performance Shaders (MPS).
  - **CPU Fallback**: Graceful fallback if no dedicated accelerator is present.
- **Self-Contained Cache**: Hugging Face autoencoder assets are isolated within the deployment directory (`hf_cache/`), avoiding external drive path dependencies.
- **Interactive Web Interface**: Clean Gradio GUI with real-time waveform inspection and audio playback.

---

## Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12` installed. *(PyTorch does not fully support Python 3.13+ yet).*
- **Git**: Installed and accessible in your system terminal / Command Prompt.
- **Hardware**:
  - **PC / Linux**: An NVIDIA GPU with CUDA drivers (CUDA 12.1+ recommended), or 8GB+ RAM for CPU.
  - **Mac**: Apple Silicon (M1/M2/M3/M4) recommended; Intel Macs supported.

---

## Installation

Choose the setup instructions for your operating system:

### Windows (PC with NVIDIA GPU)
1. Open Command Prompt or PowerShell in this folder:
   ```cmd
   cd path\to\universal_deployable
   ```
2. Run the Windows installation script:
   ```cmd
   install_pc.bat
   ```
3. Once installation completes, activate the environment:
   ```cmd
   call .venv\Scripts\activate.bat
   ```

### macOS (Apple Silicon & Intel)
1. Open Terminal and navigate to this folder:
   ```bash
   cd /path/to/universal_deployable
   ```
2. Run the Mac installation script:
   ```bash
   bash install_mac.sh
   ```
3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

### Linux (NVIDIA GPU / Cloud Server)
1. Open Terminal in this folder:
   ```bash
   cd /path/to/universal_deployable
   ```
2. Run the Linux installation script:
   ```bash
   bash install_linux.sh
   ```
3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

---

## Model Weights

The installer scripts (`install_pc.bat`, `install_mac.sh`, `install_linux.sh`) and the application GUI automatically check for and download the required DiT model weights from Hugging Face if they are not already present:

- **Hugging Face Model Repository**: [carnol16/timbral-transfer](https://huggingface.co/carnol16/timbral-transfer)
- **Model Checkpoint**: `diT_best_val.pt` (~690 MB)

```text
checkpoints/
└── diT_best_val.pt
```

*(If you ever need to download it manually or offline, download `diT_best_val.pt` directly from [https://huggingface.co/carnol16/timbral-transfer](https://huggingface.co/carnol16/timbral-transfer) and place it into the `checkpoints/` directory).*

---

## Running the Application

With your virtual environment (`.venv`) activated:

```bash
python run_app.py
```

Options:
- `--port 7860` : Run on a custom port.
- `--share`     : Create a public temporary Gradio link for remote testing.

Once launched, open the local URL in your web browser (typically `http://127.0.0.1:7860`).

---

## How to Use the GUI

1. **Upload Audio**: Drag and drop any clean, monophonic or solo audio file (vocals, whistling, violin, flute, guitar, etc.) or choose one of the samples from `external_test_audio/`.
2. **Select Target Instrument**: Choose from the 18 Slakh instrument classes.
3. **Configure Settings**:
   - **DDIM Steps**: `50`–`100` steps for optimal quality vs. speed.
   - **CFG Scale**: `3.0`–`5.0` controls how strictly the model adheres to the pitch contour conditioning.
4. **Generate**: Click **Transfer Timbre** and listen to the result!

---

## Directory Structure

```text
universal_deployable/
├── artifacts/
│   └── norm_stats.npz          # Latent & contour normalization statistics
├── checkpoints/
│   └── diT_best_val.pt         # Slakh18 114M DiT checkpoint
├── external_test_audio/        # Ready-to-use sample clips for testing
├── hf_cache/                   # Self-contained Hugging Face cache
├── models/
│   └── dit_modified.py         # Modified Diffusion Transformer definition
├── misc_tools/
│   └── gui/
│       ├── app.py              # Gradio web interface definition
│       └── inference.py        # Inference engine and device management
├── run_app.py                  # Main application entry point
├── encode_latent.py            # Latent encoding script (sa_venv)
├── decode_latent.py            # Latent decoding script (sa_venv)
├── stable-audio-3/             # Stability AI Stable Audio repo (auto-cloned during install)
├── install_pc.bat              # Windows setup script
├── install_mac.sh              # macOS setup script
├── install_linux.sh            # Linux setup script
├── requirements_pc.txt         # Windows / Linux CUDA dependencies
└── requirements_mac.txt        # macOS MPS dependencies
```

---

## Acknowledgements & Licenses

- **Dataset**: Trained on the [Slakh2100](https://www.slakh.com/) dataset under Creative Commons Attribution 4.0 (CC BY 4.0).
- **Autoencoder**: Powered by [Stability AI Stable Audio](https://github.com/Stability-AI/stable-audio-3) (SAME autoencoder, MIT License).
- **Pitch Tracking**: Pitch conditioning extracted using [TorchCREPE](https://github.com/maxrmorrison/torchcrepe) (MIT License).
- **Interface**: Built with [Gradio](https://github.com/gradio-app/gradio) (Apache 2.0).
