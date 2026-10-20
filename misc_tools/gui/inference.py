"""
gui/inference.py  –  All GPU/MPS logic for timbre transfer.
Model loads once at import time; run_timbre_transfer() is the public API.
Fully compatible with Apple Silicon (MPS), Nvidia (CUDA), and CPU.
"""
import os, sys, gc
import numpy as np
import torch
import torch.backends.cuda
import librosa
import subprocess
import torchaudio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)   # all relative artifact paths resolve from here

from models.dit_modified import ContourRendererDiT as ModifiedDiT
from data.contour import extract_contours

# We use a separate venv for stable_audio to avoid library conflicts
SA_VENV_DIR = os.path.join(PROJECT_ROOT, "sa_venv")
if sys.platform in ("win32", "cygwin"):
    SA_PYTHON = os.path.join(SA_VENV_DIR, "Scripts", "python.exe")
else:
    SA_PYTHON = os.path.join(SA_VENV_DIR, "bin", "python")
if not os.path.exists(SA_PYTHON):
    print(f"[GUI] Warning: {SA_PYTHON} not found. Falling back to {sys.executable}.")
    SA_PYTHON = sys.executable

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SR = 44100

INSTRUMENT_NAMES = {
    0: "Accordion",
    1: "Acoustic Grand Piano",
    2: "Acoustic Guitar (steel)",
    3: "Alto Sax",
    4: "Brass Section",
    5: "Cello",
    6: "Choir Aahs",
    7: "Clarinet",
    8: "Distortion Guitar",
    9: "Electric Bass (finger)",
    10: "Electric Guitar (clean)",
    11: "Electric Piano 1",
    12: "Flute",
    13: "Rock Organ",
    14: "String Ensemble 1",
    15: "Trombone",
    16: "Trumpet",
    17: "Violin",
}

# Average playing registers for each instrument in Slakh18 (log-Hz of voiced frames)
INSTRUMENT_PITCH_MEANS = {
    0: 5.68,   # Accordion
    1: 5.66,   # Acoustic Grand Piano
    2: 5.43,   # Acoustic Guitar (steel)
    3: 5.88,   # Alto Sax
    4: 5.68,   # Brass Section
    5: 5.30,   # Cello
    6: 5.83,   # Choir Aahs
    7: 6.00,   # Clarinet
    8: 5.67,   # Distortion Guitar
    9: 4.53,   # Electric Bass (finger)
    10: 5.61,  # Electric Guitar (clean)
    11: 5.41,  # Electric Piano 1
    12: 6.37,  # Flute
    13: 5.98,  # Rock Organ
    14: 5.99,  # String Ensemble 1
    15: 5.44,  # Trombone
    16: 6.19,  # Trumpet
    17: 6.01,  # Violin
}

# Build noise schedule once on CPU; move to DEVICE per-call
_T = 1000
_betas = torch.linspace(1e-4, 0.02, _T)
_alphas = 1.0 - _betas
ALPHAS_CUMPROD = torch.cumprod(_alphas, dim=0)


class InferenceEngine:
    def __init__(self):
        self._load_models()

    def _load_models(self):
        print(f"[GUI] Running on device: {DEVICE}")
        print("[GUI] Loading Phase 5 Modified DiT (Slakh18 114M)...")
        self.dit_model = ModifiedDiT(
            contour_dim=6, d_model=768, depth=16, n_heads=12,
            use_rope=True, use_instrument_embed=True, num_instruments=18,
        ).to(DEVICE)

        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "diT_best_val.pt")
        if not os.path.exists(ckpt):
            print(f"[GUI] Checkpoint not found at {ckpt}.")
            print("[GUI] Downloading diT_best_val.pt (~690 MB) from Hugging Face: carnol16/timbral-transfer...")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                import shutil
                hf_hub_download(
                    repo_id="carnol16/timbral-transfer",
                    filename="diT_best_val.pt",
                    local_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
                )
                cache_dir = os.path.join(PROJECT_ROOT, "checkpoints", ".cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                print(f"[GUI] Checkpoint downloaded successfully to {ckpt}")
            except Exception as e:
                raise RuntimeError(
                    f"Model checkpoint not found at {ckpt} and automatic download failed: {e}\n"
                    "Please download diT_best_val.pt manually from https://huggingface.co/carnol16/timbral-transfer "
                    "and place it into the 'checkpoints/' directory."
                ) from e
        
        print(f"[GUI] Loading checkpoint from: {ckpt}")
        checkpoint = torch.load(ckpt, weights_only=False, map_location=DEVICE)
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
        self.dit_model.load_state_dict(state_dict)
        self.dit_model.eval()

        print("[GUI] Loading normalization stats...")
        stats = np.load(os.path.join(PROJECT_ROOT, "artifacts/norm_stats.npz"))
        self.latent_mean = stats["latent_mean"][:, None].astype("float32")   # (256,1)
        self.latent_std  = stats["latent_std"][:, None].astype("float32")
        self.contour_mean = stats["contour_mean"][:6, None].astype("float32") # (6,1)
        self.contour_std  = stats["contour_std"][:6, None].astype("float32")
        print("[GUI] DiT model ready! (Autoencoder runs in subprocess via sa_venv)")

    def _encode(self, audio_path):
        latent_path = os.path.join(PROJECT_ROOT, "artifacts", "temp_custom_latent.npy")
        print(f"[GUI] Running encode_latent.py in subprocess...")
        cmd = [SA_PYTHON, "encode_latent.py", "--audio", audio_path, "--out", latent_path]
        env = os.environ.copy()
        env["TORCH_COMPILE_DISABLE"] = "1"
        hf_cache_dir = os.path.join(PROJECT_ROOT, "hf_cache")
        os.makedirs(hf_cache_dir, exist_ok=True)
        env["HF_HOME"] = hf_cache_dir
        env["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
        
        cuda_lib_path = os.path.join(PROJECT_ROOT, ".venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib")
        if os.path.exists(cuda_lib_path):
            env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:" + env.get("LD_LIBRARY_PATH", "")
        
        # Free GPU/MPS memory for the Autoencoder encoder subprocess
        if DEVICE in ["cuda", "mps"]:
            self.dit_model.to("cpu")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

        if DEVICE in ["cuda", "mps"]:
            self.dit_model.to(DEVICE)
        
        lat_np = np.load(latent_path)                    # (seq_len, 256)
        lat_norm = (lat_np.T - self.latent_mean) / self.latent_std   # (256, seq_len)
        lat_norm = lat_norm.T                            # (seq_len, 256)
        return torch.from_numpy(lat_norm).float().unsqueeze(0).to(DEVICE)  # (1,seq_len,256)

    def _decode(self, latent_norm):
        mean_t = torch.from_numpy(self.latent_mean).to(DEVICE)
        std_t  = torch.from_numpy(self.latent_std).to(DEVICE)
        x_t = latent_norm.transpose(1, 2)               # (1, 256, seq_len)
        x_unnorm = x_t * std_t + mean_t
        
        in_latent_path = os.path.join(PROJECT_ROOT, "artifacts", "temp_generated_latent.npy")
        out_audio_path = os.path.join(PROJECT_ROOT, "artifacts", "temp_generated_audio.wav")
        
        np.save(in_latent_path, x_unnorm.cpu().numpy())
        
        print(f"[GUI] Running decode_latent.py in subprocess...")
        cmd = [SA_PYTHON, "decode_latent.py", "--latent", in_latent_path, "--out", out_audio_path]
        env = os.environ.copy()
        env["TORCH_COMPILE_DISABLE"] = "1"
        hf_cache_dir = os.path.join(PROJECT_ROOT, "hf_cache")
        os.makedirs(hf_cache_dir, exist_ok=True)
        env["HF_HOME"] = hf_cache_dir
        env["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
        
        cuda_lib_path = os.path.join(PROJECT_ROOT, ".venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib")
        if os.path.exists(cuda_lib_path):
            env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:" + env.get("LD_LIBRARY_PATH", "")
        
        # Free GPU/MPS memory for the Autoencoder decoder subprocess
        if DEVICE in ["cuda", "mps"]:
            self.dit_model.to("cpu")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

        if DEVICE in ["cuda", "mps"]:
            self.dit_model.to(DEVICE)
        
        import soundfile as sf
        audio, _ = sf.read(out_audio_path, dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, None]
        return audio

    def _ddim_loop(self, cond_contour, instrument_id, cfg_scale, ddim_steps, progress_cb):
        seq_len = cond_contour.shape[1]
        x = torch.randn(1, seq_len, 256, device=DEVICE)
        acp = ALPHAS_CUMPROD.to(DEVICE)
        timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=DEVICE)
        inst_t = torch.tensor([instrument_id], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            for i in range(ddim_steps):
                t_step = timesteps[i].item()
                t = torch.tensor([t_step], device=DEVICE)
                v_cond   = self.dit_model(x, t, cond_contour, instrument_id=inst_t)
                # True CFG: Unconditioned drops BOTH contour (zeros) AND instrument (None)
                v_uncond = self.dit_model(x, t, torch.zeros_like(cond_contour), instrument_id=None)
                v_pred   = v_uncond + cfg_scale * (v_cond - v_uncond)

                a_bar = acp[t_step]
                prev_a_bar = torch.tensor(1.0, device=DEVICE) if i == ddim_steps - 1 else acp[timesteps[i+1].item()]
                x_0 = torch.sqrt(a_bar) * x - torch.sqrt(1.0 - a_bar) * v_pred
                eps = torch.sqrt(a_bar) * v_pred + torch.sqrt(1.0 - a_bar) * x
                x   = torch.sqrt(prev_a_bar) * x_0 + torch.sqrt(1.0 - prev_a_bar) * eps

                if progress_cb:
                    progress_cb(i + 1, ddim_steps)
        return x

    def run_timbre_transfer(self, audio_path, instrument_id=17, cfg_scale=3.0,
                            ddim_steps=50, octave_shift="Auto", progress_cb=None):
        print(f"[GUI] Loading: {audio_path}")
        audio_np = librosa.load(audio_path, sr=SR, mono=True)[0]

        print("[GUI] Encoding...")
        latent_norm = self._encode(audio_path)

        print("[GUI] Extracting contours...")
        contour = extract_contours(audio_np, target_len=latent_norm.shape[1])  # (9, seq_len)
        contour_6 = contour[:6, :].copy()
        
        # 1. REGISTER ALIGNMENT (Transpose into Target Instrument Range preserving musical key)
        voiced_mask = contour_6[1, :] > 0.5
        if voiced_mask.any():
            if octave_shift == "Auto" or octave_shift is None:
                target_mean = INSTRUMENT_PITCH_MEANS.get(instrument_id, contour_6[0, voiced_mask].mean())
                custom_mean = contour_6[0, voiced_mask].mean()
                octaves = (target_mean - custom_mean) / np.log(2)
                print("target: ", target_mean)
                print("Custom/Input: ", custom_mean)
                print("Octave: ", octaves)
            else:
                try:
                    octaves = int(octave_shift)
                except (ValueError, TypeError):
                    octaves = 0
            
            shift_octaves = int(round(octaves))
            contour_shift = shift_octaves * np.log(2)
            centroid_scale = 2.0 ** shift_octaves
            print("Contour Shift: ", contour_shift)
            print("Centroid Scale: ", centroid_scale)
            contour_6[0, voiced_mask] += contour_shift
            contour_6[5, voiced_mask] = np.clip(contour_6[5, voiced_mask] * centroid_scale, 0.0, SR / 2.0)
            print(f"[GUI] Transposed melody by {shift_octaves:+d} octave(s) (centroid scaled by {centroid_scale:.2f}x). Musical note names are preserved.")
        
        # 2. Standard Dataset Normalization
        contour_norm = (contour_6 - self.contour_mean) / self.contour_std

        cond_contour = (
            torch.from_numpy(contour_norm.astype("float32"))
            .unsqueeze(0).transpose(1, 2).to(DEVICE)    # (1, seq_len, 6)
        )

        print(f"[GUI] DDIM {ddim_steps} steps, CFG={cfg_scale}...")
        latent_out = self._ddim_loop(cond_contour, instrument_id, cfg_scale, ddim_steps, progress_cb)

        print("[GUI] Decoding...")
        audio_out = self._decode(latent_out)

        # Truncate or pad decoded output to match original source length exactly
        target_samples = len(audio_np)
        if audio_out.shape[0] > target_samples:
            audio_out = audio_out[:target_samples, :]
        elif audio_out.shape[0] < target_samples:
            pad = np.zeros((target_samples - audio_out.shape[0], audio_out.shape[1]), dtype=audio_out.dtype)
            audio_out = np.concatenate([audio_out, pad], axis=0)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        print("[GUI] Done!")
        return audio_out, SR, audio_np, contour_6


print("[GUI] Initialising inference engine (loading models)...")
engine = InferenceEngine()
