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
from models.lora import inject_lora, load_lora_weights, LoRALinear
from data.contour import extract_contours

# Curated Factory LoRA Adapters (Instrument ID -> Name, Relative Path)
LORA_ADAPTER_MAP = {
    17: ("Studio Violin (Medley Soloist)", "checkpoints/lora/violin_song121_r8_best.pt"),
    12: ("Concert Flute (Medley Soloist)", "checkpoints/lora/flute_song56_r8_best.pt"),
    3:  ("Tenor Saxophone (Medley Soloist)", "checkpoints/lora/tenor_saxophone_song85_r8_best.pt"),
    16: ("Solo Trumpet (Medley Soloist)", "checkpoints/lora/trumpet_song94_r8_best.pt"),
    7:  ("Clarinet (Medley Soloist)", "checkpoints/lora/clarinet_song201_r8_best.pt"),
}

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

# Standard acoustic playable ranges for each instrument (min_log_hz, max_log_hz)
# Based on fundamental frequency limits: (lowest note, highest note)
INSTRUMENT_PITCH_RANGES = {
    0: (4.47, 7.47),   # Accordion: F2 (87 Hz) - A6 (1760 Hz)
    1: (3.31, 8.34),   # Acoustic Grand Piano: A0 (27.5 Hz) - C8 (4186 Hz)
    2: (4.41, 6.95),   # Acoustic Guitar (steel): E2 (82.4 Hz) - C6 (1046.5 Hz)
    3: (4.93, 6.72),   # Alto Sax: Db3 (138.6 Hz) - Ab5 (830.6 Hz)
    4: (4.36, 6.95),   # Brass Section: Eb2 (77.8 Hz) - C6 (1046.5 Hz)
    5: (4.18, 6.95),   # Cello: C2 (65.4 Hz) - C6 (1046.5 Hz)
    6: (4.70, 6.95),   # Choir Aahs: A2 (110 Hz) - C6 (1046.5 Hz)
    7: (4.99, 7.53),   # Clarinet: D3 (146.8 Hz) - Bb6 (1864.7 Hz)
    8: (4.41, 7.18),   # Distortion Guitar: E2 (82.4 Hz) - E6 (1318.5 Hz)
    9: (3.71, 5.56),   # Electric Bass (finger): E1 (41.2 Hz) - C4 (261.6 Hz)
    10: (4.41, 7.18),  # Electric Guitar (clean): E2 (82.4 Hz) - E6 (1318.5 Hz)
    11: (3.71, 7.88),  # Electric Piano 1: E1 (41.2 Hz) - E7 (2637 Hz)
    12: (5.56, 7.76),  # Flute: C4 (261.6 Hz) - D7 (2349 Hz)
    13: (4.18, 7.65),  # Rock Organ: C2 (65.4 Hz) - C7 (2093 Hz)
    14: (3.71, 7.88),  # String Ensemble 1: E1 (41.2 Hz) - E7 (2637 Hz)
    15: (4.41, 6.55),  # Trombone: E2 (82.4 Hz) - F5 (698.5 Hz)
    16: (5.22, 6.95),  # Trumpet: F#3 (185 Hz) - C6 (1046.5 Hz)
    17: (5.28, 7.88),  # Violin: G3 (196 Hz) - E7 (2637 Hz)
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

        # Inject LoRA architecture
        print("[GUI] Initializing LoRA adapter architecture...")
        self.dit_model = inject_lora(self.dit_model, rank=8, alpha=16.0)
        self.dit_model = self.dit_model.to(DEVICE)
        self.current_lora_adapter = None
        self.set_lora_adapter(None, strength=0.0)

        print("[GUI] Loading normalization stats...")
        stats = np.load(os.path.join(PROJECT_ROOT, "artifacts/norm_stats.npz"))
        self.latent_mean = stats["latent_mean"][:, None].astype("float32")   # (256,1)
        self.latent_std  = stats["latent_std"][:, None].astype("float32")
        self.contour_mean = stats["contour_mean"][:6, None].astype("float32") # (6,1)
        self.contour_std  = stats["contour_std"][:6, None].astype("float32")
        print("[GUI] DiT model ready with LoRA support! (Autoencoder runs in subprocess via sa_venv)")

    def set_lora_adapter(self, adapter_path=None, strength=1.0):
        """Dynamically load or scale LoRA adapter weights."""
        if adapter_path is not None and os.path.exists(adapter_path):
            if self.current_lora_adapter != adapter_path:
                print(f"[GUI] Loading LoRA adapter: {os.path.basename(adapter_path)}")
                load_lora_weights(self.dit_model, adapter_path, device=DEVICE)
                self.current_lora_adapter = adapter_path
            # Scaling formula: (alpha / rank) * strength
            target_scale = (16.0 / 8.0) * float(strength)
        else:
            # 0.0 scale completely deactivates the adapter branch (pure base model)
            target_scale = 0.0

        for m in self.dit_model.modules():
            if isinstance(m, LoRALinear):
                m.scaling = target_scale

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
                            ddim_steps=50, octave_shift="Auto", use_lora=True,
                            lora_strength=1.0, custom_lora_path=None, progress_cb=None):
        print(f"[GUI] Loading: {audio_path}")

        # Configure LoRA Adapter
        if custom_lora_path and os.path.exists(custom_lora_path):
            print(f"[GUI] Applying Custom Artist LoRA: {os.path.basename(custom_lora_path)} (Strength: {lora_strength:.2f})")
            self.set_lora_adapter(custom_lora_path, strength=lora_strength)
        elif use_lora and instrument_id in LORA_ADAPTER_MAP:
            preset_name, rel_path = LORA_ADAPTER_MAP[instrument_id]
            adapter_full = os.path.join(PROJECT_ROOT, rel_path)
            
            # Auto-download on-the-fly from Hugging Face if not found locally
            if not os.path.exists(adapter_full):
                print(f"[GUI] LoRA preset '{preset_name}' not found locally. Auto-downloading from Hugging Face...")
                os.makedirs(os.path.dirname(adapter_full), exist_ok=True)
                try:
                    from huggingface_hub import hf_hub_download
                    filename = os.path.basename(rel_path)
                    hf_hub_download(
                        repo_id="carnol16/timbral-transfer",
                        subfolder="lora",
                        filename=filename,
                        local_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
                    )
                    cache_dir = os.path.join(PROJECT_ROOT, "checkpoints", ".cache")
                    if os.path.exists(cache_dir):
                        import shutil
                        shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"[GUI] Successfully downloaded {filename}!")
                except Exception as e:
                    print(f"[GUI] Warning: Failed to download LoRA from Hugging Face: {e}")

            print(f"[GUI] Applying Studio LoRA: {preset_name} (Strength: {lora_strength:.2f})")
            self.set_lora_adapter(adapter_full, strength=lora_strength)
        else:
            print("[GUI] Running Pure Base Slakh Model (No LoRA)")
            self.set_lora_adapter(None, strength=0.0)

        audio_np = librosa.load(audio_path, sr=SR, mono=True)[0]

        print("[GUI] Encoding...")
        latent_norm = self._encode(audio_path)

        print("[GUI] Extracting contours...")
        contour = extract_contours(audio_np, target_len=latent_norm.shape[1])  # (9, seq_len)
        contour_6 = contour[:6, :].copy()
        
        # 1. REGISTER ALIGNMENT (Transpose into Target Instrument Range preserving musical key)
        voiced_mask = contour_6[1, :] > 0.5
        if voiced_mask.any():
            voiced_f0 = contour_6[0, voiced_mask]
            # Use 5th and 95th percentiles to avoid pitch tracking noise / vocal fry artifacts
            f_low = float(np.percentile(voiced_f0, 5))
            f_high = float(np.percentile(voiced_f0, 95))

            if octave_shift == "Auto" or octave_shift is None:
                inst_min, inst_max = INSTRUMENT_PITCH_RANGES.get(instrument_id, (f_low, f_high))
                inst_name = INSTRUMENT_NAMES.get(instrument_id, f"Instrument {instrument_id}")

                print(f"[GUI] Input pitch range (5th-95th): {np.exp(f_low):.1f} Hz – {np.exp(f_high):.1f} Hz")
                print(f"[GUI] {inst_name} playable range: {np.exp(inst_min):.1f} Hz – {np.exp(inst_max):.1f} Hz")

                input_span = f_high - f_low
                inst_span = inst_max - inst_min

                if input_span <= inst_span:
                    if f_low < inst_min:
                        # Melody is too low: shift up minimum octaves needed to clear lower bound
                        octaves_needed = (inst_min - f_low) / np.log(2)
                        shift_octaves = int(np.ceil(octaves_needed))
                    elif f_high > inst_max:
                        # Melody is too high: shift down minimum octaves needed to clear upper bound
                        octaves_needed = (f_high - inst_max) / np.log(2)
                        shift_octaves = -int(np.ceil(octaves_needed))
                    else:
                        # Fully within range: 100% original pitch preserved!
                        shift_octaves = 0
                else:
                    # Input span exceeds instrument range: center phrase to balance both boundaries
                    phrase_mid = (f_low + f_high) / 2.0
                    inst_mid = (inst_min + inst_max) / 2.0
                    shift_octaves = int(round((inst_mid - phrase_mid) / np.log(2)))
            else:
                try:
                    shift_octaves = int(octave_shift)
                except (ValueError, TypeError):
                    shift_octaves = 0

            if shift_octaves != 0:
                contour_shift = shift_octaves * np.log(2)
                centroid_scale = 2.0 ** shift_octaves
                contour_6[0, voiced_mask] += contour_shift
                contour_6[5, voiced_mask] = np.clip(contour_6[5, voiced_mask] * centroid_scale, 0.0, SR / 2.0)
                print(f"[GUI] Transposed melody by {shift_octaves:+d} octave(s) (centroid scaled {centroid_scale:.2f}x) to fit {INSTRUMENT_NAMES.get(instrument_id, '')}.")
            else:
                print(f"[GUI] Preserved original pitch (0 octave shift). Within {INSTRUMENT_NAMES.get(instrument_id, '')} playable range.")
        
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
