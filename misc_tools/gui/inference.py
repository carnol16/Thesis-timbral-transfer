"""
gui/inference.py  –  All GPU logic for timbre transfer.
Model loads once at import time; run_timbre_transfer() is the public API.
"""
import os, sys, gc
import numpy as np
import torch
import torch.backends.cuda
import librosa
import subprocess
import torchaudio
import soundfile as sf

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
    3:  ("Alto Saxophone", "checkpoints/lora/tenor_saxophone_song85_r8_best.pt"),
    16: ("Solo Trumpet (Medley Soloist)", "checkpoints/lora/trumpet_song94_r8_best.pt"),
    7:  ("Clarinet (Medley Soloist)", "checkpoints/lora/clarinet_song201_r8_best.pt"),
}

# Empirical Sweet Spots per Instrument (from 3D Sweep across CFG, LoRA, and Inst Guidance)
INSTRUMENT_OPTIMAL_PRESETS = {
    17: {"cfg_scale": 3.5, "lora_strength": 1.00, "cfg_inst_scale": 1.8}, # Studio Violin (Wood resonance & smooth vibrato)
    7:  {"cfg_scale": 3.0, "lora_strength": 1.20, "cfg_inst_scale": 1.8}, # Solo Clarinet (Hollow cylindrical woodwind, odd harmonics)
    12: {"cfg_scale": 3.0, "lora_strength": 1.00, "cfg_inst_scale": 1.0}, # Concert Flute (Airy breath turbulence, pure sinusoidal)
    3:  {"cfg_scale": 4.5, "lora_strength": 0.80, "cfg_inst_scale": 1.8}, # Alto / Tenor Saxophone (Reed bite & tight harmonic lock)
    16: {"cfg_scale": 3.5, "lora_strength": 1.20, "cfg_inst_scale": 1.8}, # Solo Trumpet (Brassy harmonic drive & attack flare)
}

def get_curated_adapter_path(inst_id, version="v2"):
    """Returns curated adapter path for the requested version ('v2' or 'v1')."""
    if inst_id not in LORA_ADAPTER_MAP:
        return None, None
    name, rel_r8 = LORA_ADAPTER_MAP[inst_id]
    if version == "v1":
        return name, rel_r8
    base_stem = os.path.basename(rel_r8).split("_r8")[0]
    v2_rel = f"checkpoints/lora_norm_v2/{base_stem}_r32_best.pt"
    if os.path.exists(os.path.join(PROJECT_ROOT, v2_rel)):
        return name, v2_rel
    return name, rel_r8

# We use a separate venv for stable_audio to avoid library conflicts
SA_VENV_DIR = os.path.join(PROJECT_ROOT, "sa_venv")
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
elif torch.mps.is_available():
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
        print("[GUI] Loading Phase 5 Modified DiT (Slakh18 114M)...")
        self.dit_model = ModifiedDiT(
            contour_dim=6, d_model=768, depth=16, n_heads=12,
            use_rope=True, use_instrument_embed=True, num_instruments=18,
        ).to(DEVICE)

        ckpt = os.path.join(
            PROJECT_ROOT,
            "checkpoints/Slakh18/DiT_modern/ablation_both_multi_inst_large/diT_best_val.pt"
        )
        if not os.path.exists(ckpt):
            alt_ckpt = os.path.join(PROJECT_ROOT, "checkpoints/diT_best_val.pt")
            if os.path.exists(alt_ckpt):
                ckpt = alt_ckpt
        checkpoint = torch.load(ckpt, weights_only=False, map_location=DEVICE)
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
        self.dit_model.load_state_dict(state_dict)
        self.dit_model.eval()

        # Inject LoRA architecture
        print("[GUI] Initializing LoRA adapter architecture...")
        self.dit_model = inject_lora(self.dit_model, rank=8, alpha=16.0)
        self.dit_model = self.dit_model.to(DEVICE)
        self.current_lora_adapter = None
        self.current_rank = 8
        self.current_target_mods = ["qkv", "proj"]
        self.set_lora_adapter(None, strength=0.0)

        print("[GUI] Loading normalization stats...")
        self.update_norm_stats(use_clean=False)
        print("[GUI] DiT model ready with LoRA support! (Autoencoder runs in subprocess via sa_venv)")

    def update_norm_stats(self, use_clean=True):
        """Loads clean stats for LoRAs, or legacy stats for unadapted base model."""
        clean_p = os.path.join(PROJECT_ROOT, "artifacts/norm_stats.npz")
        legacy_p = os.path.join(PROJECT_ROOT, "artifacts/norm_stats_legacy_v1.npz")
        target_p = clean_p if (use_clean or not os.path.exists(legacy_p)) else legacy_p
        stats = np.load(target_p)
        self.latent_mean = stats["latent_mean"][:, None].astype("float32")   # (256,1)
        self.latent_std  = stats["latent_std"][:, None].astype("float32")
        self.contour_mean = stats["contour_mean"][:6, None].astype("float32") # (6,1)
        self.contour_std  = stats["contour_std"][:6, None].astype("float32")

    def set_lora_adapter(self, adapter_path=None, strength=1.0):
        """Dynamically load or scale LoRA adapter weights, auto-adapting rank & targets."""
        if adapter_path is not None and os.path.exists(adapter_path):
            if self.current_lora_adapter != adapter_path:
                print(f"[GUI] Loading LoRA adapter: {os.path.basename(adapter_path)}")
                adapter_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
                detected_rank = 8
                target_mods = ["qkv", "proj"]
                for k, v in adapter_dict.items():
                    if "lora_A" in k:
                        detected_rank = v.shape[0]
                    if any(w in k for w in ("w1", "w2", "w3", "fc1", "fc2")):
                        if "mlp" not in target_mods:
                            target_mods.append("mlp")
                
                curr_rank = getattr(self, "current_rank", 8)
                curr_mods = getattr(self, "current_target_mods", ["qkv", "proj"])
                if curr_rank != detected_rank or set(curr_mods) != set(target_mods):
                    print(f"[GUI] Re-injecting LoRA architecture for Rank={detected_rank}, Targets={target_mods}...")
                    self.dit_model = inject_lora(self.dit_model, rank=detected_rank, alpha=float(detected_rank * 2.0), target_modules=target_mods)
                    self.dit_model = self.dit_model.to(DEVICE)
                    self.current_rank = detected_rank
                    self.current_target_mods = target_mods
                
                load_lora_weights(self.dit_model, adapter_path, device=DEVICE)
                self.current_lora_adapter = adapter_path

            rank_val = getattr(self, "current_rank", 8)
            target_scale = (float(rank_val * 2.0) / float(rank_val)) * float(strength)
        else:
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
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:" + env.get("LD_LIBRARY_PATH", "")
        
        # Free GPU VRAM for the Autoencoder encoder subprocess
        if DEVICE == "cuda":
            self.dit_model.to("cpu")
            torch.cuda.empty_cache()

        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

        if DEVICE == "cuda":
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
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:" + env.get("LD_LIBRARY_PATH", "")
        
        # Free GPU VRAM for the Autoencoder decoder subprocess
        if DEVICE == "cuda":
            self.dit_model.to("cpu")
            torch.cuda.empty_cache()

        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

        if DEVICE == "cuda":
            self.dit_model.to(DEVICE)
        
        audio, _ = torchaudio.load(out_audio_path)
        # torchaudio loads as (channels, frames). Gradio expects (frames, channels)
        return audio.T.numpy().astype("float32")

    def _ddim_loop(self, cond_contour, instrument_id, cfg_scale, ddim_steps, progress_cb, cfg_inst_scale=1.8):
        B, seq_len, C = cond_contour.shape
        x = torch.randn(B, seq_len, 256, device=DEVICE)
        acp = ALPHAS_CUMPROD.to(DEVICE)
        timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=DEVICE)
        inst_t = torch.full((B,), instrument_id, dtype=torch.long, device=DEVICE)
        zero_contour = torch.zeros_like(cond_contour)

        with torch.no_grad():
            for i in range(ddim_steps):
                t_step = timesteps[i].item()
                t = torch.full((B,), t_step, device=DEVICE, dtype=torch.long)

                # Decoupled Two-Factor CFG:
                # 1. Full prediction: conditioned on both contour and instrument
                v_full = self.dit_model(x, t, cond_contour, instrument_id=inst_t)

                if cfg_scale > 1.0 or cfg_inst_scale > 1.0:
                    # 2. Contour-unconditioned prediction: conditioned on instrument only
                    v_inst = self.dit_model(x, t, zero_contour, instrument_id=inst_t)
                    delta_contour = v_full - v_inst

                    if cfg_inst_scale > 1.0:
                        # 3. Null prediction: unconditioned on both contour and instrument
                        v_null = self.dit_model(x, t, zero_contour, instrument_id=None)
                        delta_inst = v_inst - v_null
                        # Independently amplify instrument timbre and melody contour
                        v_pred = v_null + cfg_inst_scale * delta_inst + cfg_scale * delta_contour
                    else:
                        v_pred = v_inst + cfg_scale * delta_contour
                else:
                    v_pred = v_full

                a_bar = acp[t_step]
                prev_a_bar = torch.tensor(1.0, device=DEVICE) if i == ddim_steps - 1 else acp[timesteps[i+1].item()]
                x_0 = torch.sqrt(a_bar) * x - torch.sqrt(1.0 - a_bar) * v_pred
                eps = torch.sqrt(a_bar) * v_pred + torch.sqrt(1.0 - a_bar) * x
                x   = torch.sqrt(prev_a_bar) * x_0 + torch.sqrt(1.0 - prev_a_bar) * eps

                if progress_cb:
                    progress_cb(i + 1, ddim_steps)
        return x

    @staticmethod
    def _generate_window_weights(window_len, overlap_left, overlap_right):
        """Creates a smooth trapezoidal Hann blend window for seamless latent stitching."""
        w = np.ones(window_len, dtype=np.float32)
        if overlap_left > 0:
            ramp_left = 0.5 * (1.0 - np.cos(np.pi * np.arange(overlap_left) / overlap_left))
            w[:overlap_left] = ramp_left
        if overlap_right > 0:
            ramp_right = 0.5 * (1.0 - np.cos(np.pi * np.arange(overlap_right, 0, -1) / overlap_right))
            w[-overlap_right:] = ramp_right
        return w

    def run_timbre_transfer(self, audio_path, instrument_id=17, cfg_scale=None,
                            ddim_steps=50, octave_shift=0, use_lora=True,
                            lora_strength=None, lora_version="v2 High-Capacity (r=32)", custom_lora_path=None, progress_cb=None,
                            cfg_inst_scale=None, force_clean_norm=None, match_source_loudness=True):
        # Auto-apply instrument-specific optimal preset defaults if not explicitly passed
        opt = INSTRUMENT_OPTIMAL_PRESETS.get(instrument_id, {})
        if cfg_scale is None:
            cfg_scale = opt.get("cfg_scale", 3.5)
        if lora_strength is None:
            lora_strength = opt.get("lora_strength", 1.00)
        if cfg_inst_scale is None:
            cfg_inst_scale = opt.get("cfg_inst_scale", 1.8)

        print(f"[GUI] Loading: {audio_path}")

        # Configure LoRA Adapter and dynamic normalization routing
        is_lora_active = False
        use_clean_norm = False
        if custom_lora_path and os.path.exists(custom_lora_path):
            print(f"[GUI] Applying Custom Artist LoRA: {os.path.basename(custom_lora_path)} (Strength: {lora_strength:.2f})")
            self.set_lora_adapter(custom_lora_path, strength=lora_strength)
            is_lora_active = True
            use_clean_norm = False if force_clean_norm is None else force_clean_norm
        elif use_lora and instrument_id in LORA_ADAPTER_MAP:
            ver = "v1" if (lora_version and "v1" in str(lora_version).lower()) else "v2"
            preset_name, rel_path = get_curated_adapter_path(instrument_id, version=ver)
            adapter_full = os.path.join(PROJECT_ROOT, rel_path)
            use_clean_norm = False if force_clean_norm is None else force_clean_norm

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

            print(f"[GUI] Applying Studio LoRA: {preset_name} [{os.path.basename(rel_path)}] (Strength: {lora_strength:.2f})")
            self.set_lora_adapter(adapter_full, strength=lora_strength)
            is_lora_active = True
        else:
            print("[GUI] Running Pure Base Slakh Model (No LoRA)")
            self.set_lora_adapter(None, strength=0.0)
            is_lora_active = False

        # Route normalization stats: clean stats for v2 LoRA, legacy stats for v1 LoRA and base model
        self.update_norm_stats(use_clean=use_clean_norm if is_lora_active else False)

        audio_np = librosa.load(audio_path, sr=SR, mono=True)[0]

        # 1. Automatic Dynamic RMS Calibration (Prevents mastered / commercial loud loops from blowing out conditioning)
        raw_rms = float(np.sqrt(np.mean(audio_np ** 2)))
        calibrated = False
        calib_scale = 1.0
        if raw_rms > 0.04:
            calib_scale = 0.03 / max(raw_rms, 1e-4)
            audio_for_encoding = audio_np * calib_scale
            calibrated = True
            temp_calib_path = os.path.join(PROJECT_ROOT, "artifacts", "temp_calibrated_in.wav")
            sf.write(temp_calib_path, audio_for_encoding, SR)
            encode_input_path = temp_calib_path
            print(f"[GUI] Mastered audio detected (RMS={raw_rms:.3f}). Calibrated to native model range (scale={calib_scale:.3f}).")
        else:
            audio_for_encoding = audio_np
            encode_input_path = audio_path

        print("[GUI] Encoding...")
        latent_norm = self._encode(encode_input_path)

        print("[GUI] Extracting contours...")
        contour = extract_contours(audio_for_encoding, target_len=latent_norm.shape[1])  # (9, seq_len)
        contour_6 = contour[:6, :].copy()
        
        # 2. REGISTER ALIGNMENT (Transpose into Target Instrument Range preserving musical key)
        shift_octaves = 0
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
                contour_6[0, voiced_mask] += contour_shift
                print(f"[GUI] Transposed melody by {shift_octaves:+d} octave(s) to fit {INSTRUMENT_NAMES.get(instrument_id, '')}.")
            else:
                print(f"[GUI] Preserved original pitch (0 octave shift). Within {INSTRUMENT_NAMES.get(instrument_id, '')} playable range.")
                
        # 3. Standard Dataset Normalization
        contour_norm = (contour_6 - self.contour_mean) / self.contour_std
        total_frames = latent_norm.shape[1]

        # 4. Adaptive Windowing vs Single-Pass Generation
        if total_frames <= 40:
            # Short audio (<= ~3.7s): Standard single-pass (fastest, zero crossfade seams)
            cond_contour = (
                torch.from_numpy(contour_norm.astype("float32"))
                .unsqueeze(0).transpose(1, 2).to(DEVICE)    # (1, seq_len, 6)
            )
            print(f"[GUI] Single-pass DDIM {ddim_steps} steps, Contour-CFG={cfg_scale}, Inst-CFG={cfg_inst_scale}...")
            latent_out = self._ddim_loop(cond_contour, instrument_id, cfg_scale, ddim_steps, progress_cb, cfg_inst_scale=cfg_inst_scale)
        else:
            # Long audio (> ~3.7s): Dynamic onset-adaptive hybrid windowing
            dur_sec = len(audio_np) / SR
            onsets = librosa.onset.onset_detect(y=audio_for_encoding, sr=SR, hop_length=512)
            onset_density = len(onsets) / max(dur_sec, 1e-4)

            if onset_density > 3.0:
                window_size = 32
                overlap = 8
                print(f"[GUI] Adaptive Profile: Rhythmic / Fast Articulation ({onset_density:.1f} onsets/s -> W=32 frames, overlap=8)")
            else:
                window_size = 48
                overlap = 12
                print(f"[GUI] Adaptive Profile: Legato / Melodic Phrasing ({onset_density:.1f} onsets/s -> W=48 frames, overlap=12)")

            strides = window_size - overlap
            windows = []
            rms_curve = contour_norm[2]
            pos = 0
            while pos + window_size < total_frames:
                search_start = max(pos + overlap, pos + strides - 5)
                search_end = min(total_frames - (window_size - overlap), pos + strides + 5)
                if search_start < search_end:
                    best_cut = search_start + int(np.argmin(rms_curve[search_start:search_end]))
                else:
                    best_cut = pos + strides
                windows.append((pos, pos + window_size))
                pos = best_cut
            windows.append((max(0, total_frames - window_size), total_frames))

            num_windows = len(windows)
            print(f"[GUI] Sliced {total_frames} frames into {num_windows} hybrid windows: {windows}")

            cond_list = [torch.from_numpy(contour_norm[:, st:en].T.astype("float32")) for (st, en) in windows]
            cond_batch = torch.stack(cond_list, dim=0).to(DEVICE) # (num_windows, window_size, 6)

            print(f"[GUI] Running parallel batched DDIM {ddim_steps} steps across {num_windows} windows on GPU...")
            x_windows = self._ddim_loop(cond_batch, instrument_id, cfg_scale, ddim_steps, progress_cb, cfg_inst_scale=cfg_inst_scale)

            # Stitch latents with trapezoidal Hann blending
            latent_out = torch.zeros(1, total_frames, 256, device=DEVICE)
            weight_acc = torch.zeros(1, total_frames, 1, device=DEVICE)
            for idx, (st, en) in enumerate(windows):
                ov_l = 0 if idx == 0 else overlap
                ov_r = 0 if idx == num_windows - 1 else overlap
                w_np = self._generate_window_weights(window_size, ov_l, ov_r)
                w_t = torch.from_numpy(w_np).to(DEVICE).unsqueeze(0).unsqueeze(-1)
                latent_out[:, st:en, :] += x_windows[idx:idx+1] * w_t
                weight_acc[:, st:en, :] += w_t
            latent_out = latent_out / (weight_acc + 1e-8)

        print("[GUI] Decoding...")
        audio_out = self._decode(latent_out)

        # Source-matched Loudness & Headroom Management
        gen_rms = float(np.sqrt(np.mean(audio_out ** 2)))
        raw_src_rms = float(np.sqrt(np.mean(audio_np ** 2)))

        if match_source_loudness and gen_rms > 1e-5 and raw_src_rms > 1e-5:
            # Scale output to match the source audio's exact energy / perceived loudness
            rms_target_scale = raw_src_rms / gen_rms
            audio_out = audio_out * rms_target_scale
            print(f"[GUI] Matched output loudness to source (Source RMS={raw_src_rms:.4f}, Generated RMS={gen_rms:.4f}, scale={rms_target_scale:.2f}).")

        # Headroom safety limiter: ensure output never clips and preserves clean dynamics (-0.5 dBFS)
        max_peak = float(np.max(np.abs(audio_out)))
        if max_peak > 0.95:
            audio_out = audio_out * (0.95 / max_peak)
            print(f"[GUI] Headroom limiter applied (peak reduced from {max_peak:.2f} to 0.95).")

        # Truncate / pad to match original source audio duration
        target_samples = len(audio_np)
        if audio_out.shape[0] > target_samples:
            audio_out = audio_out[:target_samples, :]
        elif audio_out.shape[0] < target_samples:
            pad = np.zeros((target_samples - audio_out.shape[0], audio_out.shape[1]), dtype=audio_out.dtype)
            audio_out = np.concatenate([audio_out, pad], axis=0)

        gc.collect()
        torch.cuda.empty_cache()
        print("[GUI] Done!")
        return audio_out, SR, audio_np, contour_6


print("[GUI] Initialising inference engine (loading models)...")
engine = InferenceEngine()
