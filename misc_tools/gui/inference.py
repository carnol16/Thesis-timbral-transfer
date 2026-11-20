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
from data.contour import extract_contours, fill_unvoiced
from data.pitch_screen_dataset import normalize_contour
from data.timbre_features import retarget_raw, apply_mode_normalized, TIMBRE_MODES

# Hugging Face repo hosting the weights. Each local path below mirrors its path in the repo under
# checkpoints/, so hf_hub_download(local_dir="checkpoints") lands files exactly where they are expected.
HF_REPO_ID = "carnol16/timbral-transfer"

# v3 base model: stores its own recipe + normalization stats in the checkpoint.
BASE_MODELS = {
    "v3 Pitch-Fixed (Nov 2026)": "checkpoints/v3/v3_final_ep57.pt",
}
DEFAULT_BASE = "v3 Pitch-Fixed (Nov 2026)"


def ensure_checkpoint(rel_path):
    """Return the absolute path of a checkpoint, downloading it from Hugging Face if missing.
    rel_path must start with 'checkpoints/'; the rest is its path in HF_REPO_ID."""
    full = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(full):
        return full
    repo_path = os.path.relpath(rel_path, "checkpoints").replace(os.sep, "/")
    print(f"[GUI] {rel_path} not found locally. Downloading from Hugging Face ({HF_REPO_ID}/{repo_path})...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=HF_REPO_ID, filename=repo_path,
                    local_dir=os.path.join(PROJECT_ROOT, "checkpoints"))
    cache_dir = os.path.join(PROJECT_ROOT, "checkpoints", ".cache")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"[GUI] Downloaded {os.path.basename(rel_path)}")
    return full
# v3 CFG default: cfg_inst_scale == cfg_scale reduces the two-factor CFG to plain
# v_null + s * (v_full - v_null), the setting the pitch harness validated (s = 3.0)
V3_DEFAULT_CFG = 3.0
# Timbre source (v3 only), see data/timbre_features.py and evaluation/timbre_source_eval.py.
# Default "instrument": the source's aperiodicity + centroid are dropped (the state channel dropout
# trained), so the chosen instrument decides the timbre. Chosen by LISTENING (Nov 11 2026: preferred in
# 4 of 4 picked cases, incl. sax -> Alto Sax that sounded string-like). The spectral metric favoured
# "retarget" (cross target rank 6.95 source / 6.04 drop / 4.60 retarget), but it partly measures the
# brightness that retarget sets directly, so the listening result wins.
DEFAULT_TIMBRE_MODE = "instrument"
# Window blending for inputs longer than one window (> 40 frames, ~3.7 s).
#   "synced"  : MultiDiffusion (Bar-Tal et al. 2023): one shared full-length latent; the overlapping windows
#               are averaged after EVERY DDIM step, so neighbours agree while they are being denoised.
#   "standard": every window gets its own noise and runs all steps alone; the finished latents are
#               overlap-added once at the end (the original GUI behaviour).
# Nov 12 2026 (evaluation/window_blend_test.py, 47 same-instrument + 67 cross-instrument renders):
# standard leaves the overlap zones ~1.5 dB quieter and duller (latent norm 0.90 of the window interiors)
# and lets the target instrument drift more between windows; synced removes both at the same speed, pitch
# unchanged. Blind listening: cross-instrument 7-3 for synced, same-instrument 5-1 for standard (neither
# significant). Synced is the default because cross-instrument transfer is the main use case.
BLEND_MODES = ("synced", "standard")
DEFAULT_BLEND_MODE = "synced"
# Per-instrument v3 presets (contour CFG, instrument CFG).
# Nov 10 2026: first sweep (cfg_preset_sweep.py) + A/B/C listening, under the old defaults
#   (timbre source "keep source", no LoRA): clarinet 2/2 won by ear; alto sax, flute, trumpet kept 3/3.
# Nov 11 2026: re-sweep under the CURRENT defaults (timbre source "target only", Studio LoRA on):
#   evaluation/cfg_sweep_v3_defaults.py. With aperiodicity/centroid dropped, the instrument label
#   carries the timbre, so a stronger instrument CFG now clearly helps (timbre distance to the Slakh
#   class, paired over 12 clips, better on 10-12/12 clips, pitch follow within -0.03).
#   Violin: 1.5/1.5 was its weakest pitch setting (follow 0.94, 1.7% octave errors) -> 3/3 (0.99,
#   same timbre). Flute, clarinet, trumpet: changes within noise, kept. Not yet confirmed by ear.
V3_PRESETS = {
    17: (3.0, 3.0),   # Violin           (was 1.5/1.5; follow 0.94 -> 0.99)
    7:  (2.0, 2.0),   # Clarinet         (kept, by ear)
    1:  (3.0, 4.5),   # Grand Piano      timbre -2.55 +- 0.43 (voicing 0.87: chordal)
    4:  (4.0, 6.0),   # Brass Section    timbre -2.43 +- 0.46
    11: (4.0, 6.0),   # Electric Piano   timbre -1.82 +- 0.34
    8:  (4.0, 6.0),   # Distortion Guit. timbre -1.34 +- 0.25
    2:  (4.0, 6.0),   # Acoustic Guitar  timbre -1.32 +- 0.29
    9:  (3.0, 4.5),   # Electric Bass    timbre -0.99 +- 0.30
    3:  (3.0, 4.5),   # Alto Sax (LoRA)  real-player timbre -0.77 +- 0.25
    6:  (3.0, 4.5),   # Choir Aahs       timbre -0.77 +- 0.29
}
# v3 Studio LoRA strength per instrument (default 1.0). Violin LoRA lost 2 blind cross-instrument
# A/Bs at 1.0; at 0.8 its class rank improves (2.6 -> 2.0) at the same pitch (re-sweep, Nov 11).
V3_LORA_STRENGTH = {17: 0.8}


# v3 Studio LoRAs (Nov 10 2026): rank 8, trained on ALL train-fold (0-2) Medley songs of each
# instrument with train_lora_v3.py; scored on fold 3 with evaluation/lora_eval_v3.py
# (timbre vs the real recording 11-21% closer, pitch unchanged within +-0.02).
V3_LORA_MAP = {
    17: ("Studio Violin (Medley, 24 players)", "checkpoints/lora_v3/violin_r8_all/adapter_best.pt"),
    12: ("Concert Flute (Medley, 17 players)", "checkpoints/lora_v3/flute_r8_all/adapter_best.pt"),
    7:  ("Clarinet (Medley, 13 players)", "checkpoints/lora_v3/clarinet_r8_all/adapter_best.pt"),
    16: ("Solo Trumpet (Medley, 24 players)", "checkpoints/lora_v3/trumpet_r8_all/adapter_best.pt"),
    3:  ("Tenor Sax on Alto Sax (Medley, 3 players)", "checkpoints/lora_v3/tenor_saxophone_r8_all/adapter_best.pt"),
}


def v3_preset(instrument_id):
    """(contour CFG, instrument CFG) for the v3 model."""
    return V3_PRESETS.get(instrument_id, (V3_DEFAULT_CFG, V3_DEFAULT_CFG))


def v3_lora_strength(instrument_id):
    return V3_LORA_STRENGTH.get(instrument_id, 1.0)

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
        self.base_name = None
        self.load_base(DEFAULT_BASE)

    def load_base(self, name):
        """Load a base model by its BASE_MODELS name. No-op if it is already loaded."""
        if name == self.base_name:
            return
        if name not in BASE_MODELS:
            raise ValueError(f"Unknown base model {name!r}; choose from {list(BASE_MODELS)}")
        path = ensure_checkpoint(BASE_MODELS[name])
        print(f"[GUI] Loading base model: {name} ({os.path.basename(path)})")
        if getattr(self, "dit_model", None) is not None:    # free the previous model first
            del self.dit_model
            gc.collect(); torch.cuda.empty_cache()
        ckpt = torch.load(path, weights_only=False, map_location="cpu")

        if not (isinstance(ckpt, dict) and "ema" in ckpt):
            raise ValueError(f"{path} is not a v3 checkpoint (no 'ema' key); this build only runs v3.")
        # v3: size, stats and contour recipe all live in the checkpoint
        mc = ckpt["model_config"]
        weights, contour_dim = ckpt["ema"], ckpt["contour_dim"]
        self.recipe = dict(kind="v3", channels=list(ckpt["channels"]), fill=bool(ckpt["fill_unvoiced"]),
                           fourier=bool(ckpt["fourier"]), periods=tuple(ckpt["fourier_periods"]))
        v3_stats = {k: ckpt[k] for k in ("latent_mean", "latent_std", "contour_mean", "contour_std")}

        model = ModifiedDiT(contour_dim=contour_dim, d_model=mc["d_model"], depth=mc["depth"],
                            n_heads=mc["n_heads"], use_rope=mc["use_rope"],
                            use_instrument_embed=mc["use_instrument_embed"],
                            num_instruments=mc["num_instruments"])
        model.load_state_dict(weights)
        self.dit_model = model.to(DEVICE).eval()
        del ckpt, weights
        gc.collect(); torch.cuda.empty_cache()

        # LoRA architecture (rank 8; set_lora_adapter re-injects if an adapter differs). Adapters: V3_LORA_MAP.
        self.current_lora_adapter = None
        print("[GUI] Initializing LoRA adapter architecture...")
        self.dit_model = inject_lora(self.dit_model, rank=8, alpha=16.0).to(DEVICE)
        self.current_rank = 8
        self.current_target_mods = ["qkv", "proj"]
        self.set_lora_adapter(None, strength=0.0)
        self.latent_mean = np.asarray(v3_stats["latent_mean"])[:, None].astype("float32")    # (256, 1)
        self.latent_std = np.asarray(v3_stats["latent_std"])[:, None].astype("float32")
        self.contour_mean = np.asarray(v3_stats["contour_mean"])[:, None].astype("float32")  # (9, 1): all raw channels
        self.contour_std = np.asarray(v3_stats["contour_std"])[:, None].astype("float32")
        self.base_name = name
        n_params = sum(p.numel() for p in self.dit_model.parameters()) / 1e6
        print(f"[GUI] Ready: {name} | {n_params:.1f}M params | recipe {self.recipe}")

    def normalize_for_model(self, contour_raw, timbre_mode="source", instrument_id=None):
        """Raw (9, T) contour from extract_contours -> normalized (C, T) model input:
        the exact v3 training transform (voiced stats, unvoiced fill, Fourier pitch) + its channels."""
        r = self.recipe
        if timbre_mode == "retarget" and instrument_id is not None:
            contour_raw = retarget_raw(contour_raw, instrument_id)
        full = normalize_contour(contour_raw, self.contour_mean, self.contour_std,
                                 r["fill"], r["fourier"], r["periods"])
        # aperiodicity + centroid sit at positions 4, 5 of v3's channel list [0..5, 9..16]
        return apply_mode_normalized(full[r["channels"]], timbre_mode)

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
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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

    def _guided_v(self, x, t, cond_contour, inst_t, cfg_scale, cfg_inst_scale):
        """Decoupled two-factor CFG velocity, shared by both samplers."""
        # 1. Full prediction: conditioned on both contour and instrument
        v_full = self.dit_model(x, t, cond_contour, instrument_id=inst_t)
        if cfg_scale > 1.0 or cfg_inst_scale > 1.0:
            zero_contour = torch.zeros_like(cond_contour)
            # 2. Contour-unconditioned prediction: conditioned on instrument only
            v_inst = self.dit_model(x, t, zero_contour, instrument_id=inst_t)
            delta_contour = v_full - v_inst
            if cfg_inst_scale > 1.0:
                # 3. Null prediction: unconditioned on both contour and instrument
                v_null = self.dit_model(x, t, zero_contour, instrument_id=None)
                delta_inst = v_inst - v_null
                # Independently amplify instrument timbre and melody contour
                return v_null + cfg_inst_scale * delta_inst + cfg_scale * delta_contour
            return v_inst + cfg_scale * delta_contour
        return v_full

    @staticmethod
    def _ddim_step(x, v_pred, a_bar, prev_a_bar):
        x_0 = torch.sqrt(a_bar) * x - torch.sqrt(1.0 - a_bar) * v_pred
        eps = torch.sqrt(a_bar) * v_pred + torch.sqrt(1.0 - a_bar) * x
        return torch.sqrt(prev_a_bar) * x_0 + torch.sqrt(1.0 - prev_a_bar) * eps

    def _ddim_loop(self, cond_contour, instrument_id, cfg_scale, ddim_steps, progress_cb, cfg_inst_scale=1.8):
        """Independent sampling of a batch of windows (also the single-pass path)."""
        B, seq_len, C = cond_contour.shape
        x = torch.randn(B, seq_len, 256, device=DEVICE)
        acp = ALPHAS_CUMPROD.to(DEVICE)
        timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=DEVICE)
        inst_t = torch.full((B,), instrument_id, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            for i in range(ddim_steps):
                t_step = timesteps[i].item()
                t = torch.full((B,), t_step, device=DEVICE, dtype=torch.long)
                v_pred = self._guided_v(x, t, cond_contour, inst_t, cfg_scale, cfg_inst_scale)
                prev_a_bar = torch.tensor(1.0, device=DEVICE) if i == ddim_steps - 1 else acp[timesteps[i+1].item()]
                x = self._ddim_step(x, v_pred, acp[t_step], prev_a_bar)
                if progress_cb:
                    progress_cb(i + 1, ddim_steps)
        return x

    def _ddim_loop_synced(self, cond_full, windows, weights, instrument_id, cfg_scale, ddim_steps, progress_cb,
                          cfg_inst_scale=1.8):
        """MultiDiffusion: one full-length latent; after every DDIM step the windows are written back with
        their Hann weights and averaged, so overlapping windows continue from the same values.
        cond_full (1, T, C); windows [(start, end)] of equal length; weights [np (W,)]. Returns (1, T, 256)."""
        T = cond_full.shape[1]
        nw = len(windows)
        cond_w = torch.cat([cond_full[:, st:en] for st, en in windows], dim=0)          # (nw, W, C)
        w_t = [torch.from_numpy(w).to(DEVICE)[None, :, None] for w in weights]
        acc = torch.zeros(1, T, 1, device=DEVICE)
        for (st, en), w in zip(windows, w_t):
            acc[:, st:en] += w
        acp = ALPHAS_CUMPROD.to(DEVICE)
        timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=DEVICE)
        inst_t = torch.full((nw,), instrument_id, dtype=torch.long, device=DEVICE)
        x_full = torch.randn(1, T, 256, device=DEVICE)                                  # ONE noise tensor

        with torch.no_grad():
            for i in range(ddim_steps):
                t_step = timesteps[i].item()
                t = torch.full((nw,), t_step, device=DEVICE, dtype=torch.long)
                x = torch.cat([x_full[:, st:en] for st, en in windows], dim=0)
                v_pred = self._guided_v(x, t, cond_w, inst_t, cfg_scale, cfg_inst_scale)
                prev_a_bar = torch.tensor(1.0, device=DEVICE) if i == ddim_steps - 1 else acp[timesteps[i+1].item()]
                x = self._ddim_step(x, v_pred, acp[t_step], prev_a_bar)
                x_full = torch.zeros_like(x_full)
                for k, (st, en) in enumerate(windows):
                    x_full[:, st:en] += x[k:k+1] * w_t[k]
                x_full = x_full / (acc + 1e-8)
                if progress_cb:
                    progress_cb(i + 1, ddim_steps)
        return x_full

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
                            lora_strength=None, custom_lora_path=None, progress_cb=None,
                            cfg_inst_scale=None, match_source_loudness=True,
                            snap_to_semitones=False, timbre_mode=None, blend_mode=None):
        if blend_mode is None:
            blend_mode = DEFAULT_BLEND_MODE
        if blend_mode not in BLEND_MODES:
            raise ValueError(f"blend_mode must be one of {BLEND_MODES}")
        if timbre_mode is None:
            timbre_mode = DEFAULT_TIMBRE_MODE
        if timbre_mode not in TIMBRE_MODES:
            raise ValueError(f"timbre_mode must be one of {TIMBRE_MODES}")
        print(f"[GUI] Timbre source: {timbre_mode}")
        # Per-instrument v3 presets fill in anything not passed explicitly
        p_con, p_inst = v3_preset(instrument_id)
        if cfg_scale is None:
            cfg_scale = p_con
        if cfg_inst_scale is None:
            cfg_inst_scale = p_inst
        if lora_strength is None:
            lora_strength = v3_lora_strength(instrument_id)

        print(f"[GUI] Loading: {audio_path}")

        # Configure LoRA adapter: v3 adapters only (V3_LORA_MAP / train_lora_v3.py)
        if custom_lora_path and os.path.exists(custom_lora_path):
            print(f"[GUI] Applying Custom LoRA: {os.path.basename(custom_lora_path)} (Strength: {lora_strength:.2f}). "
                  "Must be trained with train_lora_v3.py for the v3 base.")
            self.set_lora_adapter(custom_lora_path, strength=lora_strength)
        elif use_lora and instrument_id in V3_LORA_MAP:
            lname, rel_path = V3_LORA_MAP[instrument_id]
            try:
                full = ensure_checkpoint(rel_path)
                print(f"[GUI] Applying v3 Studio LoRA: {lname} (Strength: {lora_strength:.2f})")
                self.set_lora_adapter(full, strength=lora_strength)
            except Exception as e:
                print(f"[GUI] Warning: could not get v3 LoRA {rel_path} ({e}); running pure v3.")
                self.set_lora_adapter(None, strength=0.0)
        else:
            print("[GUI] Running pure v3 base model (no LoRA)")
            self.set_lora_adapter(None, strength=0.0)

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

        # Frame grid = the SAME-L latent hop: 1 frame = 4096 samples (92.9 ms), exactly as in training
        # (131072-sample clips -> 32 frames). The frame count used to come from the encoder, whose
        # chunked encode drops frames on long files (16.25 s -> 172 instead of 175 frames). The contour
        # was then stretched over too few frames and the output drifted ahead of the input (~0.25 s
        # by the end of a 16 s file). Generation never used the encoded latent itself, only its length,
        # so we skip the encode and derive the grid from the sample count.
        LATENT_HOP = 4096
        total_frames = int(np.ceil(len(audio_for_encoding) / LATENT_HOP))
        audio_grid = np.pad(audio_for_encoding, (0, total_frames * LATENT_HOP - len(audio_for_encoding)))

        print("[GUI] Extracting contours...")
        contour = extract_contours(audio_grid, target_len=total_frames)  # (9, total_frames)
        contour = contour.copy()                 # raw (9, seq_len); pitch edits below go into this
        contour_6 = contour[:6, :]               # view: pitch edits + returned for the debug plots
        
        # 1. OPTIONAL: 12-TET Chromatic Pitch Snapping (Auto-Tune to exact semitones)
        voiced_mask = contour_6[1, :] > 0.25
        if snap_to_semitones and voiced_mask.any():
            f0 = np.exp(contour_6[0])
            midi = np.round(69.0 + 12.0 * np.log2(np.maximum(f0, 1e-4) / 440.0))
            f0_snapped = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
            contour_6[0, voiced_mask] = np.log(f0_snapped[voiced_mask])
            contour_6[0] = fill_unvoiced(contour_6[0], voiced_mask)
            print("[GUI] 12-TET Pitch Snapping applied: Locked voiced frames to discrete equal-temperament semitones.")

        # 2. REGISTER ALIGNMENT (Transpose into Target Instrument Range preserving musical key).
        # Re-enabled Nov 10 2026 for v3: it follows pitch exactly, so notes outside an instrument's range
        # came out unvoiced (e.g. a sax ending on G3 rendered as flute). Whole octaves only, key preserved.
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

                # Pick the whole-octave shift that leaves the fewest voiced frames outside the
                # instrument's range. Stay at 0 unless a shift fixes at least AUTO_OCTAVE_MIN_GAIN of the
                # frames, so one borderline note (e.g. 192 Hz vs violin's 196 Hz G3) can't move a whole phrase.
                AUTO_OCTAVE_MIN_GAIN = 0.10
                def frac_out(k):
                    f = voiced_f0 + k * np.log(2)
                    return float(np.mean((f < inst_min) | (f > inst_max)))
                cands = {k: frac_out(k) for k in range(-3, 4)}
                best_k = min(cands, key=lambda k: (cands[k], abs(k)))   # ties -> smallest shift
                shift_octaves = best_k if cands[0] - cands[best_k] >= AUTO_OCTAVE_MIN_GAIN else 0
                print(f"[GUI] Out-of-range frames: {cands[0]:.0%} unshifted, {cands[shift_octaves]:.0%} with "
                      f"{shift_octaves:+d} octave(s)")
            else:
                try:
                    shift_octaves = int(octave_shift)
                except (ValueError, TypeError):
                    shift_octaves = 0

            if shift_octaves != 0:
                contour_shift = shift_octaves * np.log(2)
                contour_6[0] += contour_shift
                print(f"[GUI] Transposed melody by {shift_octaves:+d} octave(s) to fit {INSTRUMENT_NAMES.get(instrument_id, '')}.")
            else:
                print(f"[GUI] Preserved original pitch (0 octave shift). Within {INSTRUMENT_NAMES.get(instrument_id, '')} playable range.")
                
        # 3. Normalization: the exact transform the selected base model was trained with
        contour_norm = self.normalize_for_model(contour, timbre_mode, instrument_id)

        # 4. Adaptive Windowing vs Single-Pass Generation
        if total_frames <= 40:
            # Short audio (<= ~3.7s): Standard single-pass (fastest, zero crossfade seams)
            cond_contour = (
                torch.from_numpy(contour_norm.astype("float32"))
                .unsqueeze(0).transpose(1, 2).to(DEVICE)    # (1, seq_len, C)
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

            # Trapezoidal Hann blend weights (used by both blend modes)
            weights = [self._generate_window_weights(window_size, 0 if idx == 0 else overlap,
                                                     0 if idx == num_windows - 1 else overlap)
                       for idx in range(num_windows)]

            if blend_mode == "synced":
                print(f"[GUI] Synced (MultiDiffusion) DDIM {ddim_steps} steps across {num_windows} windows: "
                      f"overlaps averaged after every step...")
                cond_full = torch.from_numpy(contour_norm.T.astype("float32")).unsqueeze(0).to(DEVICE)  # (1, T, C)
                latent_out = self._ddim_loop_synced(cond_full, windows, weights, instrument_id, cfg_scale,
                                                    ddim_steps, progress_cb, cfg_inst_scale=cfg_inst_scale)
            else:
                cond_list = [torch.from_numpy(contour_norm[:, st:en].T.astype("float32")) for (st, en) in windows]
                cond_batch = torch.stack(cond_list, dim=0).to(DEVICE) # (num_windows, window_size, C)

                print(f"[GUI] Standard blending: parallel batched DDIM {ddim_steps} steps across {num_windows} "
                      f"independent windows, overlap-added once at the end...")
                x_windows = self._ddim_loop(cond_batch, instrument_id, cfg_scale, ddim_steps, progress_cb, cfg_inst_scale=cfg_inst_scale)

                latent_out = torch.zeros(1, total_frames, 256, device=DEVICE)
                weight_acc = torch.zeros(1, total_frames, 1, device=DEVICE)
                for idx, (st, en) in enumerate(windows):
                    w_t = torch.from_numpy(weights[idx]).to(DEVICE).unsqueeze(0).unsqueeze(-1)
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
