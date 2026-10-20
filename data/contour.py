"""
Extract time-aligned control curves (F0, RMS energy, onset)
from an audio clip, resampled to the SAME latent grid (32 frames).
Compatible with CUDA, Apple Silicon (MPS), and CPU.
"""

import numpy as np
import torch
import torchcrepe
import librosa
import soundfile as sf
import os

# CONFIG
SR = 44100
TARGET_FRAMES = 32 # SAME latent length for a 131072-sample clip
FMIN, FMAX = 33.0, 2000.0 ## Min 33.0Hz (~C1, bin 3) prevents negative torchcrepe bin index bug; max 2000Hz for Flute/Piano
CREPE_MODEL = "tiny" # "tiny" is fast and ideal for real-time inference
CREPE_HOP = 128
VOICED_THRESHHOLD = 0.5 # CREPE periodicity above this = voiced

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_mono(path, start=0, duration_samples=131072):
    if os.name != "nt" and isinstance(path, str):
        path = path.replace("\\\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            path = f"/mnt/{drive}{path[2:]}"
            
    try:
        data, sr = sf.read(path, start=int(start), frames=int(duration_samples), always_2d=True, dtype='float32')
        data = data.mean(axis=1) # convert to mono
        if sr != 44100:
            data = librosa.resample(data, orig_sr=sr, target_sr=44100)
    except Exception:
        offset_sec = float(start) / 44100.0
        dur_sec = float(duration_samples) / 44100.0
        data, _ = librosa.load(path, sr=44100, mono=True, offset=offset_sec, duration=dur_sec)
        
    return data.astype(np.float32)


def extract_f0(audio):
    audio_t = torch.from_numpy(audio)[None] # (1, samples) - CREPE expects batch dimension
    fmin_safe = max(float(FMIN), 33.0)
    batch_size = 2048 if DEVICE in ["cuda", "mps"] else 512
    
    pitch, periodicity = torchcrepe.predict(
        audio_t, SR, hop_length=CREPE_HOP,
        fmin=fmin_safe, fmax=FMAX, model=CREPE_MODEL,
        return_periodicity=True, device=DEVICE, batch_size=batch_size
    )
    pitch, periodicity = pitch[0], periodicity[0]

    voiced = periodicity > VOICED_THRESHHOLD
    log_f0 = torch.zeros_like(pitch)
    log_f0[voiced] = torch.log(pitch[voiced])

    return log_f0.cpu().numpy(), voiced.cpu().numpy(), periodicity.cpu().numpy()


def pyin_pitch(audio_mono, target_len=32):
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_mono, fmin=50, fmax=2400, sr=44100, frame_length=2048, hop_length=1024)
    f0 = np.nan_to_num(f0, nan=1.0)
    log_f0 = np.zeros_like(f0)
    log_f0[voiced_flag] = np.log(f0[voiced_flag])
    aperiodicity = 1.0 - np.nan_to_num(voiced_probs, nan=0.0)
    return resample_to(fill_unvoiced(log_f0, voiced_flag), n=target_len), resample_to(voiced_flag.astype(float), n=target_len) > 0.5, resample_to(aperiodicity, n=target_len)


def extract_rms(y=None, S=None):
    return librosa.feature.rms(y=y, S=S, frame_length=2048, hop_length=CREPE_HOP)[0]


def extract_onset(audio):
    return librosa.onset.onset_strength(y=audio, sr=SR, hop_length=CREPE_HOP)


def extract_spectral_centroid(y=None, S=None):
    return librosa.feature.spectral_centroid(y=y, S=S, sr=SR, hop_length=CREPE_HOP)[0]


def extract_spectral_flatness(y=None, S=None):
    return librosa.feature.spectral_flatness(y=y, S=S, hop_length=CREPE_HOP)[0]


def extract_spectral_rolloff(y=None, S=None):
    return librosa.feature.spectral_rolloff(y=y, S=S, sr=SR, hop_length=CREPE_HOP, roll_percent=0.85)[0]


def extract_spectral_bandwidth(y=None, S=None):
    return librosa.feature.spectral_bandwidth(y=y, S=S, sr=SR, hop_length=CREPE_HOP)[0]


def resample_to(curve, n=TARGET_FRAMES):
    x_old = np.linspace(0.0, 1.0, num=len(curve))
    x_new = np.linspace(0.0, 1.0, num=n)
    return np.interp(x_new, x_old, curve)


def fill_unvoiced(log_f0, voiced):
    if voiced.sum() == 0:
        return log_f0
    idx = np.arange(len(log_f0))
    return np.interp(idx, idx[voiced], log_f0[voiced])


def extract_contours(audio, target_len=TARGET_FRAMES):
    log_f0, voiced, periodicity = extract_f0(audio)
    
    # Compute STFT once for efficiency
    S = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=CREPE_HOP))
    
    rms = extract_rms(S=S)
    onset = extract_onset(audio)
    centroid = extract_spectral_centroid(S=S)
    rolloff = extract_spectral_rolloff(S=S)
    bandwidth = extract_spectral_bandwidth(S=S)
    flatness = extract_spectral_flatness(S=S)

    log_f0_32 = resample_to(fill_unvoiced(log_f0, voiced), n=target_len)
    voiced_32 = (resample_to(voiced.astype(float), n=target_len) > 0.5).astype(float)
    rms_32 = resample_to(rms, n=target_len)
    onset_32 = resample_to(onset, n=target_len)

    aperiodicity_32 = resample_to(1.0 - periodicity, n=target_len)
    centroid_32 = resample_to(centroid, n=target_len)

    rolloff_32 = resample_to(rolloff, n=target_len)
    bandwidth_32 = resample_to(bandwidth, n=target_len)
    flatness_32 = resample_to(flatness, n=target_len)

    return np.stack([log_f0_32, voiced_32, rms_32, onset_32, aperiodicity_32, centroid_32, rolloff_32, bandwidth_32, flatness_32])
