"""
Extract time-aligned control curves (F0, RMS energy, onset)
from a violin clip, resampled to the SAME latent grid (32 frames).
Runs in the MAIN env.
"""

import numpy as np
import torch
import torchcrepe
import librosa
import soundfile as sf


#CONFIG
SR = 44100
TARGET_FRAMES = 32 #SAME latent length for a 131072-sample clip
FMIN, FMAX = 33.0, 2000.0 ## Min 33.0Hz (~C1, bin 3) prevents negative torchcrepe bin index bug; max 2000Hz for Flute/Piano
CREPE_MODEL=  "tiny" # "full" =vaccurate but slower, 'tiny' = fast
CREPE_HOP = 128
VOICED_THRESHHOLD = 0.5 #CREPE preiodicity above this = voiced
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #CREPE runs ~10x faster on the 3080


#load clip as mono+ array
TEST_WAV = "E:/Text_to_Audio/data/Medley-solos-DB/Medley-solos-DB_test-7_0004a899-677d-5f15-fac8-7eb7a91148c0.wav"

import os
import soundfile as sf

def load_mono(path, start=0, duration_samples=131072):
    import librosa
    import os
    import soundfile as sf
    if os.name != "nt" and isinstance(path, str):
        path = path.replace("\\\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            path = f"/mnt/{drive}{path[2:]}"
            
    # Use soundfile for fast seeking (librosa.load is very slow for offsets)
    try:
        data, sr = sf.read(path, start=int(start), frames=int(duration_samples), always_2d=True, dtype='float32')
        data = data.mean(axis=1) # convert to mono
        if sr != 44100:
            data = librosa.resample(data, orig_sr=sr, target_sr=44100)
    except:
        # Fallback to librosa if sf fails
        offset_sec = float(start) / 44100.0
        dur_sec = float(duration_samples) / 44100.0
        data, _ = librosa.load(path, sr=44100, mono=True, offset=offset_sec, duration=dur_sec)

        
    return data.astype(np.float32)                                 # (frames,) mono

"""
#check above function
audio = load_mono(TEST_WAV)
print("audio:", audio.shape, "duration:", round(len(audio) / SR, 3), "s")
"""


def extract_f0(audio):
    audio_t = torch.from_numpy(audio)[None] # (1, samples) - CREPE wants a batch dim
    fmin_safe = max(float(FMIN), 33.0) # Clamp to at least 33.0Hz (bin 3) to give headroom above torchcrepe bin 0 (~32.7Hz)
    pitch, periodicity = torchcrepe.predict(
        audio_t, SR, hop_length=CREPE_HOP,
        fmin=fmin_safe, fmax=FMAX, model=CREPE_MODEL,
        return_periodicity=True, device=DEVICE, batch_size=4096
    )
    pitch, periodicity = pitch[0], periodicity[0] # drop batch size => (frames,) i had to throw the chuck in there somewhere

    voiced = periodicity > VOICED_THRESHHOLD # bool per frame
    log_f0 = torch.zeros_like(pitch) #unvoiced frames stay 0
    log_f0[voiced] = torch.log(pitch[voiced])

    return log_f0.cpu().numpy(), voiced.cpu().numpy(), periodicity.cpu().numpy() #.cpu() is required: CUDA tensors can't convert to numpy directly

def pyin_pitch(audio_mono, target_len=32):
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_mono, fmin=50, fmax=2400, sr=44100, frame_length=2048, hop_length=1024)
    f0 = np.nan_to_num(f0, nan=1.0)
    log_f0 = np.zeros_like(f0)
    log_f0[voiced_flag] = np.log(f0[voiced_flag])
    aperiodicity = 1.0 - np.nan_to_num(voiced_probs, nan=0.0)
    return resample_to(fill_unvoiced(log_f0, voiced_flag), n=target_len), resample_to(voiced_flag.astype(float), n=target_len) > 0.5, resample_to(aperiodicity, n=target_len)


"""
#check above functions
log_f0, voiced = extract_f0(audio)
print("f0 frames:", log_f0.shape[0], "| voiced %:", round(100 * voiced.mean(), 1))
import numpy as np
hz = np.exp(log_f0[voiced])
print("voiced F0 Hz range:", round(hz.min(), 1), "-", round(hz.max(), 1))
"""
def extract_rms(y=None, S=None):
    rms = librosa.feature.rms(y=y, S=S, frame_length=2048, hop_length=CREPE_HOP)[0]
    return rms

def extract_onset(audio):
    onset = librosa.onset.onset_strength(y=audio, sr=SR, hop_length= CREPE_HOP)
    return onset

def extract_spectral_centroid(y=None, S=None):
    centroid = librosa.feature.spectral_centroid(y=y, S=S, sr=SR, hop_length=CREPE_HOP)[0]
    return centroid


#Measure how tone like vs versus noise a signal is across frequencys (tone like == 0 , noise like == 1)
def extract_spectral_flatness(y=None, S=None):
    flatness = librosa.feature.spectral_flatness(y=y, S=S, hop_length=CREPE_HOP)[0]
    return flatness

#The frequency below which a specified percentage (typically between 85-95%) of the total spectral energy is contained
def extract_spectral_rolloff(y=None, S=None):
    rolloff = librosa.feature.spectral_rolloff(y=y, S=S, sr=SR, hop_length=CREPE_HOP, roll_percent=0.85)[0]
    return rolloff

#measures the dispersion/spread of frequencies around the spectral centroid
def extract_spectral_bandwidth(y=None, S=None):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, S=S, sr=SR, hop_length=CREPE_HOP)[0]
    return bandwidth




"""
#check above functions
rms = extract_rms(audio)
onset = extract_onset(audio)
print("rms frames:", rms.shape[0], "| onset frames:", onset.shape[0])
print("rms range:", round(float(rms.min()), 4), "-", round(float(rms.max()), 4))
"""


"""
Maps any length curve onto a normalized 0 => 1 axisand samples are spaced
evenly betwwwn 32 points via linear interp. 
"""
def resample_to(curve, n=TARGET_FRAMES):
    #Linearly resample 1-d curve to n points
    x_old = np.linspace(0.0, 1.0, num=len(curve))
    x_new = np.linspace(0.0, 1.0, num=n)
    return np.interp(x_new, x_old, curve)

"""
Rebuilds a continous pitch line that interploates only across the voiced anchor points
filling all gaps smoothly. 
The seperate "voiced" flag still records where pitch was real - we are just giving
the model a clean continous curve to condition instead of zero-gaps
"""
def fill_unvoiced(log_f0, voiced):
    #interpolate pitch across unvoiced gaps so downsampling never blends real F0 with zeros
    if voiced.sum() == 0:
        return log_f0

    idx = np.arange(len(log_f0))
    return np.interp(idx, idx[voiced], log_f0[voiced])
"""
#Checks above functions
log_f0_filled = fill_unvoiced(log_f0, voiced)
log_f0_32 = resample_to(log_f0_filled)
voiced_32 = (resample_to(voiced.astype(float)) > 0.5).astype(float)   # soft-resample, then re-threshold
rms_32    = resample_to(rms)
onset_32  = resample_to(onset)

contour = np.stack([log_f0_32, voiced_32, rms_32, onset_32])   # (4, 32)
print("contour shape:", contour.shape)
"""
def extract_contours(audio, target_len=TARGET_FRAMES):
    #return (4, 32) conditioning: log_f0, cvoiced, rms, onset - all aligned to the latent grid
    log_f0, voiced, periodicity = extract_f0(audio)
    
    # Compute STFT once (massive CPU speedup)
    S = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=CREPE_HOP))
    
    rms = extract_rms(S=S)
    onset = extract_onset(audio) # Onset uses its own specific spectrogram internally usually, safer to leave as audio
    centroid = extract_spectral_centroid(S=S)
    rolloff = extract_spectral_rolloff(S=S)
    bandwidth = extract_spectral_bandwidth(S=S)
    flatness = extract_spectral_flatness(S=S)

    log_f0_32 = resample_to(fill_unvoiced(log_f0, voiced), n=target_len)
    voiced_32 = (resample_to(voiced.astype(float), n=target_len) > 0.5).astype(float)
    rms_32 = resample_to(rms, n=target_len)
    onset_32 = resample_to(onset, n=target_len)

    aperiodicity_32 = resample_to(1.0 - periodicity, n=target_len)  # Inverse of periodicity
    centroid_32 = resample_to(centroid, n=target_len)

    rolloff_32 = resample_to(rolloff, n=target_len)
    bandwidth_32 = resample_to(bandwidth, n=target_len)
    flatness_32 = resample_to(flatness, n=target_len)

    return np.stack([log_f0_32, voiced_32, rms_32, onset_32, aperiodicity_32, centroid_32, rolloff_32, bandwidth_32, flatness_32]) #should be (4,32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default=TEST_WAV)   # optional; falls back to TEST_WAV
    args = ap.parse_args()

    audio = load_mono(TEST_WAV)
    contour = extract_contours(audio)
    print("contour shape:", contour.shape)
