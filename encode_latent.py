import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch.backends.cuda
import numpy as np
import argparse
from stable_audio_3 import AutoencoderModel
import stable_audio_3.models.transformer as transformer

# Disable flex_attention to prevent Triton autotuning hangs on varying chunk sizes
transformer.flex_attention_available = False
transformer.flex_attention_compiled = None
import soundfile as sf

# Disable Triton/Flash attention to prevent shared memory hardware limits
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


def load_mono(path):
    import librosa
    data, _ = librosa.load(path, sr=44100, mono=True)
    return data

def chunked_encode(model, audio_t, sample_rate=44100):
    H = 2048
    # Pad audio to a multiple of H to ensure exact frame counts
    rem = audio_t.shape[-1] % H
    if rem != 0:
        import torch.nn.functional as F
        audio_t = F.pad(audio_t, (0, H - rem))
        
    total_samples = audio_t.shape[-1]
    print(f"Total audio samples to encode: {total_samples}")
    
    max_chunk = 44100 * 5  # Fallback to 5 seconds for Mac/CPU
    if torch.cuda.is_available():
        free_mem, _ = torch.cuda.mem_get_info()
        overhead = 500 * 1024**2 # 500MB safety overhead (encode uses less than decode)
        usable = max(0, free_mem - overhead)
        # Empirical: encode takes ~5000 bytes per sample
        dynamic_max = int(usable / 5000)
        # Ensure it's a multiple of 4096
        dynamic_max = dynamic_max - (dynamic_max % 4096)
        # Force a minimum of ~1 second to prevent microscopic chunks
        dynamic_max = max(40960, dynamic_max)
        max_chunk = min(dynamic_max, total_samples)
        # Make absolutely sure it's a multiple of 4096
        max_chunk = max_chunk - (max_chunk % 4096)
        
    print(f"Dynamic chunk size set to: {max_chunk} samples")
    
    if total_samples <= max_chunk or max_chunk == total_samples:
        with torch.no_grad():
            latent_out = model.encode(audio_t, sample_rate)
            latent_t = latent_out[0] if isinstance(latent_out, tuple) else latent_out
        return latent_t.to(audio_t.device)
        
    overlap_frames = 10
    max_chunk_frames = max_chunk // H
    L = max_chunk_frames - 2 * overlap_frames
    if L <= 0:
        overlap_frames = max_chunk_frames // 4
        L = max_chunk_frames - 2 * overlap_frames
        
    total_frames = total_samples // H
    
    latents = []
    for out_start in range(0, total_frames, L):
        out_end = min(total_frames, out_start + L)
        
        in_start = max(0, out_start - overlap_frames)
        in_end = min(total_frames, out_end + overlap_frames)
        
        in_start_sample = in_start * H
        in_end_sample = in_end * H
        
        chunk = audio_t[:, :, in_start_sample:in_end_sample]
        
        with torch.no_grad():
            l_out = model.encode(chunk, sample_rate)
            l_t = l_out[0] if isinstance(l_out, tuple) else l_out
            
        valid_start_frame = out_start - in_start
        valid_end_frame = out_end - in_start
        
        valid_end_frame = min(valid_end_frame, l_t.shape[-1])
        valid_latent = l_t[:, :, valid_start_frame:valid_end_frame]
        latents.append(valid_latent)
        
    return torch.cat(latents, dim=-1).to(audio_t.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="freesound_community-humming-7141.mp3")
    parser.add_argument("--out", type=str, default="artifacts/custom_latent.npy")
    args = parser.parse_args()

    print(f"Loading audio: {args.audio}")
    y = load_mono(args.audio)

    print("Loading Autoencoder...")
    if DEVICE == "cuda":
        model = AutoencoderModel.from_pretrained("same-l")
    else:
        model = AutoencoderModel.from_pretrained("same-s")

    audio_t = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print("Encoding audio to latent...")
    latent_t = chunked_encode(model, audio_t)
        
    # Ensure shape is (Batch, Seq_len, Channels) for the dataset
    if latent_t.shape[1] == 256:
        latent_t = latent_t.transpose(1, 2)
        
    latent_np = latent_t.squeeze(0).cpu().numpy()
    np.save(args.out, latent_np)
    print(f"Latent saved to {args.out}")
