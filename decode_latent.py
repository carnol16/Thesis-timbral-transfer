import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

import glob
import torch
import torch.backends.cuda
import torchaudio
from scipy.io import wavfile
import numpy as np
import argparse
from stable_audio_3 import AutoencoderModel
import stable_audio_3.models.transformer as transformer

# Disable flex_attention to prevent Triton autotuning hangs on varying chunk sizes
transformer.flex_attention_available = False
transformer.flex_attention_compiled = None

# Disable Triton/Flash attention to prevent shared memory hardware limits
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

os.environ["TORCH_COMPILE_DISABLE"] = "1"
def ensure_shape(tensor):
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] == 256:
        return tensor
    elif tensor.shape[2] == 256:
        return tensor.transpose(1, 2)
    else:
        raise ValueError(f"Unexpected shape: {tensor.shape}")

def chunked_decode(model, latent):
    total_frames = latent.shape[2]
    print(f"Total frames to decode: {total_frames}")
    
    # Force a small, safe chunk size to prevent attention hangs inside the VAE
    max_chunk = min(64, total_frames)
    print(f"Dynamic chunk size set to: {max_chunk} frames")
    
    if total_frames <= max_chunk or max_chunk == total_frames:
        with torch.no_grad():
            audio = model.decode(latent)
        return audio.to(latent.device)
        
    # Overlap-Discard (Overlap-Save) algorithm for seamless decoding
    overlap_frames = 10
    L = max_chunk - 2 * overlap_frames
    if L <= 0:
        overlap_frames = max_chunk // 4
        L = max_chunk - 2 * overlap_frames
        
    # Determine the audio hop size (H) per latent frame
    with torch.no_grad():
        test_audio = model.decode(latent[:, :, :2])
    H = test_audio.shape[-1] // 2
    
    audios = []
    for out_start in range(0, total_frames, L):
        out_end = min(total_frames, out_start + L)
        
        in_start = max(0, out_start - overlap_frames)
        in_end = min(total_frames, out_end + overlap_frames)
        
        chunk = latent[:, :, in_start:in_end]
        
        with torch.no_grad():
            a_out = model.decode(chunk)
            
        valid_start_sample = (out_start - in_start) * H
        valid_end_sample = (out_end - in_start) * H
        
        # Ensure bounds don't exceed actual audio length (padding edge cases)
        valid_end_sample = min(valid_end_sample, a_out.shape[-1])
        
        valid_audio = a_out[:, :, valid_start_sample:valid_end_sample]
        audios.append(valid_audio)
            
    return torch.cat(audios, dim=-1).to(latent.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=str, default=None, help="Optional: input latent file")
    parser.add_argument("--out", type=str, default=None, help="Optional: output wav file")
    parser.add_argument("--latent_dir", type=str, default=None, help="Optional: directory of latent files to decode")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional: output directory for decoded wav files")
    args = parser.parse_args()

    if DEVICE == "cuda":
        model = AutoencoderModel.from_pretrained("same-l")
    else:
        model = AutoencoderModel.from_pretrained("same-s")

    if args.latent_dir and args.out_dir:
        # BATCH DIRECTORY MODE: Decodes all .npy files in latent_dir
        os.makedirs(args.out_dir, exist_ok=True)
        npy_files = sorted(glob.glob(os.path.join(args.latent_dir, "*.npy")))
        print(f"[DECODE] Found {len(npy_files)} latents in {args.latent_dir}. Decoding in batch...")
        for i, npy_path in enumerate(npy_files, 1):
            stem = os.path.splitext(os.path.basename(npy_path))[0]
            out_wav = os.path.join(args.out_dir, f"{stem}.wav")
            latent_np = np.load(npy_path)
            latent = torch.from_numpy(latent_np).to(DEVICE)
            latent = ensure_shape(latent)
            with torch.no_grad():
                audio = chunked_decode(model, latent)
            audio = audio.squeeze(0).cpu()
            audio = audio / max(audio.abs().max(), 1e-8) * 0.9
            torchaudio.save(out_wav, audio, 44100)
            if i % 10 == 0 or i == len(npy_files):
                print(f"[DECODE] [{i}/{len(npy_files)}] Saved {out_wav}")
        print(f"[DECODE] Batch decoding completed!")
    elif args.latent and args.out:
        # APP MODE: Decode exactly what was requested
        print(f"Decoding custom latent: {args.latent}")
        latent_np = np.load(args.latent)
        latent = torch.from_numpy(latent_np).to(DEVICE)
        latent = ensure_shape(latent)

        with torch.no_grad():
            audio = chunked_decode(model, latent)

        audio = audio.squeeze(0).cpu()
        audio = audio / max(audio.abs().max(), 1e-8) * 0.9
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torchaudio.save(args.out, audio, 44100)
        print(f"Saved {args.out} successfully!")
    else:
        # NOTEBOOK MODE: Original hardcoded behavior
        # Load generated latent
        latent_np = np.load("artifacts/generated_latent.npy")
        latent = torch.from_numpy(latent_np).to(DEVICE)
        latent = ensure_shape(latent)

        print("Decoding generated latent...")
        with torch.no_grad():
            audio = chunked_decode(model, latent)

        audio = audio.squeeze(0).cpu()
        audio = audio / max(audio.abs().max(), 1e-8) * 0.9
        torchaudio.save("artifacts/generated_violin_v2.wav", audio, 44100)
        print("Saved artifacts/generated_violin_v2.wav successfully!")

        # Decode the Real Latent
        print("Decoding REAL latent...")
        real_np = np.load("artifacts/real_latent.npy")
        real_tensor = torch.from_numpy(real_np).to(DEVICE)
        real_tensor = ensure_shape(real_tensor)

        with torch.no_grad():
            real_audio = chunked_decode(model, real_tensor)
            
        real_audio = real_audio.squeeze(0).cpu()
        real_audio = real_audio / max(real_audio.abs().max(), 1e-8) * 0.9
        torchaudio.save("artifacts/real_violin_v2.wav", real_audio, 44100)
        print("Saved artifacts/real_violin_v2.wav successfully!")
