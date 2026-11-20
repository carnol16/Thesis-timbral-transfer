"""
gui/app.py  –  Gradio UI for TimbreDiT: Neural Timbre Transfer Demo.
"""
import gradio as gr
from misc_tools.gui.inference import engine, INSTRUMENT_NAMES, v3_preset, v3_lora_strength, V3_LORA_MAP, DEFAULT_BLEND_MODE


TIMBRE_SOURCE_CHOICES = {
    "Target instrument only (recommended)": "instrument",
    "Target character, keep source brightness changes": "retarget",
    "Keep source character (like-to-like, e.g. violin -> violin)": "source",
}
DEFAULT_TIMBRE_CHOICE = next(iter(TIMBRE_SOURCE_CHOICES))

BLEND_CHOICES = {
    "Synced (MultiDiffusion)": "synced",
    "Standard (blend once)": "standard",
}
DEFAULT_BLEND_CHOICE = next(k for k, v in BLEND_CHOICES.items() if v == DEFAULT_BLEND_MODE)


# ---------------------------------------------------------------------------
# Instrument Configuration
# ---------------------------------------------------------------------------

# Friendly display labels (no raw ID shown to the visitor)
INSTRUMENT_LABELS = {
    0:  "Accordion",
    1:  "Acoustic Grand Piano",
    2:  "Acoustic Guitar (Steel)",
    3:  "Alto Saxophone",
    4:  "Brass Section",
    5:  "Cello",
    6:  "Choir Aahs",
    7:  "Clarinet",
    8:  "Distortion Guitar",
    9:  "Electric Bass",
    10: "Electric Guitar (Clean)",
    11: "Electric Piano",
    12: "Flute",
    13: "Rock Organ",
    14: "String Ensemble",
    15: "Trombone",
    16: "Trumpet",
    17: "Violin",
}
INSTRUMENT_CHOICES = [INSTRUMENT_LABELS[k] for k in sorted(INSTRUMENT_LABELS)]

def label_to_id(label: str) -> int:
    for k, v in INSTRUMENT_LABELS.items():
        if v == label:
            return k
    raise ValueError(f"Unknown instrument label: {label!r}")

# ---------------------------------------------------------------------------
# Generation callback
# ---------------------------------------------------------------------------

def generate(audio_path, instrument_label, cfg_scale, ddim_steps_preset, octave_shift_choice,
             show_plots=False, use_lora=True, lora_strength=1.0,
             custom_lora=None, snap_to_semitones=False, inst_cfg_scale=None,
             timbre_source=DEFAULT_TIMBRE_CHOICE, blend_choice=DEFAULT_BLEND_CHOICE, progress=gr.Progress()):
    if audio_path is None:
        raise gr.Error("Please upload an audio file first.")

    inst_id = label_to_id(instrument_label)

    steps_map = {"Fast (25)": 25, "Normal (50)": 50, "Quality (100)": 100}
    ddim_steps = steps_map.get(ddim_steps_preset, 50)

    octave_map = {
        "No Shift (Original Pitch)": 0,
        "+1 Octave": 1,
        "-1 Octave": -1,
        "Auto (Match Instrument Range)": "Auto",
        "+2 Octaves": 2,
        "-2 Octaves": -2,
    }
    octave_shift = octave_map.get(octave_shift_choice, 0)

    custom_lora_path = custom_lora if isinstance(custom_lora, str) else (
        custom_lora.name if custom_lora is not None else None)

    def progress_cb(step, total):
        if step == 0:
            progress(0, desc="Encoding audio & extracting contours...")
        else:
            progress(step / total, desc=f"Generating audio... step {step}/{total}")

    inst_cfg = float(inst_cfg_scale) if inst_cfg_scale is not None else v3_preset(inst_id)[1]

    audio_out, sr, audio_in, contour = engine.run_timbre_transfer(
        audio_path=audio_path,
        instrument_id=inst_id,
        cfg_scale=float(cfg_scale),
        ddim_steps=int(ddim_steps),
        octave_shift=octave_shift,
        use_lora=bool(use_lora),
        lora_strength=float(lora_strength),
        custom_lora_path=custom_lora_path,
        progress_cb=progress_cb,
        cfg_inst_scale=float(inst_cfg),
        snap_to_semitones=bool(snap_to_semitones),
        timbre_mode=TIMBRE_SOURCE_CHOICES.get(timbre_source, "instrument"),
        blend_mode=BLEND_CHOICES.get(blend_choice, DEFAULT_BLEND_MODE),
    )

    fig = None
    if show_plots:
        progress(0.9, desc="Generating debug plots (spectrograms & contours)...")
        import matplotlib.pyplot as plt
        import numpy as np
        import librosa
        import librosa.display
        from data.contour import extract_contours

        seq_len = contour.shape[1]
        # audio_out is (frames, channels): average to mono. flatten() would interleave L/R into a
        # signal twice as long, which reads as one octave low (and half speed) in CREPE + spectrogram.
        audio_out_mono = audio_out.mean(axis=1) if audio_out.ndim == 2 else audio_out
        # Use consistent CREPE tracking across both input and generated output
        # to ensure fair, aligned intonation evaluation.
        out_contour = extract_contours(audio_out_mono, target_len=seq_len)
        unnorm_contour = contour

        # Compute intonation difference in musical cents & semitones on mutually voiced frames
        both_voiced = (unnorm_contour[1] > 0.5) & (out_contour[1] > 0.5)
        pitch_err_str = ""
        if np.any(both_voiced):
            delta_cents = 1200.0 * (out_contour[0, both_voiced] - unnorm_contour[0, both_voiced]) / np.log(2.0)
            median_cents = float(np.median(delta_cents))
            median_semi = median_cents / 100.0
            pitch_err_str = f" [Median Error: {median_cents:+.1f}¢ / {median_semi:+.2f} semitones]"

        fig, axes = plt.subplots(8, 1, figsize=(10, 20), sharex=False)

        D_in = librosa.amplitude_to_db(np.abs(librosa.stft(audio_in)), ref=np.max)
        librosa.display.specshow(D_in, sr=sr, x_axis='time', y_axis='log', ax=axes[0])
        axes[0].set_title("Input Spectrogram")

        D_out = librosa.amplitude_to_db(np.abs(librosa.stft(audio_out_mono)), ref=np.max)
        librosa.display.specshow(D_out, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title("Output Spectrogram")

        titles = [
            f"Log-F0 (Pitch){pitch_err_str}",
            "Voiced (Confidence)",
            "RMS (Volume)",
            "Onset (Attack)",
            "Aperiodicity",
            "Spectral Centroid"
        ]

        for i in range(6):
            ax = axes[i + 2]
            real_val = unnorm_contour[i] if i < 6 else np.zeros(seq_len)
            gen_val = out_contour[i]

            ax.plot(real_val, label="Real (Input / Shifted)", color="pink", linewidth=2.5)
            ax.plot(gen_val, label="Generated", color="blue", linestyle="--")

            ax.set_title(titles[i])
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Latent Time Steps")
        plt.tight_layout()

    progress(1.0, desc="Done!")
    import numpy as np
    if audio_out.dtype != np.int16:
        audio_out_int16 = (np.clip(audio_out, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        audio_out_int16 = audio_out
    return (sr, audio_out_int16), fig

# ---------------------------------------------------------------------------
# CSS — Inter font, hide footer
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
}

footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------

HEADER_MD = """
<div style="margin-bottom: 8px;">
<strong style="font-size: 1.4rem; font-family: 'Inter', sans-serif; letter-spacing: 0.02em;">
  TimbreDiT
</strong>
<span style="font-size: 0.85rem; margin-left: 10px; opacity: 0.6;">
  CalArts MFA Thesis &nbsp;&middot;&nbsp; Colton Arnold
</span>
<span style="float: right; font-size: 0.82rem;">
  <a href="https://huggingface.co/carnol16/timbral-transfer" target="_blank">HF Model</a>
  &nbsp;&nbsp;
  <a href="https://github.com/carnol16" target="_blank">GitHub</a>
  &nbsp;&nbsp;
  <a href="https://coltonarnold.com" target="_blank">coltonarnold.com</a>
</span>
</div>
"""

ABOUT_MD = """
## TimbreDiT — Neural Instrument Timbre Transfer

**TimbreDiT** is a research system that converts any melodic audio into the timbre of a target
musical instrument. Upload a recording of a hum, vocal melody, acoustic guitar, or saxophone line
and the model re-renders it in the acoustic body of a Violin, Flute, Trumpet, or any of 18
instrument classes — preserving the original pitch contour, dynamics, and musical phrasing.

The system is built around a **173M-parameter Latent Diffusion Transformer (DiT)** conditioned on
performance contours (log-F0 with Fourier pitch features, voicing confidence, RMS energy, onset
strength, aperiodicity, and spectral centroid) extracted from the source audio. A Stable Audio 3
autoencoder encodes and decodes the latent representation at 44,100 Hz.

For five acoustic instruments — Violin, Flute, Alto Saxophone, Trumpet, and Clarinet — curated
**Studio LoRA adapters** (rank 8, trained on every Medley-solos-DB training song of the instrument,
3–24 real players) are available, delivering richer body resonance, breath air, and bow texture
compared to the base model alone.

---

| Attribute | Details |
|---|---|
| **Architecture** | Modified Diffusion Transformer (DiT) with RoPE, instrument embedding |
| **Parameters** | 173 M (v3 pitch-fixed, Nov 2026) |
| **Latent Autoencoder** | Stable Audio 3 (Harmonai / Stability AI) |
| **Sample Rate** | 44,100 Hz |
| **Instrument Classes** | 18 (Slakh2100 taxonomy) |
| **Training Data** | Slakh2100 dataset (MIDI-rendered multitrack) |
| **LoRA Training Data** | Medley-solos-DB (uncompressed solo performer recordings) |
| **LoRA Adapters** | 5 (Violin, Flute, Alto Saxophone, Trumpet, Clarinet) |
| **Inference Method** | DDIM (25 / 50 / 100 steps) with Classifier-Free Guidance |
| **Hardware** | CUDA (NVIDIA) · MPS (Apple Silicon) · CPU fallback |
"""

# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="TimbreDiT — Neural Timbre Transfer",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(HEADER_MD)

    with gr.Tabs():

        # ── Studio ───────────────────────────────────────────────────────────
        with gr.TabItem("Studio"):

            gr.Markdown("""
# Timbre Transfer Demo (Slakh18 Edition)
Upload any audio (voice, guitar, humming, sax...) and the model will re-render
it in the timbre of your chosen instrument using our 173M Diffusion Transformer (Modified DiT).
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(
                        label="Source Audio (drag & drop or click to browse)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    instrument = gr.Dropdown(
                        label="Target Instrument",
                        choices=INSTRUMENT_CHOICES,
                        value="Violin",
                    )
                    octave_shift = gr.Dropdown(
                        label="Octave Transposition (Register Alignment)",
                        choices=[
                            "No Shift (Original Pitch)",
                            "+1 Octave",
                            "-1 Octave",
                            "Auto (Match Instrument Range)",
                            "+2 Octaves",
                            "-2 Octaves",
                        ],
                        value="Auto (Match Instrument Range)",
                        info="Shifts melody by full integer octaves to land in the instrument's register."
                    )
                    timbre_source_rad = gr.Radio(
                        label="Timbre Source",
                        choices=list(TIMBRE_SOURCE_CHOICES),
                        value=DEFAULT_TIMBRE_CHOICE,
                        info="Recommended: ignores the source's brightness and breathiness, so the chosen instrument "
                             "decides the sound (a flute melody really sounds like a trumpet). The middle option keeps how "
                             "brightness moves, in the target's range. 'Keep source' copies the source's character.",
                    )
                    blend_rad = gr.Radio(
                        label="Window Blending (inputs longer than ~3.7 s)",
                        choices=list(BLEND_CHOICES),
                        value=DEFAULT_BLEND_CHOICE,
                        info="Long inputs are rendered as overlapping windows. Synced keeps neighbouring windows in "
                             "agreement during sampling (no dip or timbre shift at the joins). Standard renders each "
                             "window alone and crossfades at the end (the original method). Same speed.",
                    )
                    snap_to_semitones_chk = gr.Checkbox(
                        label="Snap Pitch to Semitones (12-TET Chromatic Auto-Tune)",
                        value=False,
                        info="Locks microtonal humming/whistling drift onto dead-on musical notes (A=440Hz chromatic scale)."
                    )
                    cfg_scale = gr.Slider(
                        label="Contour CFG Scale (Melody Guidance — higher = tighter pitch lock)",
                        minimum=1.0, maximum=10.0, value=v3_preset(17)[0], step=0.5,
                    )
                    inst_cfg_sld = gr.Slider(
                        label="Instrument Prior CFG (lower = pure contour lock; higher = acoustic tone pull)",
                        minimum=0.0, maximum=8.0, value=v3_preset(17)[1], step=0.1,
                        info="Per-instrument preset; some instruments use a stronger instrument pull (e.g. brass 4 / 6)."
                    )
                    ddim_steps = gr.Radio(
                        label="Diffusion Steps",
                        choices=["Fast (25)", "Normal (50)", "Quality (100)"],
                        value="Normal (50)",
                    )

                    with gr.Accordion("Studio LoRA (real-instrument timbre)", open=True):
                        use_lora_chk = gr.Checkbox(
                            label="Enable Studio LoRA (real-player timbre)",
                            value=True,
                            interactive=True,
                            info="Rank-8 adapters trained on every Medley-solos-DB training song of the instrument "
                                 "(3-24 real players). Available for Violin, Flute, Clarinet, Trumpet, Alto Sax.",
                        )
                        lora_strength_sld = gr.Slider(
                            label="LoRA Acoustic Intensity",
                            minimum=0.0, maximum=1.5, value=v3_lora_strength(17), step=0.05,
                            info="0.0 = pure base model, 1.00 = trained strength, lower = subtler (violin preset 0.8).",
                        )
                        custom_lora_file = gr.File(
                            label="Upload Custom LoRA (.pt, trained with train_lora_v3.py for the v3 base)",
                            file_types=[".pt"],
                            type="filepath",
                        )

                    show_plots = gr.Checkbox(
                        label="Generate Debug Plots (Spectrograms & Contours)",
                        value=False,
                        info="Uncheck to generate audio faster (skips plot extraction).",
                    )
                    generate_btn = gr.Button("Generate Transfer", variant="primary", size="lg")

                with gr.Column(scale=1):
                    audio_out = gr.Audio(
                        label="Generated Output",
                        type="numpy",
                        interactive=False,
                    )
                    with gr.Accordion("Debug Plots (Comparison Contours)", open=False):
                        debug_plot = gr.Plot(label="Analysis")

                    gr.Markdown("""
### Tips
- **Studio LoRA**: Trained on uncompressed solo performer takes from Medley-solos-DB. Yields rich wooden resonance, breath air, and bow scrape.
- **Auto Octave**: Snaps the melody up/down by integer octaves to land in the target instrument's playable register while preserving all notes and musical key.
- **CFG Scale 2.5–3.5**: Natural instrument timbre with solid melody adherence.
- **Diffusion Steps**: 25 is fast preview; 50 is recommended; 100 provides highest texture quality.
                    """)

        # ── Examples ─────────────────────────────────────────────────────────
        with gr.TabItem("Examples"):
            gr.Markdown("""
## Curated Transfer Examples

Each tab below shows a **source audio clip** alongside its **generated output** from TimbreDiT.
These examples are pre-computed to give you a reference for audio quality before running your own input.
            """)

            with gr.Tabs():

                with gr.TabItem("Alto Sax to Studio Violin"):
                    gr.Markdown("""
**Source:** Live alto saxophone melody  
**Target instrument:** Violin with Studio LoRA adapter  
**Settings:** CFG 3.0 · LoRA 0.75 · 50 steps · No pitch shift
                    """)
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Source  (Alto Saxophone)")
                            gr.Audio(
                                value="external_test_audio/sax1.wav",
                                label="Source — Alto Saxophone",
                                interactive=False,
                            )
                        with gr.Column():
                            gr.Markdown("#### Generated Output  (Studio Violin)")
                            gr.Audio(
                                value=None,   # swap in your generated .wav path here
                                label="Output — Studio Violin (TimbreDiT)",
                                interactive=False,
                            )
                    gr.Markdown(
                        "_Generated output placeholder — run the Studio tab with `sax1.wav` "
                        "and Violin selected to produce this result._"
                    )

        # ── About ─────────────────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown(ABOUT_MD)

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_instrument_change(inst_choice):
        # Per-instrument CFG presets (sweep + listening test) and the Studio LoRAs (rank 8, all Medley train songs)
        inst_id = label_to_id(inst_choice)
        p_con, p_inst = v3_preset(inst_id)
        lora_available = inst_id in V3_LORA_MAP
        return (
            gr.update(value=p_con),
            gr.update(value=v3_lora_strength(inst_id)),
            gr.update(interactive=lora_available, value=lora_available),
            gr.update(value=p_inst),
        )

    preset_outputs = [cfg_scale, lora_strength_sld, use_lora_chk, inst_cfg_sld]
    instrument.change(fn=on_instrument_change, inputs=[instrument], outputs=preset_outputs)

    generate_btn.click(
        fn=generate,
        inputs=[audio_in, instrument, cfg_scale, ddim_steps, octave_shift, show_plots,
                use_lora_chk, lora_strength_sld, custom_lora_file,
                snap_to_semitones_chk, inst_cfg_sld, timbre_source_rad, blend_rad],
        outputs=[audio_out, debug_plot],
    )
