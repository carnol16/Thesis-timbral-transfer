"""
gui/app.py  –  Gradio UI for the timbre-transfer demo.
"""
import gradio as gr
from misc_tools.gui.inference import engine, INSTRUMENT_NAMES

INSTRUMENT_CHOICES = [f"{v} (ID {k})" for k, v in INSTRUMENT_NAMES.items()]

def label_to_id(label):
    return int(label.split("ID ")[-1].rstrip(")"))

def generate(audio_path, instrument_label, cfg_scale, ddim_steps_preset, octave_shift_choice, progress=gr.Progress()):
    if audio_path is None:
        raise gr.Error("Please upload an audio file first.")

    inst_id = label_to_id(instrument_label)
    
    steps_map = {"Fast (25)": 25, "Normal (50)": 50, "Quality (100)": 100}
    ddim_steps = steps_map.get(ddim_steps_preset, 50)

    octave_map = {
        "Auto (Match Instrument Range)": "Auto",
        "+2 Octaves": 2,
        "+1 Octave": 1,
        "No Shift (0)": 0,
        "-1 Octave": -1,
        "-2 Octaves": -2,
    }
    octave_shift = octave_map.get(octave_shift_choice, "Auto")

    def progress_cb(step, total):
        if step == 0:
            progress(0, desc="Encoding audio & extracting contours...")
        else:
            progress(step / total, desc=f"Generating audio... step {step}/{total}")

    audio_out, sr, audio_in, contour = engine.run_timbre_transfer(
        audio_path=audio_path,
        instrument_id=inst_id,
        cfg_scale=float(cfg_scale),
        ddim_steps=int(ddim_steps),
        octave_shift=octave_shift,
        progress_cb=progress_cb,
    )
    
    progress(0.9, desc="Generating debug plots (spectrograms & contours)...")
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    from data.contour import extract_contours, pyin_pitch
    
    seq_len = contour.shape[1]
    out_contour = extract_contours(audio_out.flatten(), target_len=seq_len)
    
    # Track generated audio with pYIN for pitch evaluation
    pyin_f0, pyin_voiced, pyin_ap = pyin_pitch(audio_out.flatten(), target_len=seq_len)
    out_contour[0] = pyin_f0
    out_contour[1] = pyin_voiced
    out_contour[4] = pyin_ap
    
    # contour returned from engine is already in physical units
    unnorm_contour = contour
    
    fig, axes = plt.subplots(8, 1, figsize=(10, 20), sharex=False)
    
    # 1. Input Spectrogram
    D_in = librosa.amplitude_to_db(np.abs(librosa.stft(audio_in)), ref=np.max)
    librosa.display.specshow(D_in, sr=sr, x_axis='time', y_axis='log', ax=axes[0])
    axes[0].set_title("Input Spectrogram")
    
    # 2. Output Spectrogram
    D_out = librosa.amplitude_to_db(np.abs(librosa.stft(audio_out.flatten())), ref=np.max)
    librosa.display.specshow(D_out, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title("Output Spectrogram")
    
    # 3-8. Contours
    titles = [
        "Log-F0 (Pitch)",
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
    return (sr, audio_out), fig


with gr.Blocks(title="Timbre Transfer", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎻 Timbre Transfer Demo (Slakh18 Edition)
    Upload any audio (voice, guitar, humming, sax...) and the model will re-render
    it in the timbre of your chosen instrument using our 114M Diffusion Transformer (Modified DiT).
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
                value="Violin (ID 17)",
            )
            octave_shift = gr.Dropdown(
                label="Octave Transposition (Pitch Snapping)",
                choices=[
                    "Auto (Match Instrument Range)",
                    "+2 Octaves",
                    "+1 Octave",
                    "No Shift (0)",
                    "-1 Octave",
                    "-2 Octaves",
                ],
                value="Auto (Match Instrument Range)",
            )
            cfg_scale = gr.Slider(
                label="CFG Scale (higher = stronger contour guidance)",
                minimum=1.0, maximum=10.0, value=3.0, step=0.5,
            )
            ddim_steps = gr.Radio(
                label="Diffusion Steps",
                choices=["Fast (25)", "Normal (50)", "Quality (100)"],
                value="Normal (50)",
            )
            generate_btn = gr.Button("🎻 Generate Transfer", variant="primary", size="lg")

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
            - **Auto Octave**: Snaps the melody up/down by integer octaves to land in the target instrument's playable register while preserving all notes and musical key.
            - **CFG Scale 2.5–3.5**: Natural instrument timbre with solid melody adherence.
            - **CFG Scale 5.0–8.0**: High contour enforcement (strong pitch vibrato & dynamics).
            - **Diffusion Steps**: 25 is fast preview; 50 is recommended; 100 provides highest texture quality.
            """)

    generate_btn.click(
        fn=generate,
        inputs=[audio_in, instrument, cfg_scale, ddim_steps, octave_shift],
        outputs=[audio_out, debug_plot],
    )
