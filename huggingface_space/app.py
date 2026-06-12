import os
import sys
import yaml
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MIMIC_DIR = ROOT / "MimicMotion"
MODELS_DIR = MIMIC_DIR / "models"
DWPOSE_DIR = MODELS_DIR / "DWPose"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_DIR = MIMIC_DIR / "configs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Model setup (runs once at startup) ─────────────────────────────────────

def _run(cmd: str, cwd=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout


def setup_models():
    """Clone repo and download weights if not already present."""
    log = []

    # Clone MimicMotion
    if not MIMIC_DIR.exists():
        log.append("Cloning MimicMotion…")
        _run(f"git clone https://github.com/tencent/MimicMotion {MIMIC_DIR}")
        log.append("✅ Repo cloned")
    else:
        log.append("✅ Repo already present")

    # Add MimicMotion to path
    if str(MIMIC_DIR) not in sys.path:
        sys.path.insert(0, str(MIMIC_DIR))

    # DWPose weights
    DWPOSE_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in {
        "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
        "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
    }.items():
        dest = DWPOSE_DIR / fname
        if not dest.exists():
            log.append(f"Downloading {fname}…")
            _run(f'wget -q "{url}" -O "{dest}"')
            log.append(f"✅ {fname}")
        else:
            log.append(f"✅ {fname} cached")

    # MimicMotion checkpoint
    ckpt = MODELS_DIR / "MimicMotion_1-1.pth"
    if not ckpt.exists():
        log.append("Downloading MimicMotion_1-1.pth (~3 GB)…")
        _run(
            "wget -q https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth"
            f' -O "{ckpt}"'
        )
        log.append("✅ MimicMotion_1-1.pth")
    else:
        log.append("✅ MimicMotion_1-1.pth cached")

    return "\n".join(log)


# ── Inference ───────────────────────────────────────────────────────────────

def run_inference(
    ref_image,
    ref_video,
    resolution: int,
    num_frames: int,
    sample_stride: int,
    frames_overlap: int,
    num_inference_steps: int,
    guidance_scale: float,
    noise_aug_strength: float,
    fps: int,
    seed: int,
    use_float16: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if ref_image is None:
        raise gr.Error("Por favor sube una foto del personaje.")
    if ref_video is None:
        raise gr.Error("Por favor sube un video de referencia.")

    run_id = uuid.uuid4().hex[:8]
    run_out = OUTPUT_DIR / run_id
    run_out.mkdir(parents=True)

    config = {
        "base_model_path": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "ckpt_path": str(MODELS_DIR / "MimicMotion_1-1.pth"),
        "test_cases": [
            {
                "ref_video_path": str(ref_video),
                "ref_image_path": str(ref_image),
                "num_frames": int(num_frames),
                "resolution": int(resolution),
                "frames_overlap": int(frames_overlap),
                "num_inference_steps": int(num_inference_steps),
                "noise_aug_strength": float(noise_aug_strength),
                "guidance_scale": float(guidance_scale),
                "sample_stride": int(sample_stride),
                "fps": int(fps),
                "seed": int(seed),
            }
        ],
    }

    cfg_path = CONFIG_DIR / f"run_{run_id}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    no_fp16 = "" if use_float16 else "--no_use_float16"
    cmd = (
        f"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 "
        f"python inference.py "
        f"--inference_config {cfg_path} "
        f"--output_dir {run_out} "
        f"{no_fp16}"
    )

    progress(0, desc="Iniciando inferencia…")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=str(MIMIC_DIR),
        env={**os.environ, "PYTHONPATH": str(MIMIC_DIR)},
    )

    cfg_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise gr.Error(f"Error en inferencia:\n{result.stderr[-1500:]}")

    videos = sorted(run_out.rglob("*.mp4"), key=os.path.getmtime, reverse=True)
    if not videos:
        raise gr.Error("No se generó ningún video. Revisa los logs.")

    return str(videos[0]), result.stdout[-2000:]


# ── UI ──────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #666; margin-bottom: 1rem; }
.section-title { font-weight: 600; margin-top: 0.5rem; }
"""

with gr.Blocks(css=CSS, title="MimicMotion — Transferencia de Movimiento") as demo:

    gr.Markdown("# 🎬 MimicMotion", elem_id="title")
    gr.Markdown(
        "Anima una **foto de personaje** con el movimiento de un **video de referencia**.<br>"
        "Basado en [MimicMotion (Tencent)](https://github.com/tencent/MimicMotion).",
        elem_id="subtitle",
    )

    # ── Setup section ──
    with gr.Accordion("⚙️ Configuración inicial (descargar modelos)", open=False):
        setup_btn = gr.Button("Descargar / verificar modelos", variant="secondary")
        setup_log = gr.Textbox(label="Log", lines=10, interactive=False)
        setup_btn.click(fn=setup_models, outputs=setup_log)

    gr.Markdown("---")

    with gr.Row():
        # ── Left column: inputs ──
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Entradas")

            ref_image = gr.Image(
                label="Foto del personaje",
                type="filepath",
                image_mode="RGB",
            )
            ref_video = gr.Video(
                label="Video de referencia (movimiento a transferir)",
            )

            gr.Markdown("### 🎛️ Parámetros")

            with gr.Row():
                resolution = gr.Slider(256, 576, value=512, step=64, label="Resolución (px alto)")
                num_frames = gr.Slider(16, 72, value=48, step=8, label="Número de frames")

            with gr.Row():
                sample_stride = gr.Slider(1, 4, value=2, step=1, label="Sample stride")
                frames_overlap = gr.Slider(2, 12, value=6, step=2, label="Frames overlap")

            with gr.Row():
                num_inference_steps = gr.Slider(10, 50, value=25, step=5, label="Pasos de difusión")
                guidance_scale = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="Guidance scale")

            with gr.Row():
                fps = gr.Slider(8, 30, value=15, step=1, label="FPS salida")
                seed = gr.Number(value=42, label="Semilla (seed)", precision=0)

            with gr.Row():
                noise_aug_strength = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Noise aug strength")
                use_float16 = gr.Checkbox(value=True, label="Usar float16 (ahorra VRAM)")

            run_btn = gr.Button("🚀 Generar video", variant="primary", size="lg")

        # ── Right column: outputs ──
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Resultado")
            output_video = gr.Video(label="Video generado", interactive=False)
            log_box = gr.Textbox(label="Log de inferencia", lines=12, interactive=False)

    # ── Tips ──
    with gr.Accordion("💡 Consejos y referencia de parámetros", open=False):
        gr.Markdown(
            """
| Parámetro | Descripción | Valor típico |
|---|---|---|
| `resolution` | Alto del video de salida en píxeles | 512–576 |
| `num_frames` | Total de frames a generar (máx 72) | 48 |
| `sample_stride` | Cada cuántos frames se muestrea el video ref. | 2 |
| `frames_overlap` | Solapamiento entre segmentos para suavidad | 6 |
| `num_inference_steps` | Pasos del modelo de difusión | 25 |
| `guidance_scale` | Fuerza de seguimiento de la pose | 2.0 |
| `fps` | FPS del video de salida | 15 |
| `seed` | Semilla para reproducibilidad | 42 |

**Consejos:**
- La foto debe mostrar el **cuerpo completo** con fondo limpio.
- El video de referencia debe mostrar movimientos **bien iluminados** y cuerpo completo.
- Si hay **OOM (out of memory)**: reduce `resolution` a 256–384 o `num_frames` a 16.
- `sample_stride=2` hace el video de salida más rápido; usa `1` para más fidelidad.
            """
        )

    run_btn.click(
        fn=run_inference,
        inputs=[
            ref_image, ref_video, resolution, num_frames, sample_stride,
            frames_overlap, num_inference_steps, guidance_scale,
            noise_aug_strength, fps, seed, use_float16,
        ],
        outputs=[output_video, log_box],
    )

# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue(max_size=3).launch(server_name="0.0.0.0", server_port=7860)
