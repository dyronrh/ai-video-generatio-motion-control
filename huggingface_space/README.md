---
title: MimicMotion — Transferencia de Movimiento
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
hardware: gpu-t4-medium
---

# MimicMotion — Photo to Video Motion Transfer

Anima una **foto de un personaje** usando el movimiento de un **video de referencia**.

Basado en [MimicMotion (Tencent Research)](https://github.com/tencent/MimicMotion).

## Uso

1. Expande **"Configuración inicial"** y pulsa **"Descargar / verificar modelos"** la primera vez.
2. Sube la **foto del personaje** (cuerpo completo, fondo limpio).
3. Sube el **video de referencia** (movimiento a transferir).
4. Ajusta los parámetros si lo deseas.
5. Pulsa **"Generar video"** y espera unos minutos.

## Requisitos de hardware

| GPU | Resolución máx. recomendada | Frames máx. |
|-----|---------------------------|-------------|
| T4 (16 GB) | 512 | 48 |
| A10G (24 GB) | 576 | 72 |
| A100 (40 GB) | 576 | 72 |

## Parámetros principales

| Parámetro | Descripción |
|---|---|
| `resolution` | Alto del video de salida en px |
| `num_frames` | Frames totales (máx 72) |
| `sample_stride` | Intervalo de muestreo del video ref |
| `guidance_scale` | Fuerza de seguimiento de la pose |
| `num_inference_steps` | Pasos de difusión (calidad vs velocidad) |

## Créditos

- **MimicMotion**: Tencent Research — [paper](https://arxiv.org/abs/2406.19680) · [repo](https://github.com/tencent/MimicMotion)
- **SVD**: Stability AI — `stable-video-diffusion-img2vid-xt-1-1`
- **DWPose**: yzd-v
