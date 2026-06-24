# LatentLipSyncLite

Aplicación Gradio para lip synchronization que combina LatentSync (modelo diffusion-based), Qwen3-TTS (síntesis de voz con voice cloning) y GFPGAN/RealESRGAN (mejora de video).

## Features

- **Lip Sync**: Sincronización labial de video a partir de audio
- **Text-to-Speech**: Generación de audio desde texto con Qwen3-TTS
- **Voice Cloning**: Clonación de voz usando samples de audio
- **Face Enhancement**: Restauración facial con GFPGAN
- **Video Upscaling**: Escalado 2x con RealESRGAN

## Requisitos

- Python 3.10+
- CUDA 12+ (NVIDIA) o ROCm (AMD)
- ~10GB VRAM recomendado

## Instalación

```bash
pip install -r requirements.txt
```

Para GPUs AMD:
```bash
pip install -r requirements_rocm.txt
```

## Modelos

Los checkpoints se cargan automáticamente desde `checkpoints/`:

- `latentsync_unet_1.5.pt` / `latentsync_unet_1.6.pt` - Modelos LatentSync
- `GFPGANv1.4.pth` - Restauración facial
- `RealESRGAN_x2plus.pth` - Escalado de video
- `whisper/tiny.pt` / `small.pt` - Transcripción de audio

## Uso

```bash
python app.py
```

Flags disponibles:
- `--colab`: Configuración para Google Colab
- `--share`: Generar link público de Gradio

## Estructura del Proyecto

```
LatentLipSyncLite/
├── app.py              # Interfaz Gradio principal
├── lipsync.py          # LipSyncService - orquestación de lip sync
├── client.py           # LipSyncInference - pipeline de inferencia
├── tts_service.py      # TTSService - síntesis de voz Qwen3-TTS
├── mejorar.py          # MejoraService - mejora de video
├── tts.py              # Script standalone de TTS
├── latentsync/         # Biblioteca core LatentSync
│   ├── models/         # UNet3D, attention, motion modules
│   ├── pipelines/      # LipsyncPipeline
│   └── whisper/        # Encoder de audio
├── qwen_tts/           # Integración Qwen3-TTS
├── gfpgan/             # Módulo de restauración facial
├── checkpoints/        # Pesos de modelos
├── configs/            # Configuraciones de modelo
├── voces/              # Samples de voz para cloning
└── utils/              # Utilidades (ffmpeg, validación)
```

## Servicios

| Servicio | Descripción |
|----------|-------------|
| `LipSyncService` | Procesa video+audio, chunking, sincronización por frases |
| `LipSyncInference` | Pipeline diffusion con DeepCache, Whisper, GFPGAN |
| `TTSService` | Carga modelo Qwen3-TTS, generación de voz, cloning |
| `MejoraService` | Detección facial, restauración GFPGAN, upscaling RealESRGAN |

## Formatos Soportados

| Tipo | Entrada | Salida |
|------|---------|--------|
| Video | `.mp4`, `.mov`, `.mkv` | `.mp4` (H.264, 25 FPS) |
| Audio | `.wav`, `.mp3` | `.wav` (16kHz) |
| Texto | String | Audio WAV |
