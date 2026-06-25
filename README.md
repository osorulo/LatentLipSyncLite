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

## Rutas configurables

Las rutas a `checkpoints/`, `configs/` y `voces/` se leen de variables de entorno (con defaults que preservan el comportamiento local). Definidas en `utils/paths.py`:

| Variable        | Default                | Descripción                          |
|-----------------|------------------------|--------------------------------------|
| `LLS_BASE_DIR`  | `.` (cwd actual)       | Raíz del proyecto                    |
| `CHECKPOINTS_DIR`| `<base>/checkpoints`  | Pesos de modelos LatentSync/Whisper/GFPGAN/ESRGAN |
| `CONFIGS_DIR`   | `<base>/configs`       | Configs YAML del UNet + scheduler    |
| `VOCES_DIR`     | `<base>/voces`         | Samples de voz para cloning           |
| `HF_HOME`       | `<base>/.hf_cache`     | Cache de HuggingFace                 |
| `GRADIO_SERVER_NAME` | `127.0.0.1`       | Host de Gradio (`0.0.0.0` en Docker) |
| `GRADIO_SERVER_PORT`  | `7860`            | Puerto de Gradio                     |
| `GRADIO_SHARE`  | `0` (=disabled)        | `1` activa el tunel publico `gradio.live` |

Ejemplo:
```bash
CHECKPOINTS_DIR=/data/ckpts VOCES_DIR=/data/voces python app.py --share
```

Flags de `app.py`:
- `--colab`: usa Google Drive para `VOCES_DIR`.
- `--host`, `--port`, `--share`: sobrescriben las env vars de Gradio.

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

---

## Despliegue en RunPod con Docker + Cloudflare R2

La imagen Docker contiene **solo codigo y dependencias** (~3-4 GB). Los pesos (`checkpoints/`) y voces (`voces/`) se almacenan en un bucket **Cloudflare R2** y se descargan al arranque del contenedor (egress de R2 es gratis). Esto permite pagar solo por el uso de la GPU en RunPod sin almacenamiento persistente alli.

### 1) Preparar Cloudflare R2 (una sola vez)

1. Dashboard de Cloudflare → **R2 Object Storage** → crear bucket `lls-models`.
2. **Manage R2 API Tokens** → crear token con permiso `Object Read & Write` para ese bucket. Anota:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_ENDPOINT` (formato `https://<account_id>.r2.cloudflarestorage.com`)

### 2) Subir modelos a R2 (una sola vez)

Instala [rclone](https://rclone.org/install/) y configura `~/.config/rclone/rclone.conf`:

```ini
[r2]
type = s3
provider = Cloudflare
access_key_id = <R2_ACCESS_KEY_ID>
secret_access_key = <R2_SECRET_ACCESS_KEY>
endpoint = <R2_ENDPOINT>
```

Desde la raiz del proyecto (con `checkpoints/` y `voces/` ya poblados):

```bash
rclone copy checkpoints/ r2:lls-models/checkpoints/ --progress
rclone copy voces/       r2:lls-models/voces/       --progress
```

Para actualizar modelos despues: repite los comandos `rclone copy` (no hace falta rebuild de la imagen).

### 3) Construir y publicar la imagen en ghcr.io

```bash
docker build -t ghcr.io/osorulo/latentlipsynclite:latest .

# Necesitas un GitHub PAT con scope write:packages
echo $GITHUB_TOKEN | docker login ghcr.io -u osorulo --password-stdin
docker push ghcr.io/osorulo/latentlipsynclite:latest
```

La imagen es **publica**; RunPod podra tirar de ella sin credenciales extra.

### 4) Crear la plantilla en RunPod

RunPod → **Templates → New Template**:

| Campo              | Valor                                     |
|--------------------|-------------------------------------------|
| Container image    | `ghcr.io/osorulo/latentlipsynclite:latest`|
| Container Disk     | 30 GB                                     |
| Volume Disk        | **Ninguno**                               |
| Exposed HTTP ports | `7860`                                    |
| GPU                | RTX 4090 / A100 40GB                      |

Variables de entorno (marca las de R2 como **Secure**):

| Variable                | Valor                                              |
|-------------------------|----------------------------------------------------|
| `R2_ACCESS_KEY_ID`     | `<secret>`                                         |
| `R2_SECRET_ACCESS_KEY`  | `<secret>`                                         |
| `R2_ENDPOINT`           | `https://<account>.r2.cloudflarestorage.com`       |
| `R2_BUCKET`             | `lls-models`                                       |
| `GRADIO_SERVER_NAME`    | `0.0.0.0`                                          |
| `GRADIO_SERVER_PORT`    | `7860`                                             |
| `GRADIO_SHARE`          | `1`                                                |

### 5) Arrancar el pod

Al arrancar, el `entrypoint.sh`:
1. Configura rclone con las creds de R2.
2. Descarga `checkpoints/` (~6 GB) y `voces/` desde R2 al filesystem del contenedor (2-5 min).
3. Lanza `python app.py --share`.

URLs publicas:
- **Proxy RunPod**: `https://<pod-id>-7860.proxy.runpod.net`
- **Tunel Gradio**: link `*.gradio.live` que aparece en los logs del pod.

### Troubleshooting

| Sintoma | Causa / Solucion |
|---------|------------------|
| Boot lento | Normal: primer arranque descarga ~6 GB. R2 egress es gratis. |
| `rclone: error: could not find endpoint` | Revisa `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. |
| Puerto 7860 no accesible | Verifica `GRADIO_SERVER_NAME=0.0.0.0` y que el puerto `7860` este expuesto en la plantilla. |
| `CUDA out of memory` | Sube a una GPU con mas VRAM (A100 80GB) o reduce `duration` en la UI. |
| `basicsr` import error | El parche `sed` del Dockerfile fallo; revisa la version instalada. |
| Modelos no aparecen | Confirma que `R2_BUCKET` coincide con el bucket y que la subida termino (`rclone lsd r2:lls-models/`). |
