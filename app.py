import gradio as gr
import shutil
from pathlib import Path

from utils.files import create_run_dir
from utils.validation import validate_file
from lipsync import LipSyncService
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Usar Google Drive para cache persistente")
args, _ = parser.parse_known_args()
sys.argv = sys.argv[:1]
VOCES_DIR = "voces"

if args.colab:
    BASE_VOZ = "/content/drive/MyDrive/LatentLipSyncLite"
    VOCES_DIR = os.path.join(BASE_VOZ, "voces")
    os.makedirs(VOCES_DIR, exist_ok=True)

    from tts_service import TTSService
    TTSService.VOCES_DIR = VOCES_DIR

def list_local_voices():
    path = Path(VOCES_DIR)
    path.mkdir(exist_ok=True)
    locals = [p.name for p in path.glob("*.wav")]
    predefined = ["Sohee", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna"]
    return predefined + locals

def process_tts(text, voice, ref_text):
    from tts_service import TTSService
    if not text: raise gr.Error("Escribe un texto")
    service = TTSService.get()
    out_path, detected_text = service.generate(text, voice, ref_text=ref_text)
    return out_path, gr.update(value=out_path, visible=True)

def list_checkpoints():
    paths = list(Path("checkpoints").glob("*.pt"))
    return [str(p) for p in paths]

def list_configs():
    paths = list(Path("configs/unet").glob("*.yaml"))
    return [str(p) for p in paths]

def get_model_choices():
    checkpoints = [str(p) for p in Path("checkpoints").glob("*.pt")]
    configs = [str(p) for p in Path("configs/unet").glob("*.yaml")]
    return checkpoints, configs

def refresh_models():
    ckpts, cfgs = get_model_choices()
    gr.Info("Listado de modelos actualizado")
    return gr.Dropdown(choices=ckpts), gr.Dropdown(choices=cfgs)

def get_video_path(video_input):
    if isinstance(video_input, dict):
        return video_input.get("path")
    return video_input

def mejora(video_file, gfpgan_weight, upscale_factor, progress=gr.Progress()):
    video_path = get_video_path(video_file)
    if not video_path:
        raise gr.Error("No se proporcionó un video válido.")

    if gfpgan_weight <= 0 and upscale_factor <= 0:
        raise gr.Error("Activa al menos una opción de mejora (GFPGAN o Upscale)")

    run_id, run_dir, WORKDIR = create_run_dir()
    video_in = run_dir / "input.mp4"
    video_out = str(WORKDIR / f"{run_id}_up_output.mp4")

    try:
        progress(0.05, desc="Preparando archivos...")
        shutil.copy(video_path, video_in)

        from mejorar import MejoraService
        service = MejoraService()

        service._upscale(
            in_video=video_in,
            out_video=video_out,
            gfpgan_weight=gfpgan_weight,
            upscale_factor=upscale_factor,
            progress=progress
        )
    
        return video_out

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise gr.Error(f"Error en el proceso: {str(e)}")

def process_sync(
    video_file,
    audio_file,
    steps,
    guidance,
    ckpt_dropdown,
    config_dropdown,
    duration = 60.0,
    progress=gr.Progress()
):
    video_path = get_video_path(video_file)
    audio_path = audio_file  

    if not video_path or not audio_path:
        raise gr.Error("Falta video o audio")

    validate_file(video_path, [".mp4", ".mov", ".mkv"])
    validate_file(audio_path, [".wav", ".mp3"])

    run_id, run_dir, WORKDIR = create_run_dir()

    video_in = run_dir / "input.mp4"
    audio_in = run_dir / "audio.wav"
    video_out = WORKDIR / f"{run_id}_output.mp4"

    try:
        progress(0.05, desc="Preparando archivos")

        shutil.copy(video_path, video_in)
        shutil.copy(audio_path, audio_in)

        progress(0.10, desc="Cargando Modelo....")
        service = LipSyncService.get()

        fue_carga_nueva = service.load_model(ckpt_dropdown, config_dropdown)

        nombre_modelo = Path(ckpt_dropdown).name
        if fue_carga_nueva:
            gr.Info(f"Modelo cargado: {nombre_modelo}")
        else:
            gr.Info(f"Usando modelo en memoria: {nombre_modelo}")

        final_video = service.sync(
            video_path=str(video_in),
            audio_path=str(audio_in),
            output_path=str(video_out),
            steps=steps,
            guidance=guidance,
            temp_dir=str(run_dir),
            progress=progress,
            ckpt=ckpt_dropdown,
            config=config_dropdown,
            duration = duration
        )

        progress(1.0, desc="Finalizado")
        return final_video

    except Exception as e:
        raise gr.Error(f"Error: {e}")

print("📦 VOCES_DIR =", VOCES_DIR)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 👄 LatentSyncLite")

    with gr.Tabs():
        with gr.TabItem("🎬 LipSync"):
            gr.Markdown("### Sincronización Labial")
            
            with gr.Accordion("Configuración de Modelos", open=False):
                with gr.Row():
                    ckpt_select = gr.Dropdown(
                        choices=list_checkpoints(),
                        label="Checkpoint"
                    )
                    config_select = gr.Dropdown(
                        choices=list_configs(),
                        label="Config"
                    )
                    refresh_btn = gr.Button("🔄")

            with gr.Row():
                steps = gr.Slider(10, 50, value=10, label="Steps")
                guidance = gr.Slider(1.0, 5.0, value=2.0, label="Guidance")
                duration = gr.Slider(10.0, 60.0, value=60.0, label="Seconds")

            run_btn = gr.Button("SYNC VIDEO", variant="primary")

            with gr.Row():
                with gr.Column():   
                    v_input = gr.Video(label="Sube tu Video", height=400)
                with gr.Column():
                    v_output = gr.Video(label="Resultado Final", height=400)

            gr.Audio(
                label="🎧 Audio (sube uno o genera con TTS)",
                type="filepath"
            )

        with gr.TabItem("🎙️ TTS"):
            gr.Markdown("### Generador de Voz")

            tts_text = gr.Textbox(label="Texto", lines=4)
            voice_select = gr.Dropdown(
                label="Voz",
                choices=list_local_voices(),
                value="Sohee"
            )
            ref_text_input = gr.Textbox(label="Texto Ref (opcional)")
            tts_btn = gr.Button("GENERAR AUDIO", variant="primary")

            audio_main = gr.Audio(
                label="🎧 Audio Generado",
                type="filepath"
            )

            audio_download = gr.File(label="⬇️ Descargar Audio", visible=False)

        with gr.TabItem("✨ Mejorar"):
            gr.Markdown("### Mejorar Video")

            gr.Markdown("**GFPGAN** restaura y mejora la calidad del rostro detectado en el video.")
            gfpgan_weight = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.0,
                label="GFPGAN (Restauración Facial)",
                info="0 = Sin mejora facial, 1.0 = Mejora máxima"
            )

            gr.Markdown("**RealESRGAN** escala el video al doble de resolución (2x).")
            upscale_factor = gr.Slider(
                minimum=0,
                maximum=2,
                step=2,
                value=0,
                label="Upscale (Resolución 2x)",
                info="0 = Sin upscale, 2 = Duplicar resolución"
            )

            with gr.Row():
                with gr.Column():
                    mejora_video_input = gr.Video(label="Sube tu Video para Mejorar", height=400)
                with gr.Column():
                    mejora_video_output = gr.Video(label="Video Mejorado", height=400)

            iniciar_mejora_btn = gr.Button("🚀 Iniciar Mejora", variant="primary")
   
    tts_btn.click(
        fn=process_tts,
        inputs=[tts_text, voice_select, ref_text_input],
        outputs=[audio_main, audio_download]
    )

    iniciar_mejora_btn.click(
        fn=mejora, 
        inputs=[mejora_video_input, gfpgan_weight, upscale_factor],
        outputs=[mejora_video_output]
    )

    def validate_audio(audio):
        if not audio:
            raise gr.Error("Primero sube o genera un audio")

    run_btn.click(
        fn=process_sync,
        inputs=[v_input, audio_main, steps, guidance, ckpt_select, config_select, duration],
        outputs=v_output
    )

    refresh_btn.click(
        fn=refresh_models,
        inputs=[],
        outputs=[ckpt_select, config_select]
    )

if __name__ == "__main__":
    demo.launch(share=True)
