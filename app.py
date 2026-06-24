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

def mejora(video_file, upscale_factor, progress=gr.Progress()):
    # Asumiendo que get_video_path y validate_file existen en tu utilidad
    video_path = get_video_path(video_file)
    if not video_path:
        raise gr.Error("No se proporcionó un video válido.")

    run_id, run_dir, WORKDIR = create_run_dir()
    video_in = run_dir / "input.mp4"
    # Aseguramos que video_out sea un string o Path compatible
    video_out = str(WORKDIR / f"{run_id}_up_output.mp4")

    try:
        progress(0.05, desc="Preparando archivos...")
        shutil.copy(video_path, video_in)

        # Import local para evitar colisiones
        from mejorar import MejoraService
        service = MejoraService()

        service._upscale(
            in_video=video_in,
            out_video=video_out,
            upscale_factor = upscale_factor,
            progress=progress
        )
    
        return video_out

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Para debug en consola
        raise gr.Error(f"Error en el proceso: {str(e)}")

def process_sync(
    video_file,
    audio_file,
    steps,
    guidance,
    mejorarES_chk,
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
            upscale=mejorarES_chk,
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
    gr.Markdown("## 👄 LatentSync + TTS")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎬 Lipsync")
           
    
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
                mejorarES_chk = gr.Checkbox(
                    label="Mejorar",
                    value=False,
                    visible=True
                )
            run_btn = gr.Button("SYNC VIDEO", variant="primary")

            with gr.Accordion("Configuración de Mejora", open=False, visible=False):
                with gr.Row():
                    with gr.Column():  
                        upscale_factor = gr.Slider(
                            minimum=0, 
                            maximum=2, 
                            step=1, 
                            value=1, 
                            label="Factor de Escalado (1x = Calidad, 2x = Resolución)"
                        )
                        
                        mejora_btn = gr.Button("Mejorar Calidad de video", variant="primary")

            with gr.Row():
                with gr.Column():   
                    v_input = gr.Video(label="Sube tu Video", height=500)
                with gr.Column():
                    v_output = gr.Video(label="Resultado Final", height=500)
           
        with gr.Column():
            gr.Markdown("### 🎙️ Generador de Voz (TTS)")
            tts_text = gr.Textbox(label="Texto", lines=3)
            voice_select = gr.Dropdown(
                label="Voz",
                choices=list_local_voices(),
                value="Sohee"
            )
            ref_text_input = gr.Textbox(label="Texto Ref (opcional)")
            tts_btn = gr.Button("GENERAR AUDIO", variant="primary")

            audio_main = gr.Audio(
                label="🎧 Audio (sube uno o genera con TTS)",
                type="filepath"
            )

            audio_download = gr.File(label="⬇️ Descargar Audio", visible=False)
   
    tts_btn.click(
        fn=process_tts,
        inputs=[tts_text, voice_select, ref_text_input],
        outputs=[audio_main, audio_download]
    )

    mejora_btn.click(
        fn=mejora, 
        inputs=[v_input, upscale_factor],
        outputs=[v_output]
        )

    def validate_audio(audio):
        if not audio:
            raise gr.Error("Primero sube o genera un audio")

    run_btn.click(
        fn=process_sync,
        inputs=[v_input, audio_main, steps, guidance, mejorarES_chk, ckpt_select, config_select, duration],
        outputs=v_output
    )

    refresh_btn.click(
        fn=refresh_models,
        inputs=[],
        outputs=[ckpt_select, config_select]
    )

if __name__ == "__main__":
    demo.launch(share=True)
