import gradio as gr
import shutil
import argparse
import sys
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Usar Google Drive para cache persistente")
parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                    help="Host para Gradio (default: 127.0.0.1, env GRADIO_SERVER_NAME)")
parser.add_argument("--port", type=int,
                    default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                    help="Puerto para Gradio (default: 7860, env GRADIO_SERVER_PORT)")
parser.add_argument("--share", action="store_true",
                    default=os.environ.get("GRADIO_SHARE", "0") == "1",
                    help="Generar link publico de Gradio (env GRADIO_SHARE=1)")
args, _ = parser.parse_known_args()
sys.argv = sys.argv[:1]

if args.colab:
    BASE_VOZ = "/content/drive/MyDrive/LatentLipSyncLite"
    os.environ["VOCES_DIR"] = os.path.join(BASE_VOZ, "voces")
    os.makedirs(os.environ["VOCES_DIR"], exist_ok=True)

import torch
import gc

from stt_service import STTService, DEFAULT_MODEL as DEFAULT_STT_MODEL
from utils.files import create_run_dir
from utils.validation import validate_file
from utils.ffmpeg import extract_audio
from utils.paths import VOCES_DIR, CHECKPOINTS_DIR, CONFIGS_DIR
from lipsync import LipSyncService


def cleanup_models():
    LipSyncService.unload()
    STTService.unload()
    from tts_service import TTSService
    TTSService.unload()
    from mejorar import MejoraService
    MejoraService.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def list_local_voices():
    path = Path(VOCES_DIR)
    path.mkdir(exist_ok=True)
    locals = [p.name for p in path.glob("*.wav")]
    predefined = ["Sohee", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna"]
    return predefined + locals

def refresh_voices():
    gr.Info("Voces actualizadas")
    return gr.Dropdown(choices=list_local_voices())

EMOTIONS = [
    ("Natural", "Natural"),
    ("😊 Feliz", "Speak in a happy and cheerful tone"),
    ("😢 Triste", "Speak in a sad and melancholic tone"),
    ("😠 Enojado", "Speak in an angry and aggressive tone"),
    ("😲 Sorprendido", "Speak in a surprised tone"),
    ("🤔 Pensativo", "Speak in a thoughtful and pensive tone"),
    ("🤫 Susurro", "Speak in a whisper"),
    ("🥳 Emocionado", "Speak in an excited and enthusiastic tone"),
    ("😴 Cansado", "Speak in a tired and sleepy tone"),
    ("🎤 Formal", "Speak in a formal and professional tone"),
    ("✏️ Personalizado", "CUSTOM"),
]

def process_tts(text, voice, ref_text, emotion, custom_instruct):
    cleanup_models()
    from tts_service import TTSService
    if not text: raise gr.Error("Escribe un texto")
    instruct = custom_instruct if emotion == "CUSTOM" else emotion
    service = TTSService.get()
    out_path, detected_text = service.generate(text, voice, ref_text=ref_text, instruct=instruct)
    return out_path, gr.update(value=out_path, visible=True)

def list_checkpoints():
    paths = list(CHECKPOINTS_DIR.glob("*.pt"))
    return [str(p) for p in paths]

def list_configs():
    paths = list((CONFIGS_DIR / "unet").glob("*.yaml"))
    return [str(p) for p in paths]

def get_model_choices():
    checkpoints = [str(p) for p in CHECKPOINTS_DIR.glob("*.pt")]
    configs = [str(p) for p in (CONFIGS_DIR / "unet").glob("*.yaml")]
    return checkpoints, configs

def refresh_models():
    ckpts, cfgs = get_model_choices()
    gr.Info("Listado de modelos actualizado")
    return gr.Dropdown(choices=ckpts), gr.Dropdown(choices=cfgs)

def suggest_config(checkpoint_path):
    if not checkpoint_path:
        return gr.Dropdown(value=None)
    ckpt_name = Path(checkpoint_path).name.lower()
    configs = list((CONFIGS_DIR / "unet").glob("*.yaml"))
    for c in configs:
        if "stage2_512" in c.name and "efficient" not in c.name and "v16" not in c.name:
            return gr.Dropdown(value=str(c))
    for c in configs:
        if "stage2" in c.name and "efficient" not in c.name:
            return gr.Dropdown(value=str(c))
    return gr.Dropdown()

def get_video_path(video_input):
    if isinstance(video_input, dict):
        return video_input.get("path")
    return video_input

def mejora(video_file, gfpgan_weight, upscale_factor, progress=gr.Progress()):
    cleanup_models()
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
    duration=60.0,
    progress=gr.Progress()
):
    cleanup_models()
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
            duration=duration
        )

        progress(1.0, desc="Finalizado")
        return final_video

    except Exception as e:
        raise gr.Error(f"Error: {e}")

def process_stt(audio_input, video_input, model_name, language, timestamps, subtitles, progress=gr.Progress()):
    cleanup_models()
    audio_path = audio_input
    video_path = get_video_path(video_input)

    if not audio_path and not video_path:
        raise gr.Error("Sube un audio o un video para transcribir")

    try:
        if not audio_path:
            validate_file(video_path, [".mp4", ".mov", ".mkv", ".avi", ".webm"])
            run_id, run_dir, _ = create_run_dir()
            audio_in = run_dir / "audio.wav"
            progress(0.1, desc="Extrayendo audio del video...")
            extract_audio(video_path, str(audio_in))
            audio_path = str(audio_in)
        else:
            validate_file(audio_path, [".wav", ".mp3", ".flac", ".ogg", ".m4a"])

        progress(0.3, desc=f"Cargando modelo {model_name}...")
        service = STTService.get()

        progress(0.5, desc="Transcribiendo...")
        result = service.transcribe(
            audio_path=audio_path,
            model_name=model_name,
            language=language,
            return_timestamps=timestamps,
        )

        text = result["text"]
        segments = result.get("segments")

        if timestamps and segments:
            lines = []
            for seg in segments:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                seg_text = seg.get("text", "").strip()
                lines.append(f"[{start:07.2f} -> {end:07.2f}] {seg_text}")
            text_out = "\n".join(lines)
        else:
            text_out = text

        progress(0.95, desc="Guardando...")
        txt_path = STTService.save_text(text_out)
        srt_path = None
        if subtitles and timestamps and segments:
            srt_path = STTService.save_srt(segments)
            gr.Info(f"Transcripción + subtítulos completada ({len(text)} caracteres)")
        else:
            gr.Info(f"Transcripción completada ({len(text)} caracteres)")

        progress(1.0, desc="Finalizado")
        return text_out, gr.update(value=txt_path, visible=True), gr.update(value=srt_path, visible=srt_path is not None)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise gr.Error(f"Error en la transcripción: {str(e)}")

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
                steps = gr.Slider(10, 50, value=20, label="Steps", info="Más = mejor sync, más lento")
                guidance = gr.Slider(1.0, 3.0, value=1.5, label="Guidance", info="Más = sync más fuerte, puede saturar")
                duration = gr.Slider(10.0, 60.0, value=60.0, label="Seconds")



            with gr.Row():
                with gr.Column():   
                    v_input = gr.Video(label="Sube tu Video", height=400)
                with gr.Column():
                    v_output = gr.Video(label="Resultado Final", height=400)

            lipsync_audio_input = gr.Audio(
                label="🎧 Audio (sube uno o genera con TTS)",
                type="filepath"
            )

            run_btn = gr.Button("SYNC VIDEO", variant="primary")

        with gr.TabItem("🎙️ TTS"):
            gr.Markdown("### Generador de Voz")

            tts_text = gr.Textbox(label="Texto", lines=4)
            with gr.Row():
                voice_select = gr.Dropdown(
                    label="Voz",
                    choices=list_local_voices(),
                    value="Sohee"
                )
                refresh_voices_btn = gr.Button("🔄", size="sm")
            with gr.Row():
                emotion_select = gr.Dropdown(
                    label="Emoción / Estilo",
                    choices=[e[0] for e in EMOTIONS],
                    value="Natural"
                )
                custom_instruct = gr.Textbox(
                    label="Instrucción personalizada",
                    placeholder="Ej: habla muy rápido y emocionado",
                    visible=False,
                    scale=2
                )
            ref_text_input = gr.Textbox(label="Texto Ref (opcional)")
            tts_btn = gr.Button("GENERAR AUDIO", variant="primary")

            tts_audio_output = gr.Audio(
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

        with gr.TabItem("📝 Audio a Texto"):
            gr.Markdown("### Transcripción con Whisper")

            with gr.Row():
                with gr.Column():
                    stt_audio_input = gr.Audio(
                        label="🎧 Audio (sube o graba)",
                        type="filepath"
                    )
                    stt_video_input = gr.Video(
                        label="🎬 Video (extrae pista de audio)",
                        height=300
                    )
                with gr.Column():
                    stt_model = gr.Dropdown(
                        choices=[
                            "openai/whisper-tiny",
                            "openai/whisper-base",
                            "openai/whisper-small",
                            "openai/whisper-medium",
                            "openai/whisper-large-v3",
                        ],
                        value=DEFAULT_STT_MODEL,
                        label="Modelo Whisper"
                    )
                    stt_language = gr.Dropdown(
                        choices=["auto", "es", "en", "fr", "de", "it", "pt", "ja", "zh", "ko"],
                        value="auto",
                        label="Idioma"
                    )
                    stt_timestamps = gr.Checkbox(
                        value=False,
                        label="Incluir timestamps (segmentos con tiempos)"
                    )
                    stt_subtitles = gr.Checkbox(
                        value=True,
                        label="Generar .SRT (subtítulos)"
                    )
                    stt_btn = gr.Button("TRANSCRIBIR", variant="primary")

            stt_text_output = gr.Textbox(
                label="📝 Texto transcrito",
                lines=12,
                show_copy_button=True
            )
            with gr.Row():
                stt_file_output = gr.File(label="⬇️ Descargar .txt", visible=False)
                stt_srt_output = gr.File(label="⬇️ Descargar .srt", visible=False)
   
    def toggle_custom_instruct(choice):
        return gr.update(visible=choice == "✏️ Personalizado")

    emotion_select.change(
        fn=toggle_custom_instruct,
        inputs=[emotion_select],
        outputs=[custom_instruct]
    )

    tts_btn.click(
        fn=process_tts,
        inputs=[tts_text, voice_select, ref_text_input, emotion_select, custom_instruct],
        outputs=[tts_audio_output, audio_download]
    )

    refresh_voices_btn.click(
        fn=refresh_voices,
        inputs=[],
        outputs=[voice_select]
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
        inputs=[v_input, lipsync_audio_input, steps, guidance, ckpt_select, config_select, duration],
        outputs=v_output
    )

    refresh_btn.click(
        fn=refresh_models,
        inputs=[],
        outputs=[ckpt_select, config_select]
    )

    ckpt_select.change(
        fn=suggest_config,
        inputs=[ckpt_select],
        outputs=[config_select]
    )

    stt_btn.click(
        fn=process_stt,
        inputs=[stt_audio_input, stt_video_input, stt_model, stt_language, stt_timestamps, stt_subtitles],
        outputs=[stt_text_output, stt_file_output, stt_srt_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
