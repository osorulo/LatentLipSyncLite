import torch
import gc
from utils.ffmpeg import get_fps, convert_to_25fps
from pathlib import Path
import subprocess
import os
import math
import shutil

class LipSyncService:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = None
        self.current_ckpt = None
        self.current_config = None

    def load_model(self, ckpt_path, config_path):
        from client import LipSyncInference
        
        if (self.model is None or 
            self.current_ckpt != ckpt_path or 
            self.current_config != config_path):
            
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                gc.collect()

            self.model = LipSyncInference(
                inference_ckpt_path=ckpt_path,
                enable_deepcache=True,
                unet_config_path=config_path,
                seed=1247
            )
            self.current_ckpt = ckpt_path
            self.current_config = config_path
            return True
        
        return False
    
    def get_duration(self, path: str) -> float:
        return float(subprocess.check_output(
            f'ffprobe -v error -show_entries format=duration '
            f'-of default=noprint_wrappers=1:nokey=1 "{path}"',
            shell=True
        ))
    
    def make_pingpong_video(
        self,
        input_video: str,
        target_duration: float,
        output_video: str,
        temp_dir: str
    ):
        input_video = os.path.abspath(input_video)
        output_video = os.path.abspath(output_video)
        temp_dir = os.path.abspath(temp_dir)

        base_dur = self.get_duration(input_video)
        if base_dur <= 0:
            raise RuntimeError("Duración de video inválida")

        normal = os.path.abspath(os.path.join(temp_dir, "pp_normal.mp4"))
        reverse = os.path.abspath(os.path.join(temp_dir, "pp_reverse.mp4"))
        concat_list = os.path.abspath(os.path.join(temp_dir, "pingpong_list.txt"))

        # video normal
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-r", "25",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            normal
        ], check=True)

        # video reverso
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", "reverse,gblur=sigma=0.3",
            "-r", "25",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            reverse
        ], check=True)

        single = base_dur * 2
        repeats = int(math.ceil(target_duration / single))

        with open(concat_list, "w") as f:
            for _ in range(repeats):
                f.write(f"file '{normal}'\n")
                f.write(f"file '{reverse}'\n")

        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-t", str(target_duration),
            "-r", "25",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ], check=True)

    def build_phrase_chunks(self,segments, chunk_seconds):
        chunks = []
        current = []
        t_start = segments[0]["start"]

        for seg in segments:
            current.append(seg)
            duration = seg["end"] - t_start

            if duration >= chunk_seconds:
                chunks.append(current)
                current = []
                t_start = seg["end"]

        if current:
            chunks.append(current)

        return chunks
    
    def add_micro_fade(self, input_video, output_video, fade_duration=0.05):
        dur = self.get_duration(input_video)

        if dur <= fade_duration:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", input_video,
                "-c:v", "copy",
                output_video
            ], check=True)
            return

        fade_start = dur - fade_duration

        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", f"fade=t=out:st={fade_start}:d={fade_duration}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ], check=True)

    def concat_with_xfade(
        self,
        videos: list[str],
        output_path: str,
        fps: int = 25,
        fade_duration: float = 0.05
    ):
        if len(videos) == 1:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", videos[0],
                "-c:v", "copy",
                output_path
            ], check=True)
            return

        durations = [self.get_duration(v) for v in videos]

        inputs = []
        for v in videos:
            inputs += ["-i", v]

        filter_lines = []

        for i in range(len(videos)):
            filter_lines.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

        current_label = "v0"
        current_time = durations[0]

        for i in range(1, len(videos)):
            offset = current_time - fade_duration
            next_label = f"x{i}"

            filter_lines.append(
                f"[{current_label}][v{i}]"
                f"xfade=transition=fade:duration={fade_duration}:offset={offset}"
                f"[{next_label}]"
            )

            current_label = next_label
            current_time += durations[i] - fade_duration

        filter_complex = ";".join(filter_lines)

        subprocess.run([
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{current_label}]",
            "-r", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True)

    def _progress(self, progress, phase, value, desc=""):
        if progress is None:
            return
        start, end = phase
        progress(start + (end - start) * value, desc=desc)

    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        steps: int,
        guidance: float,
        temp_dir: str,
        progress=None,
        upscale: bool = False,
        ckpt=None,
        config=None,
        duration = 60.0
    ) -> str:
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)
        output_path = os.path.abspath(output_path)

        chunks_dir = os.path.join(temp_dir, "chunks")
        lip_dir = os.path.join(temp_dir, "latentlipsync")

        os.makedirs(chunks_dir, exist_ok=True)
        os.makedirs(lip_dir, exist_ok=True)

        CHUNK_SECONDS = duration

        done_units = 0

        def tick(desc=""):
            nonlocal done_units
            done_units += 1
            if progress is not None:
                progress(done_units / total_units, desc=desc)

        try:
            audio_dur = self.get_duration(audio_path)

            segments = self.model.whisper_transcribe(audio_path)
            phrase_chunks = self.build_phrase_chunks(segments, CHUNK_SECONDS)
            phrase_chunks = [
                b for b in phrase_chunks
                if b[-1]["end"] - b[0]["start"] > 0.1
            ]
               

            total_units = 0
            total_units += 1                  # load model
            """if will_pingpong:
                total_units += 1              # pingpong"""
            total_units += 1                  # whisper
            total_units += len(phrase_chunks) # lipsync frases
            total_units += 1                  # concat final
            if upscale:
                total_units += 1              # upscale

            self.load_model(ckpt, config)
            tick("Modelo cargado")

            """if will_pingpong:
                pingpong_video = os.path.join(temp_dir, "video_pingpong.mp4")
                self.make_pingpong_video(
                    input_video=video_path,
                    target_duration=audio_dur,
                    output_video=pingpong_video,
                    temp_dir=temp_dir
                )
                #video_path = pingpong_video
                video_dur = self.get_duration(pingpong_video)
                tick("Procesando video ....")

            total_dur = min(video_dur, audio_dur)"""

            total_dur = audio_dur

            tick("Procesando video...")

            processed_videos = []
            phrase_weight = 1.0 / total_units

            for i, block in enumerate(phrase_chunks):
                start_t = block[0]["start"]
                end_t = block[-1]["end"]

                if start_t >= total_dur:
                    break

                length = min(end_t - start_t, total_dur - start_t)
                if length <= 0.1:
                    done_units += 1
                    continue

                base = done_units / total_units
                span = phrase_weight

                desc = f"LipSync {i+1}/{len(phrase_chunks)}"

                v_chunk = os.path.join(chunks_dir, f"video_{i}.mp4")
                a_chunk = os.path.join(chunks_dir, f"audio_{i}.wav")
                out_chunk = os.path.join(lip_dir, f"out_{i}.mp4")

                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-r", "25",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    v_chunk
                ], check=True)

                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(start_t), "-t", str(length),
                    "-i", audio_path,
                    a_chunk
                ], check=True)

                self.model.start_sync(
                    video_path=v_chunk,
                    audio_path=a_chunk,
                    video_out_path=out_chunk,
                    inference_steps=steps,
                    guidance_scale=guidance,
                    temp_dir=lip_dir,
                    progress=progress,
                    base=base,
                    span=span,
                    desc=desc
                )

                processed_videos.append(out_chunk)

                done_units += 1

                torch.cuda.empty_cache()
                gc.collect()

            if not processed_videos:
                raise RuntimeError("No se generaron segmentos sincronizados")

            xfade_video = os.path.join(temp_dir, "video_xfade.mp4")

            self.concat_with_xfade(
                processed_videos,
                xfade_video,
                fps=25,
                fade_duration=0.05
            )

            tick("Preparando resultado....")

            subprocess.run([
                "ffmpeg", "-y",
                "-i", xfade_video,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ], check=True)

            if upscale:
                upscaled = output_path.replace(".mp4", "_up.mp4")
                self.model._upscale(
                    in_video=output_path,
                    out_video=upscaled,
                    audio_path=audio_path,
                    progress=progress,
                    base=done_units / total_units,
                    span=1.0 / total_units
                )
                output_path = upscaled
                tick("Mejorando rostro")

            if progress is not None:
                progress(1.0, desc="Finalizado")

            shutil.rmtree(temp_dir)

            return output_path

        except Exception as e:
            raise RuntimeError(f"❌ Error en sync por frases: {e}")


    def sync_original(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        steps: int,
        guidance: float,
        temp_dir: str,
        progress=None,
        upscale: bool = False,
        ckpt=None, config=None
    ) -> str:
        
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_path = str(output_path)

        try:
            self.load_model(ckpt, config)
            # ==================================================
            # 0️⃣ Asegurar 25 FPS
            # ==================================================
            if progress is not None:
                progress(0.15, desc="Analizando FPS")

            fps = get_fps(video_path)

            if abs(fps - 25.0) > 0.01:
                fixed_video = (
                    Path(video_path).with_suffix("").as_posix() + "_25fps.mp4"
                )

                if progress is not None:
                    progress(0.20, desc="Convirtiendo a 25 FPS")

                convert_to_25fps(video_path, fixed_video)
                video_path = fixed_video

            # ==================================================
            # 1️⃣ LipSync
            # ==================================================

            LIP_SYNC_START = 0.25
            LIP_SYNC_END = 0.85 if upscale else 0.95
            UPScale_START = 0.85

            self.model.start_sync(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_path,
                inference_steps=steps,
                guidance_scale=guidance,
                temp_dir=temp_dir,
                progress=progress,
                base=LIP_SYNC_START,
                span=(LIP_SYNC_END - LIP_SYNC_START)
            )

            # ==================================================
            # 2️⃣ Upscale opcional (rostro)
            # ==================================================
            if upscale:
                upscaled_path = output_path.replace(".mp4", "_up.mp4")

                self.model._upscale(
                    in_video=output_path,
                    out_video=upscaled_path,
                    audio_path = audio_path,
                    base=UPScale_START,
                    span=0.15,
                    progress=progress
                )

                output_path = upscaled_path

            # ==================================================
            # Limpieza GPU / RAM
            # ==================================================
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            gc.collect()

            if progress is not None:
                progress(1.0, desc="Finalizado")

            return output_path

        except Exception as e:
            raise RuntimeError(f"❌ Error en sync(): {e}")
