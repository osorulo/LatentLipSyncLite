import torch
import gc
import os
import uuid
from transformers import pipeline

DEFAULT_MODEL = "openai/whisper-base"


class STTService:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._pipeline = None
        self._current_model = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def unload(cls):
        if cls._instance is not None:
            cls._instance._unload_models()

    def _unload_models(self):
        self._pipeline = None
        self._current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, model_name):
        if self._current_model == model_name and self._pipeline is not None:
            return self._pipeline

        if model_name == DEFAULT_MODEL:
            try:
                from tts_service import TTSService
                tts = TTSService.get()
                shared = tts.get_whisper()
                if shared is not None:
                    self._pipeline = shared
                    self._current_model = DEFAULT_MODEL
                    return self._pipeline
            except Exception:
                pass

        self._unload_models()
        print(f"--- Cargando modelo Whisper STT: {model_name} ---")
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=self.device,
        )
        self._current_model = model_name
        return self._pipeline

    def transcribe(self, audio_path, model_name=DEFAULT_MODEL, language="auto",
                   return_timestamps=False):
        asr = self._load(model_name)

        kwargs = {"return_timestamps": True}
        if language and language != "auto":
            kwargs["language"] = language

        with torch.no_grad():
            result = asr(audio_path, **kwargs)

        segments = result.get("chunks")
        text = result.get("text", "")
        return {"text": text, "segments": segments}

    @staticmethod
    def save_text(text):
        os.makedirs("temp", exist_ok=True)
        out_path = os.path.abspath(f"temp/stt_{uuid.uuid4().hex}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return out_path

    @staticmethod
    def save_srt(segments, text_without_timestamps=None):
        os.makedirs("temp", exist_ok=True)
        out_path = os.path.abspath(f"temp/stt_{uuid.uuid4().hex}.srt")
        lines = []
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            seg_text = seg.get("text", "").strip()

            def fmt(secs):
                h = int(secs // 3600)
                m = int((secs % 3600) // 60)
                s = int(secs % 60)
                ms = int((secs - int(secs)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            lines.append(str(i))
            lines.append(f"{fmt(start)} --> {fmt(end)}")
            lines.append(seg_text)
            lines.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_path