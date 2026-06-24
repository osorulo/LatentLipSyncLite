import torch
import gc
import os
import soundfile as sf
from pathlib import Path
from transformers import pipeline
from qwen_tts import Qwen3TTSModel
import uuid

import tts_service as _tts_module
_tts_module.VOCES_DIR = "voces"

VOCES_DIR = _tts_module.VOCES_DIR

class TTSService:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.current_model_type = None
        self.model = None
        self.whisper = None
        self.predefined_voices = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _unload_models(self):
        """Libera memoria VRAM"""
        self.model = None
        self.whisper = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, mode="custom"):
        """Carga el modelo óptimo según el modo"""
        if self.current_model_type == mode and self.model is not None:
            return # Ya cargado

        self._unload_models()
        print(f"--- Cargando modelo Qwen3 TTS Modo: {mode} ---")
        
        repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" if mode == "custom" else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
        self.model = Qwen3TTSModel.from_pretrained(
            repo,
            device_map="auto",
            dtype=self.dtype,
            attn_implementation="eager"
        )
        self.current_model_type = mode

    def get_whisper(self):
        if self.whisper is None:
            print("Cargando Whisper para transcripción...")
            self.whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=self.device)
        return self.whisper

    def generate(self, text, voice_name, ref_audio=None, ref_text=None):
        voice_key = Path(voice_name).stem
        is_custom = voice_key in self.predefined_voices
        mode = "custom" if is_custom else "base"
        self.load_model(mode)

        voces_dir = _tts_module.VOCES_DIR

        with torch.no_grad():
            if is_custom:
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language="Auto",
                    speaker=voice_key,
                    instruct="Natural"
                )
            else:
                audio_path = os.path.join(voces_dir, voice_name)

                if not ref_text:
                    asr = self.get_whisper()
                    ref_text = asr(audio_path)["text"]

                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    ref_text=ref_text
                )

                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Spanish",
                    voice_clone_prompt=prompt
                )

        os.makedirs("temp", exist_ok=True)
        output_path = os.path.abspath(
            f"temp/tts_{uuid.uuid4().hex}.wav"
        )

        audio_data = (
            wavs[0].cpu().numpy()
            if torch.is_tensor(wavs[0])
            else wavs[0]
        )

        sf.write(output_path, audio_data, sr)

        return output_path, ref_text
