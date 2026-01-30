import torch
import soundfile as sf
import os
import sys
from transformers import pipeline

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: No se encontró 'qwen_tts'.")
    sys.exit()

# ======================================================
# 1. CONFIGURACIÓN Y CARGA
# ======================================================
# Usamos el modelo BASE para clonación zero-shot
model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" 
device = "cuda" if torch.cuda.is_available() else "cpu"

attn_impl = "eager" # Seguro para tu AMD y la T4
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Cargando modelo base para clonación: {model_name}...")

model = Qwen3TTSModel.from_pretrained(
    model_name,
    device_map="auto",
    dtype=dtype,
    attn_implementation=attn_impl,
)



# Cargar Whisper (puedes usar el modelo 'tiny' o 'base' para que sea rápido)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

def get_ref_text(audio_path):
    print("Transcribiendo audio de referencia automáticamente...")
    result = transcriber(audio_path)
    return result["text"]

# Uso en tu script:
ref_audio_path = "voces/Raul.wav"
ref_audio_text = get_ref_text(ref_audio_path)
print(f"Texto detectado: {ref_audio_text}")

if not os.path.exists(ref_audio_path):
    print(f"❌ Error: No encontré el archivo {ref_audio_path}")
    sys.exit()

print(f"Analizando voz de referencia: {ref_audio_path}...")

# Ahora ya puedes crear el prompt sin errores
voice_clone_prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio_path,
    ref_text=ref_audio_text
)

# ======================================================
# 3. GENERACIÓN CON TU VOZ CLONADA
# ======================================================
target_text = "¡Increíble! Ahora estoy hablando con la voz de Raúl gracias a Qwen 3 TTS. El proceso de clonación ha sido un éxito."

print("Generando audio clonado...")

with torch.no_grad():
    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="Spanish", # Forzamos Spanish para mejor acento
        voice_clone_prompt=voice_clone_prompt
    )

# ======================================================
# 4. GUARDADO
# ======================================================
output_path = "resultado_clonacion_raul.wav"
audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

sf.write(output_path, audio_data, sr)

print(f"--- LISTO ---")
print(f"Escucha el resultado en: {output_path}")