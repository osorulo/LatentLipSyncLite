import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# --- Tu Clase Detectora (con correcciones menores para que sea funcional) ---
INSIGHTFACE_DETECT_SIZE = 512
LMK_ADAPT_ORIGIN_ORDER = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17, 43, 48, 49, 51, 50, 102, 103, 104, 105, 101, 73, 74, 86]

def cuda_to_int(cuda_str: str) -> int:
    if cuda_str == "cuda": return 0
    device = torch.device(cuda_str)
    return device.index if device.index is not None else 0

class FaceDetector:
    def __init__(self, device="cuda"):
        # Asegúrate de tener los modelos en 'checkpoints/auxiliary'
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"],
        )
        ctx = cuda_to_int(device) if torch.cuda.is_available() else -1
        self.app.prepare(ctx_id=ctx, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape
        faces = self.app.get(frame)
        get_face_store = None
        max_size = 0

        if len(faces) == 0:
            return None, None
        
        for face in faces:
            bbox = face.bbox.astype(np.int_).tolist()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < 50 or h < 80: continue
            if w / h > 1.5 or w / h < 0.2: continue
            if face.det_score < threshold: continue
            
            size_now = w * h
            if size_now > max_size:
                max_size = size_now
                get_face_store = face

        if get_face_store is None:
            return None, None
        
        face = get_face_store
        lmk = np.round(face.landmark_2d_106).astype(np.int_)
        return face.bbox, lmk # Simplificado para validación de presencia

# --- Script de procesamiento ---

def procesar_video_insightface(input_path, output_path):
    # Inicializar detector (usará CUDA si tienes GPU)
    detector = FaceDetector(device="cuda" if torch.cuda.is_available() else "cpu")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Codec MP4 estándar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Iniciando detección con InsightFace...")
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Usamos tu clase detectora
        # Solo nos interesa saber si devuelve algo distinto a None
        bbox, _ = detector(frame)

        if bbox is not None:
            out.write(frame)
            saved_count += 1
        
        if frame_count % 10 == 0:
            print(f"Frames procesados: {frame_count} | Guardados: {saved_count}", end="\r")

    cap.release()
    out.release()
    print(f"\nProceso completado.")
    print(f"Video final: {output_path} ({saved_count} frames)")

# Ejecutar
if __name__ == "__main__":
    procesar_video_insightface("video.mp4", "salida_limpia.mp4")