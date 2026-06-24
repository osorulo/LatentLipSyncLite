import os
import cv2
import torch
import numpy as np
import subprocess
import gc
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

class MejoraService:
    def __init__(self):
        self.restorer = None
        self.restorer2 = None
        self.face_net = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _upscale(self, in_video, out_video, base=0.1, span=0.9, upscale_factor=1, progress=None):
        # 1. Inicialización de modelos
        if self.restorer2 is None:
            if progress is not None: progress(base, desc="Cargando modelos de IA...")
            
            # RealESRGAN para fondo y resolución general
            model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.restorer = RealESRGANer(
                scale=2, 
                model_path="checkpoints/RealESRGAN_x2plus.pth",
                model=model_esrgan, 
                half=(self.device == 'cuda'), 
                device=self.device
            )
            
            # GFPGAN para restauración facial
            self.restorer2 = GFPGANer(
                model_path="checkpoints/GFPGANv1.4.pth",
                upscale=1, 
                arch="clean", 
                channel_multiplier=2, 
                device=self.device
            )

            # Detector de rostros
            proto = "checkpoints/deploy.prototxt"
            model_ssd = "checkpoints/res10_300x300_ssd_iter_140000.caffemodel"
            self.face_net = cv2.dnn.readNetFromCaffe(proto, model_ssd)

        # 2. Configuración de Video
        video_input_str = str(in_video)
        cap = cv2.VideoCapture(video_input_str)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            raise Exception("No se pudo leer el video. Verifica la ruta.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # IMPORTANTE: Definir el tamaño de salida (Original x 2)
        actual_scale = upscale_factor if upscale_factor > 0 else 1
        output_width = width * actual_scale
        output_height = height * actual_scale

        # Archivo temporal para video mudo
        temp_no_audio = str(out_video).replace(".mp4", "_temp_no_audio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_no_audio, fourcc, fps, (output_width, output_height))

        # 3. Preparación de Máscara para el rostro
        target_size = 512
        center = (target_size // 2, target_size // 2)
        mask = np.zeros((target_size, target_size, 3), dtype=np.float32)
        cv2.ellipse(mask, center, (int(target_size * 0.35), int(target_size * 0.45)), 0, 180, 360, (0.5, 0.5, 0.5), -1)
        cv2.ellipse(mask, center, (int(target_size * 0.35), int(target_size * 0.45)), 0, 0, 180, (1.0, 1.0, 1.0), -1)
        
        blur_size = int(target_size * 0.15) | 1
        cached_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        inv_mask = 1.0 - cached_mask

        face_box = None
        current_frame = 0

        # 4. Bucle de Procesamiento
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Detección de rostro
                if face_box is None:
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.face_net.setInput(blob)
                    detections = self.face_net.forward()
                    for j in range(detections.shape[2]):
                        if detections[0, 0, j, 2] > 0.5:
                            box = detections[0, 0, j, 3:7] * np.array([width, height, width, height])
                            face_box = box.astype(int)
                            break

                # Mejora Facial
                if face_box is not None:
                    x1, y1, x2, y2 = [max(0, coord) for coord in face_box]
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        roi_h, roi_w = roi.shape[:2]
                        roi_input = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                        _, _, enhanced = self.restorer2.enhance(roi_input, weight=0.25)
                        merged = (enhanced.astype(np.float32) * cached_mask + roi_input.astype(np.float32) * inv_mask)
                        final_roi = cv2.resize(merged, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                        frame[y1:y2, x1:x2] = final_roi.astype(np.uint8)

                
                if upscale_factor > 0:
                    # Aplica RealESRGAN a todo el frame (lento)
                    up, _ = self.restorer.enhance(frame, outscale=upscale_factor)
                else:
                    # Mantiene el frame original con la mejora de rostro ya aplicada (rápido)
                    up = frame
                out.write(up)

                # Reporte de progreso
                if progress is not None and current_frame % 5 == 0:
                    self._report_progress(progress, base, span, current_frame / total_frames, f"Procesando: {current_frame}/{total_frames}")
                
                current_frame += 1

        finally:
            cap.release()
            out.release()

        # 5. Fusión Final de Audio con FFmpeg
        if progress is not None:
            progress(base + span, desc="Finalizando video con audio...")

        try:
            # Comando optimizado:
            # -c:v libx264 codifica a H264 (estándar compatible)
            # -pix_fmt yuv420p asegura compatibilidad con navegadores/móviles
            # -c:a aac copia o convierte el audio original a un formato estándar
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_no_audio,
                '-i', video_input_str,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                str(out_video)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error de FFmpeg: {result.stderr}")
                # Si falla el audio, intentamos salvar el video procesado
                os.rename(temp_no_audio, out_video)
            else:
                if os.path.exists(temp_no_audio):
                    os.remove(temp_no_audio)

        except Exception as e:
            print(f"Error al procesar audio: {e}")
            if os.path.exists(temp_no_audio):
                os.rename(temp_no_audio, out_video)

        torch.cuda.empty_cache()
        gc.collect()

    def _report_progress(self, progress, base, span, value, desc):
        if progress is None: return
        clamped_value = max(0, min(1, value))
        progress_total = base + (span * clamped_value)
        progress(progress_total, desc=desc)