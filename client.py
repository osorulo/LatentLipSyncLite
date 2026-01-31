import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.whisper.whisper import load_model
from DeepCache import DeepCacheSDHelper
import json
import subprocess
import cv2
import torch
import gc
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

class LipSyncInference:
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(
            self,
            inference_ckpt_path,
            enable_deepcache,
            unet_config_path,
            seed):
            
            self.pipeline = None
            self.restorer = None
            self.device = 'cuda'
            self.ffmpeg_loglevel = 'verbose'
            
            print(f"Loaded checkpoint path: {inference_ckpt_path}")

            # Check if the GPU supports float16
            is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
            self.dtype = torch.float16 if is_fp16_supported else torch.float32

            self.config = OmegaConf.load(unet_config_path)
            
            scheduler = DDIMScheduler.from_pretrained("configs")

            if self.config.model.cross_attention_dim == 768:
                whisper_model_path = "checkpoints/whisper/small.pt"
            elif self.config.model.cross_attention_dim == 384:
                whisper_model_path = "checkpoints/whisper/tiny.pt"
            else:
                raise NotImplementedError("cross_attention_dim must be 768 or 384")
            
            audio_encoder = Audio2Feature(
                model_path=whisper_model_path,
                device="cuda",
                num_frames=self.config.data.num_frames,
                audio_feat_length=self.config.data.audio_feat_length,
            )

            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0

            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(self.config.model),
                inference_ckpt_path,
                device="cpu",
            )

            unet = unet.to(dtype=self.dtype)

            self.pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet, 
                scheduler=scheduler,
            ).to("cuda")

            # use DeepCache
            if enable_deepcache:
                helper = DeepCacheSDHelper(pipe=self.pipeline)
                helper.set_params(cache_interval=3, cache_branch_id=0)
                helper.enable()

            if seed != -1:
                set_seed(seed)
            else:
                torch.seed()

            print(f"Initial seed: {torch.initial_seed()}")

    def start_sync(self,
                   video_path,
                   audio_path,
                   video_out_path,
                   inference_steps,
                   guidance_scale,
                   temp_dir,
                   progress=None,
                   base=0.1, 
                   span=0.75,
                   desc=""):

        self.pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=video_out_path,
            num_frames=self.config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=self.dtype,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
            mask_image_path=self.config.data.mask_image_path,
            temp_dir=temp_dir,
            progress=progress,
            base = base,
            span = span,
            desc=desc
        )

    def whisper_transcribe(self, audio_path: str):
        self.model_whisper = load_model("checkpoints/whisper/tiny.pt", self.device)
        result = self.model_whisper.transcribe_t(audio_path, language="es")
        return result["segments"]
    
    def _upscale(
        self,
        in_video,
        out_video,
        audio_path,
        base=0.85, 
        span=0.15,
        progress=None
    ):
        
        if(self.restorer == None):
                 # ESRGAN para super-resolución
                """model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.restorer = RealESRGANer(
                    scale=2, model_path="checkpoints/RealESRGAN_x2plus.pth",
                    model=model_esrgan, half=(self.device == 'cuda'), device=self.device
                )"""

                 # GFPGAN para restauración facial
                self.restorer2 = GFPGANer(
                    model_path="checkpoints/GFPGANv1.4.pth",
                    upscale=1, arch="clean", channel_multiplier=2, device=self.device
                )

                proto = "checkpoints/deploy.prototxt"
                model_ssd = "checkpoints/res10_300x300_ssd_iter_140000.caffemodel"
                self.face_net = cv2.dnn.readNetFromCaffe(proto, model_ssd)

        cap = cv2.VideoCapture(in_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_video_path = out_video.replace(".mp4", "_temp.mp4")

        out = cv2.VideoWriter(
            temp_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_w, frame_h)
        )

        face_box = None  # (x1, y1, x2, y2)
        i = 0

        # =========================================================

        # --- máscara (idéntica a la tuya) ---
        # --- máscara (igual a la tuya) ---
        target_size = 512
        center = (target_size // 2, target_size // 2)
        mask = np.zeros((target_size, target_size, 3), dtype=np.float32)

        cv2.ellipse(mask, center, (int(target_size * 0.35), int(target_size * 0.45)),
                    0, 180, 360, (0.5, 0.5, 0.5), -1)
        cv2.ellipse(mask, center, (int(target_size * 0.35), int(target_size * 0.45)),
                    0, 0, 180, (1.0, 1.0, 1.0), -1)

        blur_size = int(target_size * 0.15) | 1
        cached_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        inv_mask = 1.0 - cached_mask

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # =====================================================
            # Detectar cara SOLO la primera vez
            # =====================================================
            if face_box is None:
                blob = cv2.dnn.blobFromImage(
                    frame,
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0)
                )
                self.face_net.setInput(blob)
                detections = self.face_net.forward()

                h, w = frame.shape[:2]

                for j in range(detections.shape[2]):
                    conf = detections[0, 0, j, 2]
                    if conf > 0.5:
                        box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        face_box = (x1, y1, x2, y2)
                        break

            # =====================================================
            # Aplicar mejora SOLO a la cara detectada
            # =====================================================
            if face_box is not None:
                x1, y1, x2, y2 = face_box
                roi_h = y2 - y1
                roi_w = x2 - x1
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    roi_resized = cv2.resize(
                        roi, (target_size, target_size),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                   
                    with torch.amp.autocast(device_type="cuda"):
                        _, _, enhanced = self.restorer2.enhance(
                            roi_resized, weight=0.5
                        )

                    merged = (
                        enhanced.astype(np.float32) * cached_mask +
                        roi_resized.astype(np.float32) * inv_mask
                    )

                    final_roi = cv2.resize(
                        merged, (roi_w, roi_h),
                        interpolation=cv2.INTER_LINEAR
                    )

                    #up, _ = self.restorer.enhance(roi, outscale=2)
                    """up = cv2.resize(
                        up,
                        (x2 - x1, y2 - y1),
                        interpolation=cv2.INTER_CUBIC
                    )"""
                    frame[y1:y2, x1:x2] = final_roi

            out.write(frame)

            if progress is not None and i % 5 == 0:
                self._report_progress(
                    progress,
                    base=base,
                    span=span, # Dejamos un 10% del span para el merge de audio
                    value=i / total,
                    desc=f"Upscale {i}/{total}"
                )

            i += 1

        cap.release()
        out.release()

        try:
            if progress is not None:
                self._report_progress(progress, base=base + (span * 0.9), span=span * 0.1, value=1.0, desc="Finalizando...")
            
            self._merge_audio_video(audio_path, temp_video_path, out_video)
            
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                
        except Exception as e:
            print(f"Error al mezclar audio: {e}")
            os.rename(temp_video_path, out_video)

    def _merge_audio_video(self, audio_file: str, temp_video: str, outfile: str):        
        command = (
            f'ffmpeg -y -i "{temp_video}" -i "{audio_file}" '
            f'-c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p '
            f'-c:a aac -b:a 192k -threads 0 '
            f'-loglevel {self.ffmpeg_loglevel} "{outfile}"'
        )
        subprocess.run(command, shell=True, check=True)


    def _report_progress(self, progress, base, span, value, desc):
        if progress is None: return
        # value debe ser entre 0 y 1
        clamped_value = max(0, min(1, value))
        progress_total = base + (span * clamped_value)
        progress(progress_total, desc=desc)

    

       