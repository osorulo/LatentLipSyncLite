FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Sistema: Python 3.10, ffmpeg, libs GUI, rclone
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils libpython3.10-dev \
        curl git ffmpeg unzip rclone \
        build-essential g++ \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libgomp1 ca-certificates \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-torch.txt requirements-py.txt ./
RUN pip install -r requirements-torch.txt
RUN pip install -r requirements-py.txt \
    && pip install -q gdown huggingface_hub
RUN sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/g' \
        /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py \
        /usr/local/lib/python3.10/dist-packages/basicsr/utils/img_util.py || true

COPY . /app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LLS_BASE_DIR=/app \
    CHECKPOINTS_DIR=/app/checkpoints \
    VOCES_DIR=/app/voces \
    CONFIGS_DIR=/app/configs \
    HF_HOME=/app/.hf_cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=1

EXPOSE 7860

ENTRYPOINT ["/entrypoint.sh"]