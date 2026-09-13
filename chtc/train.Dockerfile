# Training image for the ActionImages-Cogen fusion arm on CHTC GPU Lab.
#
#   docker build -t aicogen_train -f chtc/train.Dockerfile .
#   docker tag aicogen_train <dockerhub_user>/aicogen_train:fusion
#   docker push <dockerhub_user>/aicogen_train:fusion
#
# Pinned to the versions the arm was actually trained with on the origin host. torch 2.6.0+cu124
# matters: PyTorch's own SDPA dispatches to FlashAttention for this sequence shape, which is why
# flash-attn is deliberately NOT installed (see the note below).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Micromamba keeps the image small and the python pin exact.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN micromamba create -y -p /opt/env -c conda-forge python=3.10.20 && micromamba clean -a -y
ENV PATH=/opt/env/bin:$PATH

# --- versions transcribed from the origin host, not chosen here -----------------------------
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
        transformers==4.57.3 deepspeed==0.16.9 diffsynth==1.1.9 \
        accelerate safetensors einops imageio imageio-ffmpeg opencv-python-headless \
        pillow scikit-image lpips wandb numpy scipy

# DO NOT add flash-attn. The start-up log line `Flash Attention library "flash_attn" not found,
# using pytorch attention implementation` reads as a warning but is not one: measured on the
# origin host, torch.ops.aten._fused_sdp_choice returns backend 1 (FLASH_ATTENTION) for this
# arm's [1, 24, 28160, 128] bf16 attention, 54.9 ms vs 88.1 ms for the mem-efficient path. The
# external package buys nothing and costs a very long build.

# DeepSpeed's bf16 path does not skip non-finite gradients upstream; the repo patches that.
# The patch is applied at job start by run_train.sh so it survives a pip reinstall.
COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
RUN chmod -R a+rX /opt/env /app
CMD ["/bin/bash"]
