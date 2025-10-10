
# Base images:
# Mac (arm64) CPU
FROM armswdev/pytorch-arm-neoverse:r24.07-torch-2.3.0-onednn-acl
# CUDA 12.0 + cuDNN 8.3.2
#docker pull pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
# Set environment variables

USER root

ENV TZ=Europe/London \
    PATH=/usr/local/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
        man-db lsb-release net-tools iproute2 netcat lsof psmisc \
        iputils-ping network-manager iptables \
        curl git build-essential gdb lcov pkg-config wget \
        libnss3-dev ffmpeg libbz2-dev libffi-dev libgdbm-dev \
        libgdbm-compat-dev liblzma-dev libncurses5-dev \
        libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev && \
    # Upgrade pip
    python -m pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency files into container
COPY requirements.txt /opt/src/
COPY vfx_deps.sh /opt/src/

# Install custom VFX dependencies (Imath, OpenEXR, OCIO, OIIO, etc.)
RUN chmod +x /opt/src/vfx_deps.sh && \
    stdbuf -oL -eL bash -x /opt/src/vfx_deps.sh

RUN pip install --no-cache-dir -r /opt/src/requirements.txt

# Default run command: verify PyTorch + CUDA
# CMD ["python3", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"]
