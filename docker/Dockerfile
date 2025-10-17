
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

RUN pip install --upgrade pip wheel setuptools

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# Install remaining dependencies
COPY requirements.txt /opt/src/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /opt/src/requirements.txt
