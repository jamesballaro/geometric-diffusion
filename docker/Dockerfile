# Use PyTorch 2.8.0 with CUDA 12.9
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

USER root

ENV TZ=Europe/London \
    PATH=/usr/local/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
        ca-certificates \
        man-db lsb-release net-tools iproute2 netcat-traditional lsof psmisc \
        iputils-ping network-manager iptables \
        curl git build-essential gdb lcov pkg-config wget \
        libnss3-dev ffmpeg libbz2-dev libffi-dev libgdbm-dev \
        libgdbm-compat-dev liblzma-dev libncurses5-dev \
        libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv && \ 
    uv --version

# Copy dependency files
COPY pyproject.toml /opt/src/pyproject.toml
WORKDIR /opt/src

# Install torchvision matching torch 2.8.0
RUN uv pip install --system torchvision==0.23.0 torchaudio==2.8.0

RUN uv pip install --system --no-build-isolation .

RUN uv pip install --system git+https://github.com/patrick-kidger/torchcubicspline.git

WORKDIR /workspace 

CMD ["/bin/bash"]
