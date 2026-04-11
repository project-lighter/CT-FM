FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install system dependencies
#RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
ENV DEBIAN_FRONTEND=noninteractive
# Time zone setting
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 tzdata \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chown user:user /workspace/inputs /workspace/outputs

USER user

ENV PATH="/home/user/.local/bin:${PATH}"

# Upgrade pip
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# Copy the entire CT-NEXUS repository
COPY --chown=user:user ./dinov2 /opt/app/dinov2
COPY --chown=user:user extract_feat_LP.py /opt/app/extract_feat_LP.py
COPY --chown=user:user extract_feat_LP.sh /opt/app/extract_feat_LP.sh
COPY --chown=user:user requirements.txt /opt/app/requirements.txt
COPY --chown=user:user ./3dino_vit_weights.pth /opt/app/3dino_vit_weights.pth

# Set working directory for installation
WORKDIR /opt/app/

# Install nnssl package with all dependencies from pyproject.toml
RUN pip install --user -r requirements.txt

# Set working directory to feature_extraction
WORKDIR /opt/app/


ENTRYPOINT []
