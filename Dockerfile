FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install system dependencies
#RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
ENV DEBIAN_FRONTEND=noninteractive
# Time zone setting
ENV TZ=Etc/UTC

RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
 && apt-get update || true \
 && apt-get install -y --no-install-recommends --fix-missing \
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

COPY --chown=user:user extract_feat_LP.py /opt/app/extract_feat_LP.py
COPY --chown=user:user extract_feat_LP.sh /opt/app/extract_feat_LP.sh
COPY --chown=user:user requirements.txt /opt/app/requirements.txt

# Set working directory for installation
WORKDIR /opt/app/

# Install dependencies
RUN pip install --user -r requirements.txt

# Download CT-FM weights from HuggingFace Hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='project-lighter/ct_fm_feature_extractor', local_dir='./ct_fm_weights')"

# Set working directory to feature_extraction
WORKDIR /opt/app/


ENTRYPOINT []
