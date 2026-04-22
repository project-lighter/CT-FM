FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    git tzdata \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chmod 777 /workspace/inputs /workspace/outputs


WORKDIR /opt/app/
COPY requirements.txt /opt/app/requirements.txt
# Install deps as root into system site-packages so lighter_zoo's torch dep
# doesn't shadow the base image's torch 2.1.0+cu121
RUN grep -v -E "^(torch|torchvision|--extra-index-url)" requirements.txt \
    | pip install -r /dev/stdin \
    && pip install torch==2.1.0 torchvision==0.16.0 \
       --extra-index-url https://download.pytorch.org/whl/cu121 \
       --force-reinstall --no-deps

COPY extract_feat_LP.py /opt/app/extract_feat_LP.py
COPY extract_feat_LP.sh /opt/app/extract_feat_LP.sh
    

# Download CT-FM weights from HuggingFace Hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='project-lighter/ct_fm_feature_extractor', local_dir='./ct_fm_weights')"

RUN chown -R user:user /opt/app/

USER user

ENV PATH="/home/user/.local/bin:${PATH}"

ENTRYPOINT []
