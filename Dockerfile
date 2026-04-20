ARG tag=2.19.0-gpu
ARG repo_url=https://github.com/lifewatch/phyto-plankton-classification.git
ARG branch=cyto

FROM tensorflow/tensorflow:${tag}
ARG repo_url
ARG branch

LABEL maintainer='Wout Decrop (VLIZ)'
LABEL version='0.1.0'
ENV CONTAINER_MAINTAINER="Wout Decrop <wout.decrop@vliz.be>"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN for i in 1 2 3; do apt-get update && break || { echo "apt-get update failed, retrying ($i/3)"; sleep 10; }; done && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fail fast on branch/repo access before the heavier runtime dependencies install.
RUN git clone -b ${branch} --depth 1 ${repo_url} /tmp/phyto-plankton-classification

RUN for i in 1 2 3; do apt-get update && break || { echo "apt-get update failed, retrying ($i/3)"; sleep 10; }; done && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        curl \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    pip3 install --no-cache-dir --upgrade pip "setuptools<60.0.0" wheel

ENV LANG=C.UTF-8

WORKDIR /srv

ENV DISABLE_AUTHENTICATION_AND_ASSUME_AUTHENTICATED_USER=yes
ENV PLANKTONCLAS_CONFIG=/srv/config.yaml

COPY config.yaml /srv/config.yaml

# Use the cloned GitHub branch as the source for the runtime assets that
# the packaged service still expects at startup.
RUN pip install --no-cache-dir --ignore-installed blinker blinker && \
    pip install --no-cache-dir -r /tmp/phyto-plankton-classification/requirements.txt && \
    mkdir -p /srv/models /srv/data && \
    cp -R /tmp/phyto-plankton-classification/data/. /srv/data/ && \
    cp -R /tmp/phyto-plankton-classification/models/. /srv/models/ && \
    rm -rf /tmp/phyto-plankton-classification

EXPOSE 5000

CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
