FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ARG GIT_SSL_NO_VERIFY=true
ARG HTTPS_PROXY="http://cloudproxy.sei.cmu.edu:80"
ARG HTTP_PROXY="http://cloudproxy.sei.cmu.edu:80"
ARG NO_PROXY="127.0.0.1,10.0.0.0/8,localhost,.sei.cmu.edu,.cert.org,.zscalergov.net,.zpagov.net,.duosecurity.com"

# add proxy in image for when trying to apt update from container
ARG APT_PROXY=/etc/apt/apt.conf.d/proxy.conf
RUN echo -e \
    "Acquire::http::Proxy \"http://cloudproxy.sei.cmu.edu:80/\";" \
    >> ${APT_PROXY} && \
    echo -e \
    "Acquire::https::Proxy \"http://cloudproxy.sei.cmu.edu:80/\";" \
    >> ${APT_PROXY} && \
    >> ${APT_PROXY} && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        curl \
        sudo \
        perl \
        make \
	emacs-nox \
        git \
        gosu \
        wget
ENV APT_CONF=/etc/apt/apt.conf
RUN mkdir -p /usr/local/share/ca-certificates
RUN wget http://aia.sei.cmu.edu/sei-ecdsa-root-ca-g4.crt \
        -O /usr/local/share/ca-certificates/sei-ecdsa-root-ca-g4.crt
RUN wget http://aia.sei.cmu.edu/ZscalerRootCertificate-2048-SHA256.crt \
        -O /usr/local/share/ca-certificates/ZscalerRootCA.crt
RUN update-ca-certificates
# RUN echo "ca_certificate=${CERT}" > /root/.wgetrc && \
#     echo "Acquire::https { CaInfo ${CERT}; }" > ${APT_CONF} && \
#     git config --global http.sslCAInfo ${CERT}


# Install uv into /usr/local/bin (which is already on PATH)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

RUN mkdir -p /mnt/work/src /mnt/work/data
RUN useradd -m docker-user
WORKDIR /mnt/work

COPY --chmod=755 ./scripts/entrypoint.sh /entrypoint.sh
# set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
