FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    ca-certificates \
    gnupg \
    git \
    make \
    curl \
    rsync \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install eccodes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-tools \
    libeccodes-dev \
 && rm -rf /var/lib/apt/lists/*

 ENV HOME=/home/earth2
 WORKDIR /home/earth2

# Installing newer version of cmake (for dm-tree and natten)
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
  && sh -c 'echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" > /etc/apt/sources.list.d/kitware.list' \
  && apt-get update \
  && apt-get install -y --no-install-recommends cmake \
  && rm -rf /var/lib/apt/lists/*

# Disable contraint files in the container
# PyTorch container started constraining packages which we don't want
ENV PIP_CONSTRAINT=
RUN for PIP_CONF in "/etc/xdg/pip/pip.conf" "/root/.pip/pip.conf" "/usr/pip.conf" "/root/.config/pip/pip.conf" "/etc/pip.conf"; do \
        if [ -f "$PIP_CONF" ]; then \
          sed -i 's/^no-cache-dir = true/#no-cache-dir = true/' "$PIP_CONF"; \
        fi; \
    done

# Install tox for venv managment
RUN uv tool install tox --with tox-uv

# Use the build argument
ARG PYTHON_VERSION=3.12.9
RUN /usr/bin/uv python install "${PYTHON_VERSION}"

# Set CUDA architectures for building without GPU present
# Covers: Volta (7.0), Ampere (8.0, 8.6), Ada Lovelace (8.9), Hopper (9.0), Blackwell (10.0)
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0"
ENV MAX_JOBS="8"
ENV UV_PROJECT_ENVIRONMENT="${HOME}/.venv"

# Optional: Setup CI dependencies by copying repo, running make setup ci, then cleaning up
ARG DEV=false
RUN if [ "$DEV" = "true" ]; then \
        mkdir /home/earth2/sample-project && cd /home/earth2/sample-project && \
        uv tool install tox --with tox-uv && \
        uv init . --python 3.12 &&  uv venv --clear && \
        uv add torch==2.10 && \
        uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git" && \
        uv sync; \
    fi
RUN if [ "$DEV" = "true" ]; then \
        cd /home/earth2/sample-project && \
        uv add hatchling cucim-cu12 && \
        uv add "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl" && \
        uv sync && \
        uv add earth2studio --extra all && \
        uv sync; \
    fi
# Configure .bashrc to auto-activate virtual environment
RUN if [ "$DEV" = "true" ]; then \
        echo "source \${UV_PROJECT_ENVIRONMENT}/bin/activate" >> ${HOME}/.bashrc; \
    fi
