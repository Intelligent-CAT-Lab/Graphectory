# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ARG TARGETPLATFORM
ARG TARGETARCH

LABEL org.opencontainers.image.title="Graphectory" \
      org.opencontainers.image.description="Reproducible graph analysis for agentic software systems" \
      org.opencontainers.image.source="https://github.com/Intelligent-CAT-Lab/Graphectory" \
      org.opencontainers.image.documentation="https://github.com/Intelligent-CAT-Lab/Graphectory/blob/main/README.md"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    HOME=/home/graphectory \
    PATH="/home/graphectory/.local/bin:$PATH"

WORKDIR /opt/graphectory

# Install system dependencies for pygraphviz and scientific computing
# All packages available for linux/amd64 and linux/arm64 in Debian Bookworm
RUN apt-get update && apt-get install --no-install-recommends -y \
      build-essential \
      graphviz \
      libgraphviz-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and package structure for layer caching
COPY pyproject.toml README.md LICENSE ./
COPY graph_construction/__init__.py graph_construction/__init__.py
COPY graph_analysis/__init__.py graph_analysis/__init__.py
COPY lang_construction/__init__.py lang_construction/__init__.py

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir .

# Copy remaining project files (data, plots, scripts)
COPY . .

# Create non-root user and directories
RUN useradd --create-home --uid 10001 --shell /bin/bash graphectory && \
    mkdir -p /output /opt/graphectory/figures && \
    chown -R graphectory:graphectory \
      /opt/graphectory /output /home/graphectory && \
    chmod 0755 docker/*.sh

USER graphectory

VOLUME ["/output"]

# Interactive shell by default; run `reproduce` for automated analysis
CMD ["bash"]

