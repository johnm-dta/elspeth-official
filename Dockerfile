FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:${PATH}"

# OS deps for builds and uv installation
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Install uv for locked, reproducible installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy only what we need to install the package
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY example ./example

# Install dependencies and the CLI into the system site-packages
RUN uv sync --frozen --no-dev && uv pip install --system .

# Create a writable workspace for mounting host configs/data
# Operators mount their directory here: -v $(pwd):/workspace
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /bin/bash appuser \
    && mkdir -p /workspace \
    && chown -R appuser:appuser /workspace

WORKDIR /workspace
USER appuser

# Usage: docker run -v $(pwd):/workspace elspeth --settings config.yaml --secrets secrets.yaml
ENTRYPOINT ["elspeth"]
CMD ["--help"]
