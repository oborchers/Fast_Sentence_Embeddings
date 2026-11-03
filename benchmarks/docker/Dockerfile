# Linux development and functional probes only. Never use this image for README numbers.
# Public multi-architecture python:3.13.16-slim index, resolved 2026-11-03.
FROM python:3.13.16-slim@sha256:cf50a4f6ac4f026458b08d5222023d700ac806410dff5e9ca8728fa44f2b850f

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git procps ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.13.0

ENV UV_PROJECT_ENVIRONMENT=/opt/fse-env \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FSE=2.0.0.dev0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    FSE_BENCH_LOCK=/coordination/bench.lock \
    FSE_HEAVY_LOCK=/coordination/heavy.lock

# Build this checkout's extension in Linux. No model or corpus enters the image.
# The development version is explicit because Git worktree metadata is host-specific.
WORKDIR /opt/fse-build
COPY pyproject.toml uv.lock setup.py README.md LICENSE NOTICE ./
COPY src/ ./src/
RUN test "$(python -c 'import platform; print(platform.python_version())')" = 3.13.16 \
    && uv sync --locked --no-default-groups --group test --no-editable --python /usr/local/bin/python \
    && /opt/fse-env/bin/python -c 'import fse; from fse._backends._cpu import CpuBackend; CpuBackend()' \
    && uv pip check --python /opt/fse-env/bin/python

# Mount the same checkout at /source:ro. Rebuild this image after product changes.
# Mount /results, /work, /cache and /coordination separately and keep them outside Git.
WORKDIR /source
ENTRYPOINT ["/opt/fse-env/bin/python"]
CMD ["/source/benchmarks/speed.py", "--local-info"]
