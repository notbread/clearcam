# Podman-first: plain `podman build`/`docker build` work; no BuildKit-only
# syntax is used, and the image runs rootless with `--userns=keep-id`.
FROM python:3.14-slim-trixie

# ffmpeg          -> H.264 encoding for --annotate / --trim-only
# libgl1          -> OpenCV runtime (opencv-python is not the headless wheel)
# libglib2.0-0t64 -> glib runtime required by OpenCV
# libgomp1        -> OpenMP, used by torch and OpenVINO
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
       libgl1 \
       libglib2.0-0t64 \
       libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    HOME=/home/app

# Unprivileged by default. UID/GID 1000 is what `podman run --userns=keep-id`
# maps the host user to, so videos written back into a bind mount keep the
# host ownership instead of turning into root-owned files.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
  && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin app

WORKDIR /app
RUN chown app:app /app \
  && mkdir -p /data \
  && chown app:app /data
USER app

# Dependencies first: this layer only rebuilds when the lockfile changes.
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app src ./src
RUN uv sync --frozen --no-dev --no-editable

# Weights are resolved relative to the working directory, so run from /data and
# pass the videos directory as an argument (mount it, e.g. -v /videos:/videos).
WORKDIR /data

ENTRYPOINT ["clearcam"]
CMD ["--help"]
