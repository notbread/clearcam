# ClearCam

Detect persons and animals (cats, dogs, foxes, etc.) in motionEye videos, annotate
detected objects in place, and delete the rest to save space. Videos never leave their
folder, so motionEye's web UI keeps listing and playing them.
Runs on YOLO weights (PyTorch) or OpenVINO-exported models (Intel iGPU accelerated).

## Setup

This project uses `uv` for dependency management.

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies and create virtual environment:
   ```bash
   uv sync
   ```

PyTorch is resolved to the CPU-only build (`pyproject.toml` pins the `pytorch-cpu`
index): this project runs on CPU and Intel iGPU, and the default PyPI CUDA build
would only add ~6 GB of unusable CUDA libraries.

## Container (podman-first)

Podman is the primary way to deploy the app; `docker build`/`docker run` accept the
same commands, minus the podman-specific flags noted below. The image ships ffmpeg,
CPU PyTorch, OpenCV and OpenVINO, and runs as an unprivileged UID/GID 1000.

### Build

```bash
podman build -t clearcam .
```

`Containerfile` is a symlink to `Dockerfile`, so podman picks it up without `-f`.

### Run

Mount the videos directory and pass its container path as the argument. Videos are
annotated in place, so motionEye keeps listing them:

```bash
podman run --rm \
  --userns=keep-id \
  -v /mnt/disk3/containers/motioneye/var/Camera1/2026-09-15:/videos:Z \
  clearcam \
  /videos --annotate --delete-no-object --dry-run
```

- `--userns=keep-id` maps your host UID/GID onto the container's `app` user, so
  annotated videos written back into the mount stay owned by you instead of turning
  into root-owned files. If your host UID is not 1000, use
  `--userns=keep-id:uid=1000,gid=1000`.
- `:Z` relabels the mount for SELinux hosts (Fedora, RHEL); drop it elsewhere.
- Intel iGPU inference needs the render device and your host `render` group:
  add `--device /dev/dri --group-add keep-groups`.
- Weights are resolved relative to the working directory (`/data`). Mount one there
  to cache downloaded `*.pt` weights between runs, e.g. `-v ood-weights:/data`, or
  mount exported models read-only and point `--model` at them:
  ```bash
  -v "$PWD/yolo26n_openvino_model:/models/yolo26n_openvino_model:ro,Z" \
  ... /videos --annotate --model /models/yolo26n_openvino_model
  ```

### Prebuilt image

`.github/workflows/docker.yml` builds the image on manual dispatch
(`workflow_dispatch`, main branch only), smoke-tests it, and pushes it to the
GitHub Container Registry:

```bash
podman pull ghcr.io/<owner>/clearcam:latest
```

Package visibility follows the repository, so for a private repo authenticate first
with a token that has `read:packages`:

```bash
podman login ghcr.io -u <github-user>
```

## Usage

```bash
uv run clearcam <src_videos_dir> [options]
```

### Example

```bash
mkdir -p src_videos
# Put your videos in src_videos
uv run clearcam src_videos --dry-run
# Annotate detected videos in place, delete the rest
uv run clearcam src_videos --annotate --delete-no-object
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolo11n.pt` | `.pt` weights or an exported `*_openvino_model` directory |
| `--device` | auto | `cpu`, `intel:gpu`, `cuda:0`, ... (auto: `intel:gpu` for OpenVINO models, `cpu` otherwise) |
| `--conf` | `0.3` | Confidence threshold |
| `--sample-interval` | `1.0` | Seconds between sampled frames |
| `--classes` | person + COCO animals | Restrict detection, e.g. `--classes person cat dog` |
| `--dry-run` | off | Preview actions without changing files |
| `--recursive` | off | Scan subdirectories for videos |
| `--trim-only` | off | Trim videos in-place to detected segments (requires ffmpeg) |
| `--annotate` | off | Draw boxes around detected objects and replace the video in-place (mutually exclusive with `--trim-only`, requires ffmpeg) |
| `--delete-no-object` | off | Delete videos with no detected objects (frees space) |

## Annotation mode

`--annotate` rewrites each detected video in place with a bounding box and
`label confidence` caption drawn around every detected object. Detection runs on
sampled frames (every `--sample-interval` seconds); the sampled boxes are held and
drawn on the frames in between, so annotation stays fast while keeping the video smooth.

The output is re-encoded as **H.264 / yuv420p** with `+faststart`, so it plays in the
motionEye web UI (and any browser) and starts streaming immediately. It is typically
*smaller* than the original motionEye file.

Undetected videos are left untouched; detected videos are replaced in place (the
original is removed after the annotated copy is verified). Videos are never moved, so
motionEye keeps listing them under the same camera and date.

A small green marker is drawn in the bottom-right corner of every annotated frame. On a
re-run the app detects that marker and skips already-annotated videos, so running the
same folder twice does not re-annotate.

### Crash safety

The workflow is journaled. For each detected video the app:

1. appends an entry to `annotate_journal.csv` (in the source directory),
2. renames the video to `<name>.not_labeled`,
3. writes the annotated video to the original filename,
4. deletes the `.not_labeled` original and removes the journal entry.

If the process is interrupted (e.g. power outage) between steps 2-4, the next
`--annotate` run recovers automatically before processing: it validates any
partially-written output, re-annotates from the surviving `.not_labeled` original
when needed, and clears the journal. Recovery is idempotent, so re-running after a
failed recovery simply tries again.

## Deleting empty videos

`--delete-no-object` deletes videos in which nothing was detected, freeing space. Use
`--dry-run` first to see what would be removed. Videos with errors (unreadable) are
left in place so you can inspect them.

## OpenVINO (Intel iGPU)

Export a model once, then use the exported directory for accelerated inference:

```bash
# FP16 export (recommended; no extra dependencies)
uv run clearcam-convert --model yolo11n.pt        # -> yolo11n_openvino_model/

# INT8 export (faster, needs nncf: uv pip install nncf)
uv run clearcam-convert --model yolo11n.pt --int8
```

Then run detection against the exported model — the Intel iGPU is used automatically
(falling back to CPU if unavailable):

```bash
uv run clearcam src_videos --model yolo11n_openvino_model
```

## Detection

Uses YOLO11 trained on the COCO dataset. Default target classes:
- Person
- Cat, Dog, Bird, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe.

*Note: Foxes are not explicitly in the COCO dataset but are often detected as "dog" or "cat" due to visual similarities.*

## Automation

`cleanup.sh` is a cron wrapper: each night it runs `--annotate --delete-no-object` on
yesterday's motionEye footage using the OpenVINO model (Intel iGPU), writes a daily
log, and prunes logs older than 30 days. Detected videos are annotated in place
(visible in motionEye's web UI); videos with no detected objects are deleted.

It runs from the `uv` environment in this directory. To run the same job from the
container instead (no host venv or ffmpeg needed), call podman from cron:

```bash
podman run --rm --userns=keep-id \
  -v /mnt/disk3/containers/motioneye/var/Camera1/${YESTERDAY}:/videos:Z \
  -v ood-weights:/data \
  ghcr.io/<owner>/clearcam:latest \
  /videos --annotate --delete-no-object --model /data/yolo26n_openvino_model
```
