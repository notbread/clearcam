from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2

from .inference.yolo import FrameDetection, UltralyticsDetector

logger = logging.getLogger(__name__)

# Suffix appended after the full filename (video.mp4 -> video.mp4.not_labeled).
# The suffix is not in VIDEO_EXTENSIONS, so find_videos() never re-picks it.
NOT_LABELED_SUFFIX = ".not_labeled"

_COLOR = (0, 255, 0)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6

# Persistent marker drawn in the bottom-right corner of every annotated frame so
# re-runs can detect already-annotated videos and skip them (boxes alone are not
# a reliable marker since they only appear where an object is).
MARKER_SIZE = 6


def draw_marker(frame) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w - MARKER_SIZE - 2, h - MARKER_SIZE - 2), (w - 2, h - 2), _COLOR, -1)


def pending_path(src: Path) -> Path:
    """Path of the temporary renamed original while annotation is in progress."""
    return src.with_name(src.name + NOT_LABELED_SUFFIX)


def _draw_detections(frame, detections: list[FrameDetection], names: dict[int, str]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR, _BOX_THICKNESS)
        caption = f"{names.get(det.cls_id, str(det.cls_id))} {det.confidence:.2f}"
        cv2.putText(frame, caption, (x1, max(0, y1 - 6)), _FONT, _FONT_SCALE, _COLOR, 1, cv2.LINE_AA)


def _start_encoder(dst: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """H.264/yuv420p so the result plays in browsers (motionEye web UI);
    +faststart puts the moov atom first so playback starts immediately."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(dst),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def annotate_video(
    detector: UltralyticsDetector,
    src: Path,
    dst: Path,
    conf: float,
    fps: float,
) -> bool:
    """Decode src, run detection on every frame, draw boxes + labels, and
    encode the annotated video to dst (H.264 via ffmpeg)."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        logger.error("Cannot open %s for annotation", src.name)
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    names = detector.class_names
    detections: list[FrameDetection] = []
    encoder: subprocess.Popen | None = None
    frame_idx = 0
    ok = True
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if encoder is None:
                height, width = frame.shape[:2]
                encoder = _start_encoder(dst, width, height, fps)
            detections = detector.predict_frame(frame, conf)
            if detections:
                _draw_detections(frame, detections, names)
            draw_marker(frame)
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
            frame_idx += 1
            if frame_idx % 500 == 0:
                logger.info("  Annotating: %d/%d frames (%.0f%%)", frame_idx, total_frames, frame_idx / max(1, total_frames) * 100)
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                logger.error("ffmpeg encode failed for %s: %s", dst.name, encoder.stderr.read().decode(errors="replace"))
                ok = False
    except Exception as e:
        logger.error("Annotation of %s failed: %s", src.name, e)
        ok = False
    finally:
        cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()

    if ok and frame_idx == 0:
        logger.error("Annotation of %s produced no frames", src.name)
        ok = False
    if not ok and dst.exists():
        dst.unlink()
    return ok
