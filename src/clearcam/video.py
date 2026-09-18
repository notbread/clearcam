from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Same green the annotator draws boxes in (BGR). Used as a watermark to detect
# already-annotated videos so re-runs don't annotate them again.
_ANNOTATION_COLOR = (0, 255, 0)


@dataclass
class VideoInfo:
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def probe_video(path: Path) -> VideoInfo | None:
    """Open a video once to validate readability and extract fps/frame count."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        ret, _ = cap.read()
        if not ret:
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(fps=fps, frame_count=frame_count)
    finally:
        cap.release()


def has_annotation_mark(frame: np.ndarray) -> bool:
    """True if the bottom-right corner carries the annotator's green marker
    (drawn on every annotated frame), so re-runs skip the video."""
    h, w = frame.shape[:2]
    region = frame[h - 12:h, w - 12:w]
    g = region[:, :, 1].astype(np.int32)
    r = region[:, :, 0].astype(np.int32)
    b = region[:, :, 2].astype(np.int32)
    mask = (g > 150) & (g - r > 60) & (g - b > 60)
    return int(mask.sum()) > 20


def looks_annotated(path: Path) -> bool:
    """Sample a few frames; if any carries the green watermark, treat the video
    as already annotated so a re-run leaves it alone."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frac in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * frac))
            ret, frame = cap.read()
            if ret and has_annotation_mark(frame):
                return True
        return False
    finally:
        cap.release()
