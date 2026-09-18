from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# COCO class ids: person + animals (bird, cat, dog, horse, sheep, cow,
# elephant, bear, zebra, giraffe)
DEFAULT_TARGET_CLASSES: tuple[int, ...] = (0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".wmv")


@dataclass
class Settings:
    src_dir: Path
    model: str = "yolo11n.pt"
    device: str | None = None  # None -> auto (prefer intel:gpu for OpenVINO models)
    conf: float = 0.3
    sample_interval: float = 1.0
    classes: tuple[str, ...] = field(default_factory=tuple)  # empty -> DEFAULT_TARGET_CLASSES
    dry_run: bool = False
    recursive: bool = False
    trim_only: bool = False
    annotate: bool = False
    delete_no_object: bool = False
