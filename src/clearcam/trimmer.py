from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def trim_output_path(src: Path) -> Path:
    return src.parent / f"{src.stem}_trimmed{src.suffix}"


def trim_video(src: Path, dst: Path, start_sec: float, end_sec: float) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-to", f"{end_sec:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg failed: %s", result.stderr)
            return False
        return True
    except Exception as e:
        logger.error("ffmpeg error: %s", e)
        return False
