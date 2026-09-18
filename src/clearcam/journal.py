from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ANNOTATE_CSV = "annotate_journal.csv"


@dataclass
class JournalEntry:
    path: Path
    labels: tuple[str, ...]


def _csv_path(src_dir: Path) -> Path:
    return src_dir / ANNOTATE_CSV


def append(src_dir: Path, video: Path, labels: set[str]) -> None:
    """Record a video about to be annotated. Flushed immediately so the entry
    survives a power outage."""
    csv_file = _csv_path(src_dir)
    is_new = not csv_file.exists()
    with csv_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["path", "labels", "timestamp"])
        writer.writerow([str(video), "|".join(sorted(labels)), datetime.now(timezone.utc).isoformat()])
        f.flush()
        os.fsync(f.fileno())


def remove(src_dir: Path, video: Path) -> None:
    """Drop a completed entry via atomic rewrite (temp file + os.replace)."""
    entries = [e for e in pending_entries(src_dir) if e.path != video]
    csv_file = _csv_path(src_dir)
    tmp = csv_file.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "labels", "timestamp"])
        for e in entries:
            writer.writerow([str(e.path), "|".join(e.labels), datetime.now(timezone.utc).isoformat()])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_file)
    if not entries:
        csv_file.unlink(missing_ok=True)


def pending_entries(src_dir: Path) -> list[JournalEntry]:
    """All entries awaiting annotation (i.e. interrupted by a crash)."""
    csv_file = _csv_path(src_dir)
    if not csv_file.exists():
        return []
    entries: list[JournalEntry] = []
    with csv_file.open(newline="") as f:
        for row in csv.DictReader(f):
            labels = tuple(l for l in row.get("labels", "").split("|") if l)
            entries.append(JournalEntry(path=Path(row["path"]), labels=labels))
    return entries
