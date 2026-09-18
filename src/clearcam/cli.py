from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .annotator import annotate_video, pending_path
from .config import VIDEO_EXTENSIONS, Settings
from .inference import create_detector
from . import journal
from .trimmer import trim_output_path, trim_video
from .video import looks_annotated, probe_video

logger = logging.getLogger(__name__)


@dataclass
class Stats:
    processed: int = 0
    detected: int = 0
    no_object: int = 0
    error: int = 0
    skipped: int = 0
    annotated: int = 0
    deleted: int = 0


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description="Detect persons/animals in videos; annotate detected ones in place, delete the rest.")
    parser.add_argument("src_dir", help="Directory containing source videos")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model (.pt or exported *_openvino_model dir) (default: yolo11n.pt)")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu, intel:gpu, cuda:0 (default: auto — intel:gpu for OpenVINO models, cpu otherwise)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Seconds between sampled frames (default: 1.0)")
    parser.add_argument("--classes", nargs="+", metavar="CLASS", default=[], help="Class names to detect, e.g. --classes person cat dog (default: person + all COCO animals)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without changing files")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories for videos")
    parser.add_argument("--trim-only", action="store_true", help="Trim videos in-place to detected object segments (requires ffmpeg)")
    parser.add_argument("--annotate", action="store_true", help="Draw boxes around detected objects and replace the video in-place (crash-safe via journal CSV, requires ffmpeg)")
    parser.add_argument("--delete-no-object", action="store_true", help="Delete videos with no detected objects (frees space)")
    args = parser.parse_args(argv)

    if args.annotate and args.trim_only:
        parser.error("--annotate and --trim-only are mutually exclusive")

    return Settings(
        src_dir=Path(args.src_dir).expanduser(),
        model=args.model,
        device=args.device,
        conf=args.conf,
        sample_interval=args.sample_interval,
        classes=tuple(args.classes),
        dry_run=args.dry_run,
        recursive=args.recursive,
        trim_only=args.trim_only,
        annotate=args.annotate,
        delete_no_object=args.delete_no_object,
    )


def find_videos(settings: Settings) -> list[Path]:
    if settings.recursive:
        return [f for f in settings.src_dir.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
    return [f for f in settings.src_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]


def _annotate_video_file(video_file: Path, settings: Settings, detector, stats: Stats,
                         video_info, result) -> None:
    """Annotate a detected video in-place, journaled for crash recovery."""
    journal.append(settings.src_dir, video_file, result.labels)
    pending = pending_path(video_file)
    video_file.rename(pending)
    if annotate_video(detector, pending, video_file, settings.conf, video_info.fps):
        pending.unlink()
        journal.remove(settings.src_dir, video_file)
        logger.info("  -> Annotated in place: %s", video_file.name)
        stats.annotated += 1
    else:
        logger.error("  Annotation failed for %s; leaving %s for recovery", video_file.name, pending.name)
        stats.error += 1


def _vid_stride_of(settings: Settings, video_info) -> int:
    return max(1, int(video_info.fps * settings.sample_interval))


def process_video(video_file: Path, settings: Settings, detector, stats: Stats) -> None:
    logger.info("Processing: %s", video_file.name)

    trimmed_path = trim_output_path(video_file)
    if settings.trim_only and trimmed_path.exists():
        logger.info("  Skipping: %s already exists", trimmed_path.name)
        stats.skipped += 1
        return

    try:
        if settings.annotate and not settings.dry_run and looks_annotated(video_file):
            logger.info("  Already annotated, skipping: %s", video_file.name)
            stats.skipped += 1
            return

        video_info = probe_video(video_file)
        if video_info is None:
            logger.warning("Cannot read %s, leaving it in place", video_file.name)
            stats.error += 1
            return

        vid_stride = _vid_stride_of(settings, video_info)
        logger.info("  fps=%.1f, sampling every %d frames (~%.1fs)", video_info.fps, vid_stride, settings.sample_interval)

        result = detector.detect(video_file, settings.conf, vid_stride, stop_on_first=not settings.trim_only)

        if result.labels:
            if settings.annotate:
                if settings.dry_run:
                    logger.info("  -> Would annotate in place: %s", video_file.name)
                    stats.annotated += 1
                else:
                    _annotate_video_file(video_file, settings, detector, stats, video_info, result)
                stats.detected += 1
            elif settings.trim_only:
                start_sec = result.first_frame / video_info.fps
                end_sec = result.last_frame / video_info.fps

                if result.first_frame == 0 and end_sec >= video_info.duration - 1:
                    logger.info("  No trimming needed (objects span entire video)")
                    stats.detected += 1
                    stats.processed += 1
                    return

                if settings.dry_run:
                    logger.info("  -> Would trim %.2fs-%.2fs to %s", start_sec, end_sec, trimmed_path.name)
                else:
                    if trim_video(video_file, trimmed_path, start_sec, end_sec):
                        logger.info("  -> Trimmed to %s (%.2fs-%.2fs)", trimmed_path.name, start_sec, end_sec)
                    else:
                        logger.error("  Failed to trim %s", video_file.name)
                stats.detected += 1
            else:
                logger.info("  Detected: %s (leaving in place)", ", ".join(sorted(result.labels)))
                stats.detected += 1
        else:
            if settings.delete_no_object:
                if settings.dry_run:
                    logger.info("  No objects found -> would delete %s", video_file.name)
                else:
                    video_file.unlink()
                    logger.info("  No objects found -> deleted %s", video_file.name)
                    stats.deleted += 1
            else:
                logger.info("  No objects found, keeping %s", video_file.name)
            stats.no_object += 1

        stats.processed += 1

    except Exception as e:
        logger.error("Error processing %s: %s", video_file.name, e)
        stats.error += 1


def _output_is_valid(video: Path, pending: Path) -> bool:
    """An interrupted annotate leaves a truncated mp4 (moov atom written only on
    writer release). Trust the output only if it decodes and reaches roughly the
    source's frame count."""
    src_info, dst_info = probe_video(pending), probe_video(video)
    if src_info is None or dst_info is None:
        return False
    return dst_info.frame_count >= src_info.frame_count * 0.9


def recover_annotations(settings: Settings, detector, stats: Stats) -> set[Path]:
    """Finish or redo annotations interrupted by a crash/power outage.

    Returns the set of recovered video paths so the normal loop skips them
    (they are already annotated)."""
    recovered: set[Path] = set()
    for entry in journal.pending_entries(settings.src_dir):
        video, pending = entry.path, pending_path(entry.path)
        logger.info("Recovering interrupted annotation: %s", video.name)
        try:
            if pending.exists() and video.exists():
                if _output_is_valid(video, pending):
                    # Crash after annotation completed: clean up the original.
                    pending.unlink()
                else:
                    # Output is truncated/corrupt: redo from the original.
                    logger.info("  Output incomplete, re-annotating from original")
                    video.unlink()
                    info = probe_video(pending)
                    if annotate_video(detector, pending, video, settings.conf, info.fps):
                        pending.unlink()
                    else:
                        logger.error("  Recovery annotation failed for %s, will retry next run", video.name)
                        stats.error += 1
                        continue
            elif pending.exists():
                # Crash mid-annotation before output was written: redo.
                info = probe_video(pending)
                if info is None:
                    logger.error("  Cannot read %s, skipping recovery", pending.name)
                    stats.error += 1
                    continue
                if annotate_video(detector, pending, video, settings.conf, info.fps):
                    pending.unlink()
                else:
                    logger.error("  Recovery annotation failed for %s, will retry next run", video.name)
                    stats.error += 1
                    continue
            elif not video.exists():
                # Both gone: nothing recoverable, drop the stale entry.
                logger.warning("  Both %s and %s missing, dropping stale entry", video.name, pending.name)
            # else: only the valid original/output exists -> fall through to cleanup.
            journal.remove(settings.src_dir, video)
            stats.annotated += 1
            if video.exists():
                recovered.add(video)
        except Exception as e:
            logger.error("Recovery of %s failed: %s", video.name, e)
            stats.error += 1
    return recovered


def main(argv: list[str] | None = None) -> None:
    settings = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if (settings.trim_only or settings.annotate) and not shutil.which("ffmpeg"):
        logger.error("--trim-only/--annotate require ffmpeg to be installed and on PATH")
        sys.exit(1)

    if not settings.src_dir.is_dir():
        logger.error("Source directory '%s' does not exist.", settings.src_dir)
        sys.exit(1)

    try:
        detector = create_detector(settings)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    stats = Stats()
    recovered: set[Path] = set()
    if settings.annotate and not settings.dry_run:
        recovered = recover_annotations(settings, detector, stats)

    videos = find_videos(settings)
    if not videos and not recovered:
        logger.error("No videos found in %s", settings.src_dir)
        sys.exit(1)

    logger.info("Found %d videos. Starting detection...", len(videos))
    if settings.dry_run:
        logger.info("Dry-run mode: no files will be changed.")

    for video_file in videos:
        if video_file in recovered:
            logger.info("Skipping already-recovered: %s", video_file.name)
            stats.skipped += 1
            continue
        process_video(video_file, settings, detector, stats)

    logger.info("--- Summary ---")
    logger.info("Processed: %d", stats.processed)
    logger.info("Detected:  %d", stats.detected)
    logger.info("No object: %d", stats.no_object)
    logger.info("Errors:    %d", stats.error)
    if settings.trim_only or settings.annotate:
        logger.info("Skipped:   %d", stats.skipped)
    if settings.annotate:
        logger.info("Annotated: %d", stats.annotated)
    if settings.delete_no_object:
        logger.info("Deleted:   %d", stats.deleted)


if __name__ == "__main__":
    main()
