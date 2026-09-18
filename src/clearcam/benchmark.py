from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_TARGET_CLASSES, VIDEO_EXTENSIONS
from .inference.yolo import UltralyticsDetector
from .video import probe_video

logger = logging.getLogger(__name__)


@dataclass
class ModelRun:
    label: str
    model_path: str
    device: str | None
    detected: bool
    labels: set[str]
    max_conf: float
    elapsed: float
    error: str | None = None


@dataclass
class ModelSummary:
    label: str
    model_path: str
    device: str | None
    total: int
    detected: int
    no_object: int
    errors: int
    total_sec: float
    avg_sec: float
    min_sec: float
    max_sec: float
    classes_found: set[str]
    avg_max_conf: float


def _short_name(model_path: str) -> str:
    stem = Path(model_path).name
    stem = stem.removesuffix(".pt")
    stem = stem.removesuffix("_openvino_model")
    return stem


def _resolve_device(model_path: str) -> str | None:
    if Path(model_path).is_dir():
        return "intel:gpu"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple YOLO models against the same video set."
    )
    parser.add_argument("src_dir", help="Directory containing source videos")
    parser.add_argument(
        "--model", action="append", default=[],
        help="Model path (.pt or _openvino_model/). Repeat to compare multiple models. "
             "Add ':DEVICE' to override device for that model, e.g. yolo11s.pt:cpu",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Seconds between sampled frames (default: 1.0)")
    parser.add_argument("--classes", nargs="+", metavar="CLASS", default=[], help="Target classes (default: person + animals)")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories for videos")
    parser.add_argument("--no-errors", action="store_true", help="Omit per-video details for successful runs (summary only)")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("At least one --model is required")
    return args


def _parse_model_spec(raw: str) -> tuple[str, str | None]:
    parts = raw.rsplit(":", 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return parts[0], None


def find_videos(src_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(f for f in src_dir.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS)


def _create_detector(model_path: str, device: str | None, class_names: tuple[str, ...]) -> UltralyticsDetector:
    return UltralyticsDetector(model_path, device, class_names)


def _run_model(video: Path, detector: UltralyticsDetector, conf: float, vid_stride: int) -> ModelRun:
    start = time.perf_counter()
    try:
        labels: set[str] = set()
        max_conf = 0.0
        kwargs: dict = {
            "source": str(video), "stream": True, "conf": conf,
            "vid_stride": vid_stride, "verbose": False,
        }
        if detector.device is not None:
            kwargs["device"] = detector.device
        for res in detector._model.predict(**kwargs):
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls = int(box.cls[0])
                    if cls in detector.target_classes:
                        conf_val = float(box.conf[0])
                        if conf_val > max_conf:
                            max_conf = conf_val
                        labels.add(detector._model.names[cls])
                if labels:
                    break
        elapsed = time.perf_counter() - start
        return ModelRun(
            label="",
            model_path="",
            device=detector.device,
            detected=bool(labels),
            labels=labels,
            max_conf=max_conf,
            elapsed=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("  Error on %s: %s", video.name, exc)
        return ModelRun(
            label="",
            model_path="",
            device=None,
            detected=False,
            labels=set(),
            max_conf=0.0,
            elapsed=elapsed,
            error=str(exc),
        )


def _build_summary(runs: list[ModelRun], label: str, model_path: str, device: str | None) -> ModelSummary:
    total = len(runs)
    detected = sum(1 for r in runs if r.detected)
    errors = sum(1 for r in runs if r.error)
    no_object = total - detected - errors
    classes_found: set[str] = set()
    confs: list[float] = []
    times: list[float] = []
    for r in runs:
        if r.detected:
            classes_found |= r.labels
            confs.append(r.max_conf)
        if not r.error:
            times.append(r.elapsed)
    return ModelSummary(
        label=label,
        model_path=model_path,
        device=device,
        total=total,
        detected=detected,
        no_object=no_object,
        errors=errors,
        total_sec=sum(times),
        avg_sec=sum(times) / len(times) if times else 0.0,
        min_sec=min(times) if times else 0.0,
        max_sec=max(times) if times else 0.0,
        classes_found=classes_found,
        avg_max_conf=sum(confs) / len(confs) if confs else 0.0,
    )


def _print_results_table(video_runs: list[tuple[Path, list[ModelRun]]], model_labels: list[str], no_errors: bool) -> None:
    if no_errors:
        return
    header = f"{'Video':>30s} | " + " | ".join(f"{lbl:>14s}" for lbl in model_labels)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for video, runs in video_runs:
        cells: list[str] = []
        for r in runs:
            if r.error:
                cells.append(f"{'ERR':>14s}")
            elif r.detected:
                cells.append(f"{r.elapsed:>5.1f}s DET{len(r.labels):>3d}{r.max_conf:>4.1f}")
            else:
                cells.append(f"{r.elapsed:>5.1f}s ---")
        print(f"{video.name:>30s} | " + " | ".join(cells))


def _find_disagreements(video_runs: list[tuple[Path, list[ModelRun]]], model_labels: list[str]) -> list[str]:
    lines: list[str] = []
    for video, runs in video_runs:
        detecting = [r for r in runs if r.detected]
        missing = [r for r in runs if not r.detected and not r.error]
        if detecting and missing:
            found_by = ", ".join(
                f"{r.label}({', '.join(sorted(r.labels))})" for r in detecting
            )
            missed_by = ", ".join(r.label for r in missing)
            lines.append(f"  {video.name}: found by [{found_by}], missed by [{missed_by}]")
    return lines


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src_dir = Path(args.src_dir).expanduser()
    if not src_dir.is_dir():
        logger.error("Source directory '%s' does not exist.", src_dir)
        sys.exit(1)

    model_specs = [_parse_model_spec(m) for m in args.model]
    class_names = tuple(args.classes)

    videos = find_videos(src_dir, args.recursive)
    if not videos:
        logger.error("No videos found in %s", src_dir)
        sys.exit(1)

    logger.info("Benchmark: %d videos × %d models", len(videos), len(model_specs))

    all_runs: dict[int, list[ModelRun]] = {}  # model_index -> list of runs
    summaries: list[ModelSummary] = []

    for idx, (model_path, device) in enumerate(model_specs):
        resolved_device = device if device is not None else _resolve_device(model_path)
        label = _short_name(model_path)
        logger.info("Loading model %d/%d: %s (device: %s)...", idx + 1, len(model_specs), label, resolved_device or "auto")

        try:
            detector = _create_detector(model_path, resolved_device, class_names)
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_path, exc)
            sys.exit(1)

        runs: list[ModelRun] = []
        for i, video in enumerate(videos):
            if len(videos) > 1:
                logger.info("  [%s] %d/%d: %s", label, i + 1, len(videos), video.name)
            video_info = probe_video(video)
            if video_info is None:
                logger.warning("  Cannot read %s, skipping", video.name)
                runs.append(ModelRun(
                    label=label, model_path=model_path, device=resolved_device,
                    detected=False, labels=set(), max_conf=0.0, elapsed=0.0,
                    error="Cannot probe video",
                ))
                continue
            vid_stride = max(1, int(video_info.fps * args.sample_interval))
            run = _run_model(video, detector, args.conf, vid_stride)
            run.label = label
            run.model_path = model_path
            run.device = resolved_device
            if run.detected:
                logger.info("    [+] %s (%.2fs)", ", ".join(sorted(run.labels)), run.elapsed)
            elif run.error:
                logger.info("    [ERROR] %s", run.error)
            else:
                logger.info("    [-] no objects (%.2fs)", run.elapsed)
            runs.append(run)
        all_runs[idx] = runs
        summaries.append(_build_summary(runs, label, model_path, resolved_device))

    video_runs: list[tuple[Path, list[ModelRun]]] = []
    for v_idx, video in enumerate(videos):
        video_runs.append((video, [all_runs[m_idx][v_idx] for m_idx in range(len(model_specs))]))

    model_labels = [s.label for s in summaries]

    _print_results_table(video_runs, model_labels, args.no_errors)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for s in summaries:
        print(f"  {s.label}:")
        print(f"    Device:      {s.device or 'auto'}")
        print(f"    Videos:      {s.total} (detected={s.detected}, no-object={s.no_object}, errors={s.errors})")
        print(f"    Total time:  {s.total_sec:.1f}s")
        print(f"    Avg time:    {s.avg_sec:.2f}s/video  (min={s.min_sec:.2f}s, max={s.max_sec:.2f}s)")
        print(f"    Classes:     {', '.join(sorted(s.classes_found)) if s.classes_found else '(none)'}")
        print(f"    Avg max conf: {s.avg_max_conf:.2f}")
        print()

    if len(summaries) >= 2:
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                a, b = summaries[i], summaries[j]
                agree = sum(
                    1 for v in range(len(videos))
                    if all_runs[i][v].detected == all_runs[j][v].detected
                )
                agree_pct = agree / len(videos) * 100 if videos else 0
                ratio = a.total_sec / b.total_sec if b.total_sec else float("inf")
                print(f"  {a.label} vs {b.label}:")
                print(f"    Agreement:   {agree}/{len(videos)} ({agree_pct:.0f}%)")
                print(f"    Speed ratio: {ratio:.2f}x  ({a.label} is {'faster' if ratio < 1 else 'slower'})")
                det_ratio = (a.detected / len(videos)) / (b.detected / len(videos)) if b.detected else float("inf")
                print(f"    Detection ratio: {a.detected}/{b.detected} ({det_ratio:.2f}x)")
                print()

    disagreements = _find_disagreements(video_runs, model_labels)
    if disagreements:
        print("=" * 70)
        print("DISAGREEMENTS (videos where models differ)")
        print("=" * 70)
        for line in disagreements:
            print(line)
    else:
        print("(All models agree on every video.)")


if __name__ == "__main__":
    main()