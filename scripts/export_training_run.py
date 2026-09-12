#!/usr/bin/env python3
"""Export compact, reviewable metrics from one logical training run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METRICS = (
    "actor/distillation/loss",
    "actor/distillation/abs_loss",
    "actor/grad_norm",
    "actor/lr",
    "perf/mfu/actor",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "perf/total_num_tokens",
    "perf/throughput",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "training/num_turns/mean",
    "training/off_policy/trajectory_staleness/mean",
    "training/off_policy/trajectory_staleness/max",
)
FIELD_RE = re.compile(r"(?:^| - )([^: ]+):([^ ]+)")


def number(value: str) -> float | int | None:
    for wrapper in ("np.float64(", "np.int64("):
        if value.startswith(wrapper) and value.endswith(")"):
            value = value[len(wrapper) : -1]
            break
    try:
        parsed = float(value)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def read_metrics(log_path: Path) -> dict[int, dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    if not log_path.is_file():
        return rows
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if "training/global_step:" not in line:
                continue
            fields = dict(FIELD_RE.findall(line))
            step = number(fields.get("training/global_step", ""))
            if not isinstance(step, int):
                continue
            row = {"step": step}
            for key in METRICS:
                value = number(fields.get(key, ""))
                if value is not None:
                    row[key] = value
            rows[step] = row
    return rows


def read_profile(profile_dir: Path) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "max_s": 0.0, "errors": 0}
    )
    if not profile_dir.is_dir():
        return stats
    for path in sorted(profile_dir.glob("*.jsonl")):
        with path.open(errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = event.get("event")
                duration = event.get("duration_s")
                if not isinstance(name, str) or not isinstance(duration, (int, float)):
                    continue
                item = stats[name]
                item["count"] += 1
                item["total_s"] += duration
                item["max_s"] = max(item["max_s"], duration)
                item["errors"] += event.get("status") not in (None, "ok")
    return stats


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_loss_curve(path: Path, rows: list[dict]) -> None:
    points = [
        (row["step"], row["actor/distillation/loss"])
        for row in rows
        if "actor/distillation/loss" in row
    ]
    if not points:
        return

    width, height = 1000, 500
    left, top, right, bottom = 80, 45, 25, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    steps, losses = zip(*points)
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if min_loss == max_loss:
        min_loss -= 0.5
        max_loss += 0.5

    def xy(step: float, loss: float) -> tuple[float, float]:
        x_span = max(max_step - min_step, 1)
        x = left + (step - min_step) / x_span * plot_width
        y = top + (max_loss - loss) / (max_loss - min_loss) * plot_height
        return x, y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for index in range(6):
        fraction = index / 5
        y = top + fraction * plot_height
        value = max_loss - fraction * (max_loss - min_loss)
        draw.line((left, y, width - right, y), fill="#dddddd", width=1)
        draw.text((8, y - 8), f"{value:.4f}", fill="#444444", font=font)
    draw.line((left, top, left, height - bottom), fill="#555555", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#555555", width=2)
    draw.line([xy(step, loss) for step, loss in points], fill="#2878b5", width=2)
    draw.text((left, 12), "Distillation loss", fill="#222222", font=font)
    draw.text((left, height - 38), f"step {min_step}", fill="#444444", font=font)
    end_label = f"step {max_step}"
    end_width = draw.textbbox((0, 0), end_label, font=font)[2]
    draw.text((width - right - end_width, height - 38), end_label, fill="#444444", font=font)

    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--status", choices=("completed", "failed"), required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--git-commit", default="")
    args = parser.parse_args()

    args.record_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.record_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    segments = summary.setdefault("segments", [])
    segment = {
        "name": args.segment,
        "log": str(args.log),
        "profile_dir": str(args.profile_dir) if args.profile_dir else "",
        "status": args.status,
        "resume_from": args.resume_from,
    }
    segments[:] = [item for item in segments if item.get("name") != args.segment]
    segments.append(segment)

    metrics: dict[int, dict[str, float | int]] = {}
    profile_rows = []
    for item in segments:
        metrics.update(read_metrics(Path(item["log"])))
        profile_path = item.get("profile_dir")
        if not profile_path:
            continue
        for event, values in sorted(read_profile(Path(profile_path)).items()):
            profile_rows.append(
                {
                    "segment": item["name"],
                    "event": event,
                    **values,
                    "mean_s": values["total_s"] / values["count"],
                }
            )

    metric_rows = [metrics[step] for step in sorted(metrics)]
    write_csv(args.record_dir / "metrics.csv", ["step", *METRICS], metric_rows)
    write_loss_curve(args.record_dir / "loss_curve.png", metric_rows)
    write_csv(
        args.record_dir / "profile_summary.csv",
        ["segment", "event", "count", "total_s", "mean_s", "max_s", "errors"],
        profile_rows,
    )

    summary.update(
        {
            "algorithm": args.algorithm,
            "status": args.status,
            "git_commit": args.git_commit,
            "checkpoint_dir": args.checkpoint_dir,
            "steps": len(metric_rows),
            "last_step": metric_rows[-1]["step"] if metric_rows else None,
            "final_loss": metric_rows[-1].get("actor/distillation/loss") if metric_rows else None,
        }
    )
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)


if __name__ == "__main__":
    main()
