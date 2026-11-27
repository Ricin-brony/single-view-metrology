"""Compare pixel, manual, and automatic measurements against separate ground truth."""

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median
import sys

from .metrology import measure_scene


def _read_truth(path):
    heights = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        row = raw.split("#", 1)[0].split()
        if not row:
            continue
        try:
            object_id, value = row
            height = float(value)
        except ValueError as error:
            raise ValueError(f"{path}:{number} requires an object ID and height in meters") from error
        if object_id in heights or not isfinite(height) or height <= 0:
            raise ValueError(f"{path}:{number} requires a unique object ID and a finite positive height")
        heights[object_id] = height
    if not heights:
        raise ValueError(f"{path} contains no target heights")
    return heights


def evaluate_scenes(scene_dirs, output_dir, modes=("pixel", "manual", "auto")):
    """Run all methods on the same scenes and summarize only against truth.txt."""
    scenes = [Path(scene) for scene in scene_dirs]
    modes = tuple(dict.fromkeys(modes))
    if not scenes or not modes:
        raise ValueError("Evaluation requires at least one scene and one mode")
    output = Path(output_dir)
    rows = []
    for index, scene in enumerate(scenes, start=1):
        case_id = f"{index:03d}-{scene.name}"
        records = {mode: measure_scene(scene, output / "runs" / case_id / mode, mode=mode)
                   for mode in modes}
        # This file is consumed here, after all methods have produced estimates.
        truth = _read_truth(scene / "truth.txt")
        for mode, record in records.items():
            estimates = {row["object_id"]: row for row in record["measurements"]}
            for object_id, expected in truth.items():
                estimate = estimates.get(object_id) or estimates.get("")
                height = estimate["height_m"] if estimate is not None else None
                reason = estimate["reason"] if estimate is not None else "No result for this target ID"
                absolute = abs(height - expected) if height is not None else None
                rows.append({
                    "case_id": case_id, "scene": scene.name, "mode": mode, "object_id": object_id,
                    "height_m": height, "truth_m": expected, "absolute_error_m": absolute,
                    "absolute_relative_error": absolute / expected if absolute is not None else None,
                    "status": "ok" if height is not None else "rejected", "reason": reason,
                })

    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"scene_count": len(scenes), "modes": {}}
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        accepted = [row for row in subset if row["status"] == "ok"]
        summary["modes"][mode] = {
            "attempted": len(subset), "successful": len(accepted),
            "success_rate": len(accepted) / len(subset),
            "mean_absolute_error_m": mean(row["absolute_error_m"] for row in accepted) if accepted else None,
            "median_absolute_relative_error": (
                median(row["absolute_relative_error"] for row in accepted) if accepted else None
            ),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate generated or captured scenes with known heights.")
    parser.add_argument("--scene", type=Path, action="append", required=True, help="Scene directory. Repeat per scene.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", action="append", choices=("pixel", "manual", "auto"), help="Method to include. Default: all.")
    args = parser.parse_args(argv)
    try:
        summary = evaluate_scenes(args.scene, args.output, args.mode or ("pixel", "manual", "auto"))
    except (OSError, ValueError) as error:
        print(f"Evaluation could not run: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
