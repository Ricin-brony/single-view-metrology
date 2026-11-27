"""Run single-view height measurement on a scene directory."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

from .geometry import (
    estimate_height,
    estimate_vanishing_point,
    focal_length_check,
    ground_vanishing_line,
)
from .markers import MarkerError, detect_scene
from .results import write_replay, write_results


DIRECTIONS = ("h1", "h2", "v")


def _read_text(path):
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read {path.name}: {error}") from error


def _read_rows(path, columns):
    rows = []
    for number, text in enumerate(_read_text(path).splitlines(), start=1):
        row = text.split("#", 1)[0].split()
        if row:
            if len(row) != columns:
                raise ValueError(f"{path.name}:{number} must have {columns} fields")
            rows.append(row)
    return rows


def _read_reference(scene):
    rows = _read_rows(scene / "reference.txt", 2)
    if len(rows) != 1:
        raise ValueError("reference.txt must contain one reference ID and height in meters")
    object_id, value = rows[0]
    try:
        height = float(value)
    except ValueError as error:
        raise ValueError("Reference height must be a finite positive number") from error
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Reference height must be a finite positive number")
    return object_id, height


def _coordinates(values, filename):
    try:
        points = np.asarray(values, dtype=float).reshape(2, 2)
    except ValueError as error:
        raise ValueError(f"{filename} has invalid pixel coordinates") from error
    if not np.isfinite(points).all():
        raise ValueError(f"{filename} requires finite pixel coordinates")
    return points.tolist()


def _read_objects(scene):
    objects = {}
    for object_id, *pixels in _read_rows(scene / "objects.txt", 5):
        if object_id in objects:
            raise ValueError(f"objects.txt contains duplicate object ID {object_id}")
        objects[object_id] = _coordinates(pixels, "objects.txt")
    return objects


def _read_line_groups(scene):
    groups = {label: [] for label in DIRECTIONS}
    for *pixels, label in _read_rows(scene / "lines.txt", 5):
        if label not in groups:
            raise ValueError(f"lines.txt contains unknown direction {label!r}")
        groups[label].append(_coordinates(pixels, "lines.txt"))
    return groups


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON number {value!r}")


def _read_json_float(token):
    value = float(token)
    if not np.isfinite(value):
        _reject_json_constant(token)
    return value


def _read_profile(scene):
    try:
        profile = json.loads(
            _read_text(scene / "rig.json"),
            parse_constant=_reject_json_constant,
            parse_float=_read_json_float,
        )
    except ValueError as error:
        raise ValueError(f"Cannot load rig.json: {error}") from error
    if not isinstance(profile, dict):
        raise ValueError("rig.json must contain an object")
    return profile


def _profile_object_ids(profile):
    roles = profile.get("objects")
    if not isinstance(roles, list):
        return []
    return list(dict.fromkeys(
        item["object_id"] for item in roles
        if isinstance(item, dict) and isinstance(item.get("object_id"), str)
    ))


def _read_image(scene):
    path = scene / "image.jpg"
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read image.jpg: {error}") from error
    try:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR) if content else None
    except cv2.error as error:
        raise ValueError("Cannot decode image.jpg") from error
    if image is None:
        raise ValueError("Cannot decode image.jpg")
    return image, hashlib.sha256(content).hexdigest()


def _fit_scene_geometry(groups, image_shape):
    points = {}
    masks = {}
    for label in DIRECTIONS:
        try:
            point, mask = estimate_vanishing_point(groups[label])
        except ValueError as error:
            raise ValueError(f"Cannot estimate {label} direction: {error}") from error
        points[label] = point
        masks[label] = np.asarray(mask, dtype=bool).tolist()
    ground = ground_vanishing_line(points["h1"], points["h2"])
    focal_length, dot = focal_length_check(points["h1"], points["h2"], image_shape)
    return {
        "vanishing_points": {label: point.tolist() for label, point in points.items()},
        "ground_vanishing_line": ground.tolist(),
        "line_inliers": masks,
        "focal_length_check": {"focal_length_px": focal_length, "raw_dot_product": dot},
    }


def _pixel_height(reference, target):
    ref_span = np.linalg.norm(np.subtract(reference[1], reference[0]))
    target_span = np.linalg.norm(np.subtract(target[1], target[0]))
    if min(ref_span, target_span) <= 1e-9:
        raise ValueError("Pixel height requires distinct top and base points")
    return reference[2] * target_span / ref_span


def _measurement(object_id, height=None, reason=None):
    return {"object_id": object_id, "height_m": height,
            "status": "ok" if height is not None else "rejected", "reason": reason}


def measure_scene(scene_dir, output_dir, debug_dir=None, mode="manual"):
    """Measure from the scene inputs and save a record without loading truth.txt."""
    scene = Path(scene_dir)
    output = Path(output_dir)
    debug = Path(debug_dir) if debug_dir is not None else None
    if mode not in {"manual", "auto", "pixel"}:
        raise ValueError(f"Unknown measurement mode {mode!r}")
    if debug is not None and debug.resolve() == scene.resolve():
        raise ValueError("Replay directory must differ from the input scene directory")

    record = {
        "mode": mode,
        "status": "rejected",
        "source": None,
        "reference": None,
        "rig": None,
        "markers": None,
        "objects": {},
        "line_groups": {},
        "geometry": None,
        "measurements": [],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }
    image = None
    expected_targets = []
    try:
        image, checksum = _read_image(scene)
        height, width = image.shape[:2]
        record["source"] = {"filename": "image.jpg", "sha256": checksum, "width": width, "height": height}
        reference_id, reference_height = _read_reference(scene)
        record["reference"] = {"object_id": reference_id, "height_m": reference_height}
        if mode == "auto":
            profile = _read_profile(scene)
            record["rig"] = profile
            expected_targets = [object_id for object_id in _profile_object_ids(profile) if object_id != reference_id]
            objects, groups, record["markers"] = detect_scene(image, profile)
        else:
            objects = _read_objects(scene)
            expected_targets = [object_id for object_id in objects if object_id != reference_id]
            groups = _read_line_groups(scene) if mode == "manual" else {}
        if reference_id not in objects:
            raise ValueError(f"Reference ID {reference_id!r} has no annotated object")
        if not expected_targets:
            raise ValueError("The scene has no target objects")

        record["objects"] = {name: np.asarray(points, dtype=float).tolist() for name, points in objects.items()}
        record["line_groups"] = {name: np.asarray(lines, dtype=float).tolist() for name, lines in groups.items()}
        if mode != "pixel":
            record["geometry"] = _fit_scene_geometry(groups, image.shape)

        reference = (*objects[reference_id], reference_height)
        for object_id in expected_targets:
            try:
                if mode == "pixel":
                    estimated = _pixel_height(reference, objects[object_id])
                else:
                    geometry = record["geometry"]
                    estimated = estimate_height(geometry["vanishing_points"]["v"],
                                                geometry["ground_vanishing_line"],
                                                reference, objects[object_id])
                if not np.isfinite(estimated) or estimated <= 0:
                    raise ValueError("The estimated height is not finite and positive")
                record["measurements"].append(_measurement(object_id, float(estimated)))
            except ValueError as error:
                record["measurements"].append(_measurement(object_id, reason=str(error)))
        successes = sum(row["status"] == "ok" for row in record["measurements"])
        record["status"] = "ok" if successes == len(expected_targets) else "partial" if successes else "rejected"
    except ValueError as error:
        if isinstance(error, MarkerError):
            record["markers"] = error.diagnostics
        record["measurements"] = [
            _measurement(object_id, reason=str(error)) for object_id in (expected_targets or [""])
        ]

    write_results(output, image, record)
    if debug is not None and record["objects"] and record["line_groups"]:
        write_replay(scene, debug, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate object heights from a photo and a ruler reference."
    )
    parser.add_argument("--mode", choices=("manual", "auto", "pixel"), default="manual")
    parser.add_argument("--input", type=Path, required=True, help="Scene directory")
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--save-debug", type=Path, help="Write the annotations as a replayable scene")
    args = parser.parse_args(argv)
    try:
        record = measure_scene(args.input, args.output, args.save_debug, args.mode)
    except (OSError, ValueError) as error:
        print(f"Measurement could not run: {error}", file=sys.stderr)
        return 1
    if record["status"] != "ok":
        for row in record["measurements"]:
            if row["status"] == "rejected":
                print(row["reason"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
