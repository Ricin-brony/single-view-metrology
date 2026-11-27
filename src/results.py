"""Write measurement summaries, geometry overlays, and replay inputs."""

import csv
import json
from shutil import copyfile

import cv2
import numpy as np


_DIRECTION_COLORS = {"h1": (250, 120, 30), "h2": (0, 170, 230), "v": (180, 60, 180)}


def _pixel(point):
    # OpenCV accepts signed 32-bit coordinates before clipping.
    return tuple(int(round(value)) for value in np.clip(point, -(2**29), 2**29))


def _draw_segment(image, start, end, color, width=2):
    height, columns = image.shape[:2]
    visible, p1, p2 = cv2.clipLine((0, 0, columns, height), _pixel(start), _pixel(end))
    if visible:
        cv2.line(image, p1, p2, color, width, cv2.LINE_AA)


def _draw_ground_line(image, ground):
    a, b, c = ground
    height, width = image.shape[:2]
    if abs(a) + abs(b) < 1e-12:
        return  # The ground line is at infinity.
    if abs(a) >= abs(b):
        start, end = (-c / a, 0), ((-c - b * (height - 1)) / a, height - 1)
    else:
        start, end = (0, -c / b), (width - 1, (-c - a * (width - 1)) / b)
    _draw_segment(image, start, end, (40, 210, 80), 2)


def _label(image, text, point, color):
    position = _pixel(point)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_vanishing_points(image, points):
    height, width = image.shape[:2]
    for index, (label, (x, y, w)) in enumerate(points.items()):
        color = _DIRECTION_COLORS[label]
        if abs(w) > 1e-12:
            location = (x / w, y / w)
            if 0 <= location[0] < width and 0 <= location[1] < height:
                cv2.circle(image, _pixel(location), 5, color, -1, cv2.LINE_AA)
                _label(image, label, (location[0] + 8, location[1] - 5), color)
            continue
        start = np.array([110, 56 + index * 34])
        direction = np.array([x, y])
        direction /= np.linalg.norm(direction)
        cv2.arrowedLine(image, _pixel(start), _pixel(start + 34 * direction), color, 2, cv2.LINE_AA)
        _label(image, f"{label} at infinity", (12, start[1] + 14), color)


def _draw_transferred_height(image, record, reference_points, target_points):
    # Extend the line between the feet to the ground horizon, and use its
    # vanishing point to place the reference top over the target foot.
    ref_base, ref_top = (np.array([*point, 1.0]) for point in reference_points)
    base, top = (np.array([*point, 1.0]) for point in target_points)
    ground = record["geometry"]["ground_vanishing_line"]
    depth_direction = np.cross(np.cross(ref_base, base), ground)
    transferred = np.cross(np.cross(ref_top, depth_direction), np.cross(base, top))
    if abs(transferred[2]) > 1e-12:
        transferred_top = transferred[:2] / transferred[2]
        _draw_segment(image, ref_top[:2], transferred_top, (240, 200, 80), 1)
        _draw_segment(image, base[:2], transferred_top, (240, 200, 80), 1)


def _overlay(image, record):
    canvas = image.copy()
    geometry = record["geometry"]
    if geometry is not None:
        _draw_ground_line(canvas, geometry["ground_vanishing_line"])
        _draw_vanishing_points(canvas, geometry["vanishing_points"])
    for label, segments in record["line_groups"].items():
        for start, end in segments:
            _draw_segment(canvas, start, end, _DIRECTION_COLORS[label], 1)

    reference = record["reference"]
    reference_id = reference["object_id"] if reference is not None else None
    objects = record["objects"]
    rows = {row["object_id"]: row for row in record["measurements"]}
    for object_id, (base, top) in objects.items():
        measured = rows[object_id]["height_m"] if object_id in rows else None
        if geometry is not None and measured is not None and reference_id in objects:
            _draw_transferred_height(canvas, record, objects[reference_id], (base, top))
        display_height = reference["height_m"] if object_id == reference_id else measured
        color = (60, 210, 80) if display_height is not None else (55, 70, 235)
        _draw_segment(canvas, base, top, color, 3)
        for point in (base, top):
            cv2.circle(canvas, _pixel(point), 4, color, -1, cv2.LINE_AA)
        text = f"{object_id} {display_height * 100:.1f} cm" if display_height is not None else object_id
        _label(canvas, text, (top[0] + 8, top[1] - 8), color)

    _label(canvas, f"{record['mode']}  {record['status']}", (12, 24), (245, 245, 245))
    return canvas


def write_results(output, image, record):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "heights.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("object_id", "height_m", "status", "reason"))
        writer.writeheader()
        writer.writerows(record["measurements"])
    (output / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    path = output / "overlay.png"
    if image is None:
        path.unlink(missing_ok=True)
        return
    success, encoded = cv2.imencode(".png", _overlay(image, record))
    if not success:
        raise ValueError("Cannot encode overlay.png")
    path.write_bytes(encoded.tobytes())


def _format_points(points):
    return " ".join(f"{value:.17g}" for point in points for value in point)


def write_replay(scene, replay, record):
    replay.mkdir(parents=True, exist_ok=True)
    copyfile(scene / "image.jpg", replay / "image.jpg")
    copyfile(scene / "reference.txt", replay / "reference.txt")
    objects = [f"{object_id} {_format_points(points)}" for object_id, points in record["objects"].items()]
    lines = [f"{_format_points(segment)} {label}"
             for label, segments in record["line_groups"].items() for segment in segments]
    (replay / "objects.txt").write_text("\n".join(objects) + "\n", encoding="utf-8")
    (replay / "lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
