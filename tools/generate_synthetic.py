"""Render reproducible ArUco scenes with an independent pinhole camera.

Run from the repository root with ``python -m tools.generate_synthetic --output
<directory>``. No measurement code or estimated height is used by this renderer.
All geometry is in meters, with world Z upward and camera depth positive.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (2000, 1400)
DICTIONARY_NAME = "DICT_4X4_50"
GROUND_RECTANGLE = np.array([
    [-1.10, 1.90, 0.0],  # 0: rear left
    [1.00, 1.90, 0.0],   # 1: rear right
    [1.00, 0.40, 0.0],   # 2: front right
    [-1.10, 0.40, 0.0],  # 3: front left
])


def _camera(pitch_degrees=0.0, yaw_degrees=0.0, roll_degrees=0.0):
    pitch, yaw, roll = np.deg2rad([pitch_degrees, yaw_degrees, roll_degrees])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    forward = np.array([np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])
    down = np.cross(forward, right)
    rotation = np.array([np.cos(roll) * right + np.sin(roll) * down,
                         -np.sin(roll) * right + np.cos(roll) * down,
                         forward])
    return {
        "intrinsic_matrix": [[1600.0, 0.0, 1000.0],
                             [0.0, 1600.0, 650.0], [0.0, 0.0, 1.0]],
        "world_to_camera_rotation": rotation.tolist(),
        "position_world_m": [0.0, -2.8, 1.0],
        "pitch_degrees": pitch_degrees,
        "yaw_degrees": yaw_degrees,
        "roll_degrees": roll_degrees,
    }


def _project(points, camera):
    points = np.asarray(points, dtype=float)
    rotation = np.asarray(camera["world_to_camera_rotation"])
    position = np.asarray(camera["position_world_m"])
    intrinsic = np.asarray(camera["intrinsic_matrix"])
    camera_points = (rotation @ (points - position).T).T
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("The synthetic scene contains points behind the camera")
    homogeneous = (intrinsic @ camera_points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _marker_corners(origin, right, down):
    """Return black-border corners in ArUco's canonical 0, 1, 2, 3 order."""
    origin, right, down = map(np.asarray, (origin, right, down))
    return np.array([origin, origin + right, origin + right + down, origin + down])


def _paint_marker(image, marker_id, world_corners, camera, dictionary):
    # Quiet white margins extend outside the coded square. Annotation endpoints
    # remain at the black-border corners, not at the paper edge or tag center.
    marker_pixels, quiet = 180, 24
    coded = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
    paper = cv2.copyMakeBorder(coded, quiet, quiet, quiet, quiet,
                               cv2.BORDER_CONSTANT, value=255)
    paper = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    source = np.float32([[quiet, quiet], [quiet + marker_pixels - 1, quiet],
                         [quiet + marker_pixels - 1, quiet + marker_pixels - 1],
                         [quiet, quiet + marker_pixels - 1]])
    destination = _project(world_corners, camera).astype(np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(paper, transform, size, flags=cv2.INTER_NEAREST)
    mask = cv2.warpPerspective(np.full(paper.shape[:2], 255, dtype=np.uint8),
                               transform, size, flags=cv2.INTER_NEAREST)
    image[mask != 0] = warped[mask != 0]


def _write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_scene(scene_dir, name, camera, target_depth, missing_ids=()):
    scene_dir.mkdir(parents=True, exist_ok=True)
    width, height = IMAGE_SIZE
    image = np.full((height, width, 3), (246, 244, 240), dtype=np.uint8)
    rectangle_pixels = np.rint(_project(GROUND_RECTANGLE, camera)).astype(np.int32)
    cv2.fillConvexPoly(image, rectangle_pixels, (225, 232, 231))
    cv2.polylines(image, [rectangle_pixels], True, (155, 164, 163), 2, cv2.LINE_AA)

    world_objects = {
        "reference": [[-0.47, 0.85, 0.0], [-0.47, 0.85, 0.20]],
        "target": [[0.32, target_depth, 0.0], [0.32, target_depth, 0.30]],
    }
    image_objects = {object_id: _project(points, camera)
                     for object_id, points in world_objects.items()}
    for endpoints in image_objects.values():
        first, second = np.rint(endpoints).astype(int)
        cv2.line(image, tuple(first), tuple(second), (100, 108, 112), 3, cv2.LINE_AA)

    markers = {}
    ground_right, ground_down = np.array([0.20, 0.0, 0.0]), np.array([0.0, -0.20, 0.0])
    for marker_id, center in enumerate(GROUND_RECTANGLE):
        markers[marker_id] = _marker_corners(center - (ground_right + ground_down) / 2,
                                              ground_right, ground_down)
    for object_id, (base_id, top_id) in (("reference", (4, 5)), ("target", (6, 7))):
        base, top = world_objects[object_id]
        markers[base_id] = _marker_corners(base, [0.18, 0, 0], [0, -0.18, 0])
        markers[top_id] = _marker_corners(top, [0.12, 0, 0], [0, 0, -0.12])

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Far surfaces are drawn first. All role markers have physically distinct
    # footprints, so the chosen measurement corners are not occluded.
    for marker_id in sorted(markers, key=lambda i: -np.mean(markers[i][:, 1])):
        if marker_id not in missing_ids:
            _paint_marker(image, marker_id, markers[marker_id], camera, dictionary)

    encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    if not encoded:
        raise RuntimeError("OpenCV could not encode the synthetic JPG")
    # Path.write_bytes also works with Chinese Windows output directory names.
    (scene_dir / "image.jpg").write_bytes(buffer.tobytes())
    (scene_dir / "reference.txt").write_text("reference 0.20\n", encoding="utf-8")
    (scene_dir / "truth.txt").write_text("target 0.30\n", encoding="utf-8")
    object_rows = [object_id + " " + " ".join(f"{value:.9f}" for value in points.ravel())
                   for object_id, points in image_objects.items()]
    (scene_dir / "objects.txt").write_text("\n".join(object_rows) + "\n", encoding="utf-8")

    lines = []
    for first, second, label in ((0, 1, "h1"), (3, 2, "h1"),
                                  (0, 3, "h2"), (1, 2, "h2")):
        lines.append((_project(GROUND_RECTANGLE[[first, second]], camera), label))
    lines.extend((points, "v") for points in image_objects.values())
    line_rows = [" ".join(f"{value:.9f}" for value in points.ravel()) + " " + label
                 for points, label in lines]
    (scene_dir / "lines.txt").write_text("\n".join(line_rows) + "\n", encoding="utf-8")

    _write_json(scene_dir / "rig.json", {
        "dictionary": DICTIONARY_NAME,
        "plane_tag_ids": [0, 1, 2, 3],
        "objects": [
            {"object_id": "reference", "base": {"id": 4, "corner": 0},
             "top": {"id": 5, "corner": 0}},
            {"object_id": "target", "base": {"id": 6, "corner": 0},
             "top": {"id": 7, "corner": 0}},
        ],
    })
    _write_json(scene_dir / "scene.json", {
        "scene_id": name,
        "split": "synthetic",
        "source": "Deterministic independent pinhole-camera rendering with OpenCV ArUco tags",
        "attribution": "Generated by tools.generate_synthetic; no external photographs",
        "image_dimensions": {"width": width, "height": height},
        "orientation_history": [],
        "resize_history": [],
        "camera": camera,
        "world": {"ground_rectangle_m": GROUND_RECTANGLE.tolist(), "objects": world_objects},
        "rendered_marker_ids": [i for i in range(8) if i not in missing_ids],
        "expected_missing_ids": list(missing_ids),
        "reference_origin": "Known 0.20 m world-space vertical segment",
        "truth_origin": "Known 0.30 m world-space vertical segment; held out from measurement",
    })


def generate_scenes(output_dir):
    """Create three deterministic scene bundles and return their directory paths."""
    output_dir = Path(output_dir)
    scenes = {
        "level_equal_depth": (_camera(), 0.85, ()),
        "pitched_staggered": (_camera(8.0, 8.0, 3.0), 1.55, ()),
        "missing_marker": (_camera(8.0, 8.0, 3.0), 1.55, (7,)),
    }
    result = {}
    for name, (camera, target_depth, missing_ids) in scenes.items():
        result[name] = output_dir / name
        _write_scene(result[name], name, camera, target_depth, missing_ids)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Parent directory for scene bundles")
    arguments = parser.parse_args(argv)
    for name, directory in generate_scenes(arguments.output).items():
        print(f"Generated {name}: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
