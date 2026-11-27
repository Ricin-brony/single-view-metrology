"""Read the geometric inputs of a prepared scene from printed ArUco tags.

This module detects image geometry only. It does not consume marker lengths,
object heights, camera calibration, or evaluation ground truth.
"""

from collections import Counter

import cv2
import numpy as np


class MarkerError(ValueError):
    """A scene cannot provide its required marker geometry."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


def _diagonal_center(corners):
    """Projective center of a square: intersect its opposite-corner diagonals."""
    homogeneous = np.column_stack((corners, np.ones(4)))
    intersection = np.cross(
        np.cross(homogeneous[0], homogeneous[2]),
        np.cross(homogeneous[1], homogeneous[3]),
    )
    if not np.isfinite(intersection).all() or abs(intersection[2]) < 1e-12:
        raise MarkerError("A detected marker has invalid diagonals",
                          {"reason": "invalid_marker_geometry"})
    return (intersection[:2] / intersection[2]).tolist()


def _profile_error(message, field):
    raise MarkerError(message, {"reason": "invalid_profile", "field": field})


def _validate_profile(profile):
    if not isinstance(profile, dict):
        _profile_error("Marker profile must be an object", "profile")
    name = profile.get("dictionary")
    if not isinstance(name, str) or not name.startswith("DICT_"):
        _profile_error("dictionary must name a predefined ArUco dictionary", "dictionary")
    code = getattr(cv2.aruco, name, None)
    if type(code) is not int:
        _profile_error("Unknown predefined ArUco dictionary", "dictionary")
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(code)
    except cv2.error:
        _profile_error("Unsupported predefined ArUco dictionary", "dictionary")
    limit = len(dictionary.bytesList)
    plane_ids = profile.get("plane_tag_ids")
    if not isinstance(plane_ids, list) or len(plane_ids) != 4:
        _profile_error("plane_tag_ids must contain four IDs in cyclic order", "plane_tag_ids")

    used_ids = set()

    def register(marker_id, field):
        # bool and 1.0 must not silently alias the integer ID 1.
        if type(marker_id) is not int or not 0 <= marker_id < limit:
            _profile_error(f"Marker ID must be an integer from 0 to {limit - 1}", field)
        if marker_id in used_ids:
            _profile_error("Each marker ID must have exactly one configured role", field)
        used_ids.add(marker_id)

    for index, marker_id in enumerate(plane_ids):
        register(marker_id, f"plane_tag_ids[{index}]")
    items = profile.get("objects")
    if not isinstance(items, list) or not items:
        _profile_error("objects must be a nonempty list", "objects")
    object_ids = set()
    for index, item in enumerate(items):
        field = f"objects[{index}]"
        if not isinstance(item, dict):
            _profile_error("Each object must be an object with endpoint roles", field)
        object_id = item.get("object_id")
        if (not isinstance(object_id, str) or not object_id or object_id in object_ids
                or any(character.isspace() for character in object_id) or "#" in object_id):
            _profile_error("object_id must be unique and contain no whitespace or #", field + ".object_id")
        object_ids.add(object_id)
        for role in ("base", "top"):
            endpoint = item.get(role)
            if not isinstance(endpoint, dict):
                _profile_error("Each endpoint needs an ID and canonical corner", field + "." + role)
            register(endpoint.get("id"), field + "." + role + ".id")
            corner = endpoint.get("corner")
            if type(corner) is not int or not 0 <= corner <= 3:
                _profile_error("Endpoint corner must be an integer from 0 to 3",
                               field + "." + role + ".corner")
    return name, dictionary, plane_ids, items, used_ids


def _grayscale(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or image.ndim not in (2, 3)
            or (image.ndim == 3 and image.shape[2] not in (3, 4))):
        raise MarkerError("image must be a nonempty uint8 grayscale, BGR, or BGRA array",
                          {"reason": "invalid_image"})
    image = np.ascontiguousarray(image)
    if image.ndim == 2:
        return image
    conversion = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
    return cv2.cvtColor(image, conversion)


def _validate_plane(centers, diagnostics):
    points = np.asarray(centers, dtype=float)
    points = (points - points.mean(axis=0)) / max(float(np.ptp(points, axis=0).max()), 1.0)
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(turns > 1e-8) or np.all(turns < -1e-8)):
        diagnostics["reason"] = "invalid_plane"
        raise MarkerError("Plane marker centers must form a nondegenerate convex cyclic quadrilateral",
                          diagnostics)


def _validate_axes(objects, image_shape, diagnostics):
    # Condition coordinates so the coincidence tolerance does not depend on the
    # image's pixel count or the distance of an axis from the pixel origin.
    scale = float(max(image_shape))
    origin = np.array([image_shape[1], image_shape[0]], dtype=float) / 2
    axes = []
    for object_id, endpoints in objects.items():
        points = (np.asarray(endpoints, dtype=float) - origin) / scale
        line = np.cross(np.append(points[0], 1.0), np.append(points[1], 1.0))
        length = float(np.linalg.norm(line[:2]))
        if length <= 1e-9:
            diagnostics.update({"reason": "degenerate_axis", "object_id": object_id})
            raise MarkerError("Object base and top must be distinct image points", diagnostics)
        line /= np.linalg.norm(line)
        for previous_id, previous in axes:
            if np.linalg.norm(np.cross(line, previous)) <= 1e-7:
                diagnostics.update({"reason": "coincident_vertical_axes",
                                    "object_ids": [previous_id, object_id]})
                raise MarkerError("Object vertical axes coincide and cannot independently define a direction",
                                  diagnostics)
        axes.append((object_id, line))


def detect_scene(image, profile):
    """Return ``(objects, h1/h2/v line groups, JSON-safe diagnostics)``.

    Endpoint corners follow the marker's canonical printed orientation, as
    returned by OpenCV, rather than the screen's top-left corner.
    """
    name, dictionary, plane_ids, items, required = _validate_profile(profile)
    gray = _grayscale(image)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, rejected = detector.detectMarkers(gray)
    detected_ids = [] if ids is None else [int(value) for value in ids.flatten()]
    markers = [
        {"id": marker_id, "corners": points.reshape(4, 2).astype(float).tolist(),
         "center": _diagonal_center(points.reshape(4, 2))}
        for marker_id, points in zip(detected_ids, corners)
    ]
    diagnostics = {
        "dictionary": name, "detected_ids": detected_ids,
        "markers": markers, "rejected_candidates": len(rejected),
    }
    diagnostics["required_ids"] = sorted(required)
    counts = Counter(detected_ids)
    missing = sorted(required - counts.keys())
    duplicated = sorted(marker_id for marker_id in required if counts[marker_id] > 1)
    if missing or duplicated:
        diagnostics.update({"missing_ids": missing, "duplicate_ids": duplicated})
        reason = "duplicate_markers" if duplicated else "missing_markers"
        diagnostics["reason"] = reason
        raise MarkerError(
            f"Missing required markers: {missing}; duplicate required markers: {duplicated}",
            diagnostics,
        )
    by_id = {item["id"]: item for item in markers}
    centers = [by_id[marker_id]["center"] for marker_id in plane_ids]
    diagnostics.update({"plane_tag_ids": list(plane_ids), "plane_centers": centers})
    _validate_plane(centers, diagnostics)
    objects = {}
    for item in items:
        objects[item["object_id"]] = tuple(
            by_id[item[role]["id"]]["corners"][item[role]["corner"]]
            for role in ("base", "top")
        )
    diagnostics["derived_objects"] = objects
    _validate_axes(objects, gray.shape, diagnostics)
    groups = {
        "h1": [(centers[0], centers[1]), (centers[3], centers[2])],
        "h2": [(centers[0], centers[3]), (centers[1], centers[2])],
        "v": list(objects.values()),
    }
    diagnostics.update({
        "plane_tag_ids": list(plane_ids), "plane_centers": centers,
        "derived_objects": objects, "line_groups": groups,
    })
    return objects, groups, diagnostics
