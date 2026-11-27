"""CPU-only projective geometry for single-view height measurement.

Pixels use (x, y); homogeneous points and lines use three components. Points
at infinity remain homogeneous throughout estimation. No focal length,
target dimensions, tag size, or trained model supplies the measurement scale.
"""

from itertools import combinations
import math

import numpy as np


_EPS = 1e-12
_VERTICAL_ANGLE = math.radians(2.0)


def _vector3(value, name):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite three-vector") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError(f"{name} cannot be the zero homogeneous vector")
    vector = vector / scale
    vector = vector / np.linalg.norm(vector)
    if vector[np.argmax(np.abs(vector))] < 0:
        vector = -vector
    return vector


def _point(value, name="point"):
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have two or three finite coordinates") from exc
    if point.shape == (2,):
        point = np.r_[point, 1.0]
    return _vector3(point, name)


def _pixel_point(value, name):
    point = _point(value, name)
    if point[2] == 0:
        raise ValueError(f"{name} must be a finite image point")
    with np.errstate(over="ignore", invalid="ignore"):
        point = point / point[2]
    if not np.isfinite(point).all():
        raise ValueError(f"{name} has unrepresentable pixel coordinates")
    return point


def _line_scale(line):
    line = _vector3(line, "line")
    normal_length = np.linalg.norm(line[:2])
    # A horizon at infinity is valid; signed pixel distance is then inapplicable.
    if normal_length > _EPS:
        line = line / normal_length
    return line


def line_through(p, q):
    """Return p x q for distinct pixel-2 or homogeneous-3 points.

    Finite lines have unit (a,b) normal. The line at infinity is retained.
    """
    first, second = _point(p), _point(q)
    line = np.cross(first, second)
    if np.linalg.norm(line) <= _EPS:
        raise ValueError("two distinct points are required to define a line")
    return _line_scale(line)


def ground_vanishing_line(v1, v2):
    """Join distinct ground directions; keep a valid horizon at infinity.

    For a finite horizon, (a,b) has unit length, so l dot (x,y,1) is a
    signed pixel distance. An infinite horizon has no such distance scale.
    """
    return line_through(v1, v2)


def focal_length_check(v1, v2, image_shape):
    """Return (f, raw_dot) under square pixels and principal point (w/2,h/2).

    Infinite directions give (None,None); a nonnegative finite raw dot gives
    (None,raw_dot). This optional diagnostic never enters height estimation.
    """
    try:
        shape = np.asarray(image_shape, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_shape must contain positive height and width") from exc
    if shape.ndim != 1 or len(shape) < 2 or not np.isfinite(shape[:2]).all() or np.any(shape[:2] <= 0):
        raise ValueError("image_shape must contain positive height and width")
    first, second = _point(v1, "first vanishing point"), _point(v2, "second vanishing point")
    if first[2] == 0 or second[2] == 0:
        return None, None
    center = np.array([shape[1] / 2, shape[0] / 2])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = float(np.dot(first[:2] / first[2] - center, second[:2] / second[2] - center))
    if not math.isfinite(raw):
        return None, None
    return (math.sqrt(-raw) if raw < 0 else None), raw


def _height_factor(vertical, horizon, base, top, name):
    base, top = _pixel_point(base, f"{name} base"), _pixel_point(top, f"{name} top")
    span = top[:2] - base[:2]
    span_length = float(np.linalg.norm(span))
    if span_length <= _EPS * max(1.0, float(np.linalg.norm(base[:2]))):
        raise ValueError(f"{name} has coincident or numerically indistinguishable endpoints")
    ground_distance = float(np.dot(horizon, base))
    if abs(ground_distance) <= _EPS * np.linalg.norm(base):
        raise ValueError(f"{name} base is on or numerically too close to the ground vanishing line")

    direction = vertical[:2] - vertical[2] * base[:2]
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= _EPS:
        raise ValueError(f"{name} base coincides with the vertical vanishing point")
    sine = abs(span[0] * direction[1] - span[1] * direction[0]) / (span_length * direction_length)
    if sine > math.sin(_VERTICAL_ANGLE):
        raise ValueError(f"{name} endpoints violate the vertical direction (over 2 degrees)")

    base_cross_top = np.cross(base, top)
    vertical_cross_top = np.cross(vertical, top)
    denominator = float(np.dot(vertical_cross_top, vertical_cross_top))
    if denominator <= _EPS**2 * float(np.dot(top, top)):
        raise ValueError(f"{name} top is at or too close to the vertical vanishing point")
    lam = -float(np.dot(base_cross_top, vertical_cross_top)) / denominator
    # b + lambda*v is a positive multiple of t for two visible endpoints.
    if 1.0 + lam * vertical[2] <= _EPS:
        raise ValueError(f"{name} endpoints cross the camera projection plane")
    factor = lam / ground_distance
    if not math.isfinite(factor) or factor == 0:
        raise ValueError(f"{name} has a degenerate height factor")
    return factor, ground_distance


def estimate_height(vz, vanishing_line, reference, target):
    """Recover height from reference=(base,top,height_m), target=(base,top).

    For normalized finite b,t, solve t ~ b + lambda*v. The factor
    lambda/(ground_line dot b) is proportional to physical height, so the
    target/reference factor ratio cancels the unknown camera and scale.
    This retains vertical perspective and works when v.w is zero.

    Endpoints must represent upright segments above the same support plane.
    A coarse 2-degree collinearity guard rejects inconsistent annotations;
    it is not an uncertainty bound or a claim of measurement accuracy.
    """
    try:
        base_r, top_r, known_height = reference
        base_t, top_t = target
        known_height = float(known_height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reference must be (base,top,height_m), target must be (base,top)") from exc
    if not math.isfinite(known_height) or known_height <= 0:
        raise ValueError("reference height must be finite and positive")
    vertical = _point(vz, "vertical vanishing point")
    horizon = _vector3(vanishing_line, "ground vanishing line")
    if abs(float(np.dot(vertical, horizon))) <= _EPS:
        raise ValueError("vertical direction lies in the ground plane")
    reference_factor, reference_side = _height_factor(vertical, horizon, base_r, top_r, "reference")
    target_factor, target_side = _height_factor(vertical, horizon, base_t, top_t, "target")
    if np.sign(reference_side) != np.sign(target_side):
        raise ValueError("reference and target bases are on opposite sides of the ground horizon")
    height = known_height * (target_factor / reference_factor)
    if not math.isfinite(height) or height <= 0:
        raise ValueError("height is nonpositive or nonfinite; check endpoint orientations")
    return float(height)


def _prepare_lines(lines):
    try:
        values = np.asarray(lines, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("lines must be homogeneous Nx3 lines or endpoint segments") from exc
    if values.ndim == 3 and values.shape[1:] == (2, 2):
        values = values.reshape(-1, 4)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] not in (3, 4) or not np.isfinite(values).all():
        raise ValueError("at least two finite homogeneous Nx3 lines or Nx4 pixel segments are required")
    if values.shape[1] == 4:
        anchors = (values[:, :2] + values[:, 2:]) / 2
        homogeneous = np.array([line_through(row[:2], row[2:]) for row in values])
    else:
        homogeneous = np.array([_vector3(row, "input line") for row in values])
        lengths = np.linalg.norm(homogeneous[:, :2], axis=1)
        if np.any(lengths == 0):
            raise ValueError("observed image lines cannot be the line at infinity")
        homogeneous = homogeneous / lengths[:, None]
        anchors = -homogeneous[:, 2, None] * homogeneous[:, :2]
    homogeneous = homogeneous / np.linalg.norm(homogeneous[:, :2], axis=1)[:, None]
    return homogeneous, anchors


def _fit_vanishing_point(lines, anchors):
    center = np.mean(anchors, axis=0)
    spread = float(np.mean(np.linalg.norm(anchors - center, axis=1)))
    scale = math.sqrt(2) / spread if spread > _EPS else 1.0
    # Transform image coordinates before solving the line-incidence equations.
    normalized = lines.copy()
    normalized[:, :2] /= scale
    normalized[:, 2] += lines[:, :2] @ center
    normalized /= np.linalg.norm(normalized[:, :2], axis=1)[:, None]
    _, singular, right = np.linalg.svd(normalized, full_matrices=True)
    if singular[1] <= _EPS * singular[0]:
        raise ValueError("at least two distinct projected lines are required")
    point = right[-1]
    point = np.r_[point[:2] / scale + center * point[2], point[2]]
    return _vector3(point, "vanishing point")


def _angular_residuals(lines, anchors, point):
    directions = point[:2] - point[2] * anchors
    lengths = np.linalg.norm(directions, axis=1)
    numerators = np.abs(lines @ point)
    sine = np.divide(numerators, lengths, out=np.zeros_like(lengths), where=lengths > _EPS)
    sine[(lengths <= _EPS) & (numerators > _EPS)] = 1.0
    return np.arcsin(np.clip(sine, 0, 1))


def estimate_vanishing_point(lines, ransac_threshold=None):
    """Fit a homogeneous direction and return (three-vector, boolean inlier mask).

    Input may be Nx3 homogeneous lines, Nx4 pixel endpoint segments, or
    Nx2x2 endpoints. None fits all lines with normalized least squares.
    RANSAC thresholds are ANGLES IN RADIANS, not pixel distances: residuals
    compare each line direction to its ray towards the candidate point.
    Segments use their midpoint; Nx3 lines use the point nearest the image
    origin because their original finite extent is unknown. Prefer segments
    when a physically meaningful local angular tolerance is needed.

    RANSAC uses deterministic hypotheses (fixed seed, at most 256 pairs).
    Two-line consensus is returned, not promoted to a confidence guarantee.
    """
    coefficients, anchors = _prepare_lines(lines)
    if ransac_threshold is None:
        return _fit_vanishing_point(coefficients, anchors), np.ones(len(coefficients), dtype=bool)
    try:
        threshold = float(ransac_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RANSAC threshold must be an angle in (0, pi/2) radians") from exc
    if not math.isfinite(threshold) or not 0 < threshold < math.pi / 2:
        raise ValueError("RANSAC threshold must be an angle in (0, pi/2) radians")
    count = len(coefficients)
    if count * (count - 1) // 2 <= 256:
        pairs = list(combinations(range(count), 2))
    else:
        rng = np.random.default_rng(0)
        pair_set = set()
        while len(pair_set) < 256:
            pair_set.add(tuple(sorted(rng.choice(count, 2, replace=False).tolist())))
        pairs = sorted(pair_set)
    best_mask, best_score = None, (-1, -math.inf)
    for first, second in pairs:
        candidate = np.cross(coefficients[first], coefficients[second])
        if np.linalg.norm(candidate) <= _EPS:
            continue
        candidate = _vector3(candidate, "candidate vanishing point")
        residual = _angular_residuals(coefficients, anchors, candidate)
        mask = residual <= threshold
        score = (int(mask.sum()), -float(np.median(residual[mask])))
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None or best_mask.sum() < 2:
        raise ValueError("no nondegenerate vanishing-point consensus")
    for _ in range(3):
        point = _fit_vanishing_point(coefficients[best_mask], anchors[best_mask])
        refined = _angular_residuals(coefficients, anchors, point) <= threshold
        if refined.sum() < 2:
            raise ValueError("vanishing-point consensus became degenerate after refinement")
        if np.array_equal(refined, best_mask):
            return point, refined
        best_mask = refined
    point = _fit_vanishing_point(coefficients[best_mask], anchors[best_mask])
    return point, _angular_residuals(coefficients, anchors, point) <= threshold
