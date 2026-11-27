"""Geometry contracts checked against independently projected 3D scenes."""

import math
import unittest

import numpy as np

try:
    from src import geometry
except ImportError:
    geometry = None


def camera_scene(pitch=0.0, roll=0.0, target_height=1.5, target_depth=7.5):
    """Forward-project physical points; do not use the metrology estimator."""
    pitch, roll = np.deg2rad([pitch, roll])
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rotation = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    )
    intrinsics = np.array([[900, 0, 640], [0, 900, 480], [0, 0, 1]])
    direction_projection = intrinsics @ rotation

    def project(x, height, depth):
        homogeneous = direction_projection @ np.array([x, 1.8 - height, depth])
        if homogeneous[2] <= 0:
            raise AssertionError("fixture placed a point behind the camera")
        return homogeneous[:2] / homogeneous[2]

    vertical = direction_projection @ np.array([0, -1, 0])
    horizon = np.linalg.inv(direction_projection).T @ np.array([0, 1, 0])
    reference = (project(-0.6, 0, 4), project(-0.6, 1.0, 4), 1.0)
    target = (project(1.1, 0, target_depth), project(1.1, target_height, target_depth))
    return vertical, horizon, reference, target


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(geometry, "the geometry module has not been implemented")

    def assertProjectivelyEqual(self, actual, expected, tolerance=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        residual = np.linalg.norm(np.cross(actual, expected))
        self.assertLessEqual(residual, tolerance * np.linalg.norm(actual) * np.linalg.norm(expected))

    def test_line_through_accepts_pixels_scaled_points_and_infinity(self):
        self.assertProjectivelyEqual(geometry.line_through([5, 2], [-10, -18, -2]), [1, 0, -5])
        self.assertProjectivelyEqual(geometry.line_through([5, 2], [0, 1, 0]), [1, 0, -5])

    def test_line_through_rejects_coincident_or_invalid_points(self):
        for first, second in [([1, 2], [2, 4, 2]), ([0, 0, 0], [1, 2]), ([1, np.nan], [2, 3])]:
            with self.subTest(first=first), self.assertRaises(ValueError):
                geometry.line_through(first, second)

    def test_ground_line_is_normalized_and_preserves_infinite_horizon(self):
        line = geometry.ground_vanishing_line([10, 150, 1], [600, 150, 1])
        self.assertProjectivelyEqual(line, [0, 1, -150])
        self.assertAlmostEqual(np.linalg.norm(line[:2]), 1)
        self.assertProjectivelyEqual(
            geometry.ground_vanishing_line([1, 0, 0], [0, 1, 0]), [0, 0, 1]
        )

    def test_ground_line_rejects_same_direction(self):
        with self.assertRaises(ValueError):
            geometry.ground_vanishing_line([1, 2, 3], [-2, -4, -6])

    def test_focal_diagnostic_uses_image_center_and_keeps_raw_dot(self):
        focal, raw = geometry.focal_length_check([-480, 240, 1], [1120, 240, 1], (480, 640, 3))
        self.assertAlmostEqual(focal, 800)
        self.assertAlmostEqual(raw, -640000)
        self.assertEqual(geometry.focal_length_check([0, 1, 0], [1, 2, 1], (480, 640)), (None, None))
        focal, raw = geometry.focal_length_check([420, 240], [520, 240], (480, 640))
        self.assertIsNone(focal)
        self.assertEqual(raw, 20000)

    def test_focal_diagnostic_rejects_invalid_image_shape(self):
        with self.assertRaises(ValueError):
            geometry.focal_length_check([1, 2], [3, 4], (0, 640))

    def test_height_matches_level_hand_calculated_example(self):
        value = geometry.estimate_height(
            [0, 1, 0], [0, 1, -150], ([160, 350], [160, 250], 1.0), ([350, 250], [350, 175])
        )
        self.assertAlmostEqual(value, 1.5, places=10)

    def test_height_matches_finite_vertical_hand_example(self):
        value = geometry.estimate_height(
            [0, 0, 1], [0, 1, -50], ([100, 200], [50, 100], 1.0), ([200, 150], [100, 75])
        )
        self.assertAlmostEqual(value, 1.5, places=10)

    def test_height_matches_independent_camera_with_pitch_roll_and_depth(self):
        for pitch, roll, height, depth in [(0, 0, 1.5, 7.5), (18, 23, 2.4, 8), (-15, -12, 0.6, 6), (0, 0, 2.4, 4)]:
            with self.subTest(pitch=pitch, roll=roll, height=height):
                self.assertAlmostEqual(
                    geometry.estimate_height(*camera_scene(pitch, roll, height, depth)), height, places=8
                )

    def test_height_accepts_a_valid_infinite_ground_line(self):
        self.assertAlmostEqual(geometry.estimate_height(*camera_scene(90, 0, 0.6, 6)), 0.6, places=8)

    def test_height_is_invariant_to_independent_homogeneous_scales(self):
        vertical, horizon, reference, target = camera_scene(18, 23)
        point = lambda p, scale: np.r_[p, 1.0] * scale
        scaled_reference = (point(reference[0], -3), point(reference[1], 0.2), 2.0)
        scaled_target = (point(target[0], 7), point(target[1], -0.8))
        self.assertAlmostEqual(
            geometry.estimate_height(vertical * -5, horizon * 11, scaled_reference, scaled_target), 3.0, places=8
        )

    def test_height_rejects_wrong_orientation_and_inconsistent_vertical(self):
        reference = ([160, 350], [160, 250], 1.0)
        for target in [([350, 250], [350, 300]), ([350, 250], [450, 175]), ([350, 250], [350, 250])]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                geometry.estimate_height([0, 1, 0], [0, 1, -150], reference, target)

    def test_height_rejects_horizon_base_and_direction_in_ground(self):
        reference = ([160, 350], [160, 250], 1.0)
        with self.assertRaises(ValueError):
            geometry.estimate_height([0, 1, 0], [0, 1, -150], reference, ([350, 150], [350, 125]))
        with self.assertRaises(ValueError):
            geometry.estimate_height([1, 0, 0], [0, 1, -150], reference, ([350, 250], [350, 175]))

    def test_height_rejects_invalid_reference_scale_and_infinite_endpoint(self):
        for height in [0, -1, float("nan"), float("inf")]:
            with self.subTest(height=height), self.assertRaises(ValueError):
                geometry.estimate_height([0, 1, 0], [0, 1, -150], ([160, 350], [160, 250], height), ([350, 250], [350, 175]))
        with self.assertRaises(ValueError):
            geometry.estimate_height([0, 1, 0], [0, 1, -150], ([160, 350], [160, 250], 1), ([350, 250], [0, 1, 0]))

    def test_vanishing_point_fits_two_finite_lines(self):
        point, mask = geometry.estimate_vanishing_point([[0, 1, -100], [1, 0, -200]])
        self.assertProjectivelyEqual(point, [200, 100, 1])
        np.testing.assert_array_equal(mask, [True, True])

    def test_vanishing_point_keeps_parallel_lines_at_infinity(self):
        point, mask = geometry.estimate_vanishing_point([[1, 0, -10], [2, 0, -100], [-3, 0, 270]])
        self.assertProjectivelyEqual(point, [0, 1, 0])
        self.assertTrue(mask.all())

    def test_vanishing_point_accepts_endpoint_segments(self):
        segments = np.array([[[0, 0], [40, 20]], [[0, 100], [40, 100]], [[200, 0], [200, 30]]])
        point, _ = geometry.estimate_vanishing_point(segments)
        self.assertProjectivelyEqual(point, [200, 100, 1])
        point_flat, _ = geometry.estimate_vanishing_point(segments.reshape(-1, 4))
        self.assertProjectivelyEqual(point_flat, [200, 100, 1])

    def test_vanishing_point_ransac_rejects_wrong_direction_at_infinity(self):
        point, mask = geometry.estimate_vanishing_point(
            [[1, 0, -10], [1, 0, -50], [1, 0, -90], [0, 1, -50]], ransac_threshold=0.01
        )
        self.assertProjectivelyEqual(point, [0, 1, 0])
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_vanishing_point_ransac_rejects_finite_outlier(self):
        segments = [[0, 0, 30, 20], [0, 100, 30, 110], [100, 0, 120, 20], [300, 0, 300, 40], [0, -100, 50, -100]]
        point, mask = geometry.estimate_vanishing_point(segments, ransac_threshold=0.01)
        self.assertProjectivelyEqual(point, [300, 200, 1])
        np.testing.assert_array_equal(mask, [True, True, True, True, False])

    def test_vanishing_point_rejects_insufficient_duplicate_or_invalid_lines(self):
        for lines in [[], [[1, 0, 0]], [[1, 0, -10], [-2, 0, 20]], [[0, 0, 1], [1, 0, -5]], [[1, 0, np.inf], [0, 1, 0]]]:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                geometry.estimate_vanishing_point(lines)
        for threshold in [0, -0.1, float("nan")]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                geometry.estimate_vanishing_point([[1, 0, -10], [1, 0, -50]], threshold)


if __name__ == "__main__":
    unittest.main()
