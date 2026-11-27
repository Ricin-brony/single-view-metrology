"""Independent, rendered-marker tests for the automatic scene reader."""

import copy
import json
import unittest

import cv2
import numpy as np

from src.markers import MarkerError, detect_scene


POSITIONS = {
    0: (40, 600), 1: (980, 600), 2: (980, 850), 3: (40, 850),
    4: (300, 650), 5: (300, 200), 6: (750, 700), 7: (800, 350),
}
DEFAULT_PROFILE = object()


def profile():
    return {
        "dictionary": "DICT_4X4_50",
        "plane_tag_ids": [0, 1, 2, 3],
        "objects": [
            {"object_id": "reference", "base": {"id": 4, "corner": 0},
             "top": {"id": 5, "corner": 0}},
            {"object_id": "target", "base": {"id": 6, "corner": 0},
             "top": {"id": 7, "corner": 0}},
        ],
    }


def scene(omit=(), extra=(), dictionary_name="DICT_4X4_50", positions=None):
    image = np.full((1050, 1200), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    for marker_id, (x, y) in list((POSITIONS if positions is None else positions).items()) + list(extra):
        if marker_id in omit:
            continue
        image[y:y + 80, x:x + 80] = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
    return image


class MarkerSceneTests(unittest.TestCase):
    def detect(self, image, config=DEFAULT_PROFILE):
        self.assertIsNotNone(detect_scene, "automatic marker scene reader is not implemented")
        return detect_scene(image, profile() if config is DEFAULT_PROFILE else config)

    def test_rendered_ids_produce_role_endpoints_and_plane_line_groups(self):
        objects, groups, diagnostics = self.detect(scene())
        np.testing.assert_allclose(objects["reference"], [[300, 650], [300, 200]], atol=1)
        np.testing.assert_allclose(objects["target"], [[750, 700], [800, 350]], atol=1)
        np.testing.assert_allclose(groups["h1"], [
            [[79.5, 639.5], [1019.5, 639.5]],
            [[79.5, 889.5], [1019.5, 889.5]],
        ], atol=1)
        np.testing.assert_allclose(groups["h2"], [
            [[79.5, 639.5], [79.5, 889.5]],
            [[1019.5, 639.5], [1019.5, 889.5]],
        ], atol=1)
        np.testing.assert_allclose(groups["v"], [
            [[300, 650], [300, 200]], [[750, 700], [800, 350]],
        ], atol=1)
        json.dumps(diagnostics, allow_nan=False)
        self.assertEqual(set(diagnostics["detected_ids"]), set(range(8)))
        self.assertEqual(len(diagnostics["markers"]), 8)

    def test_missing_required_marker_is_an_explicit_failure(self):
        self.assertIsNotNone(detect_scene, "automatic marker scene reader is not implemented")
        with self.assertRaises(MarkerError) as caught:
            detect_scene(scene(omit=(7,)), profile())
        self.assertEqual(caught.exception.diagnostics["reason"], "missing_markers")
        self.assertEqual(caught.exception.diagnostics["missing_ids"], [7])

    def test_duplicate_required_detection_is_an_explicit_failure(self):
        self.assertIsNotNone(detect_scene, "automatic marker scene reader is not implemented")
        with self.assertRaises(MarkerError) as caught:
            detect_scene(scene(extra=[(7, (550, 40))]), profile())
        self.assertEqual(caught.exception.diagnostics["reason"], "duplicate_markers")
        self.assertEqual(caught.exception.diagnostics["duplicate_ids"], [7])

    def test_invalid_profile_shapes_fail_with_marker_error(self):
        malformed = [None, [], {}, {**profile(), "objects": []},
                     {**profile(), "objects": {}}, {**profile(), "objects": [None]},
                     {**profile(), "plane_tag_ids": [0, 1, 2]}]
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(MarkerError) as caught:
                    self.detect(scene(), value)
                self.assertEqual(caught.exception.diagnostics["reason"], "invalid_profile")

    def test_marker_ids_are_unique_plain_integers_within_the_dictionary(self):
        malformed = []
        for value in (True, 3.0, "3", -1, 50):
            config = profile()
            config["plane_tag_ids"][3] = value
            malformed.append(config)
        for value in (True, 4.0, -1, 50):
            config = profile()
            config["objects"][0]["base"]["id"] = value
            malformed.append(config)
        config = profile()
        config["plane_tag_ids"] = [0, 1, 2, 2]
        malformed.append(config)
        config = profile()
        config["objects"][1]["top"]["id"] = 4
        malformed.append(config)
        for config in malformed:
            with self.subTest(config=config):
                with self.assertRaises(MarkerError) as caught:
                    self.detect(scene(), config)
                self.assertEqual(caught.exception.diagnostics["reason"], "invalid_profile")

    def test_endpoint_corners_and_object_names_are_validated(self):
        malformed = []
        for value in (True, 0.0, -1, 4, "0", None):
            config = profile()
            config["objects"][0]["base"]["corner"] = value
            malformed.append(config)
        for value in (None, {}, 1):
            config = profile()
            config["objects"][0]["top"] = value
            malformed.append(config)
        for value in (None, "", "  ", 123, "reference", "two words", "target#2"):
            config = profile()
            config["objects"][1]["object_id"] = value
            malformed.append(config)
        for config in malformed:
            with self.subTest(config=config):
                with self.assertRaises(MarkerError) as caught:
                    self.detect(scene(), config)
                self.assertEqual(caught.exception.diagnostics["reason"], "invalid_profile")

    def test_invalid_dictionary_is_a_profile_error(self):
        for value in (None, 0, "DICT_NOT_A_DICTIONARY", "__dict__"):
            with self.subTest(value=value):
                config = profile()
                config["dictionary"] = value
                with self.assertRaises(MarkerError) as caught:
                    self.detect(scene(), config)
                self.assertEqual(caught.exception.diagnostics["reason"], "invalid_profile")

    def test_invalid_images_return_marker_errors_instead_of_opencv_assertions(self):
        for image in (None, [], np.zeros((0, 8), np.uint8), np.zeros((10, 10), float),
                      np.zeros((10, 10, 2), np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(MarkerError) as caught:
                    self.detect(image)
                self.assertEqual(caught.exception.diagnostics["reason"], "invalid_image")

    def test_self_crossing_plane_order_is_rejected(self):
        config = profile()
        config["plane_tag_ids"] = [0, 2, 1, 3]
        with self.assertRaises(MarkerError) as caught:
            self.detect(scene(), config)
        self.assertEqual(caught.exception.diagnostics["reason"], "invalid_plane")

    def test_collinear_plane_markers_are_rejected(self):
        positions = dict(POSITIONS)
        positions.update({0: (40, 850), 1: (250, 850), 2: (500, 850), 3: (980, 850)})
        with self.assertRaises(MarkerError) as caught:
            self.detect(scene(positions=positions))
        self.assertEqual(caught.exception.diagnostics["reason"], "invalid_plane")

    def test_coincident_object_axes_are_rejected(self):
        positions = dict(POSITIONS)
        positions.update({6: (300, 780), 7: (300, 350)})
        with self.assertRaises(MarkerError) as caught:
            self.detect(scene(positions=positions))
        self.assertEqual(caught.exception.diagnostics["reason"], "coincident_vertical_axes")

    def test_perspective_marker_uses_diagonal_intersection_not_corner_average(self):
        image = scene(omit=(0,))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 80)
        source = np.float32([[0, 0], [79, 0], [79, 79], [0, 79]])
        destination = np.float32([[40, 580], [160, 600], [130, 700], [50, 680]])
        transform = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(marker, transform, (1200, 1050),
                                    flags=cv2.INTER_NEAREST, borderValue=255)
        _, _, diagnostics = self.detect(np.minimum(image, warped))
        center = next(item["center"] for item in diagnostics["markers"] if item["id"] == 0)
        # The two specified diagonals intersect at (92.0588, 649.4118).
        # Their four-corner average (95, 640) is not the projection of the center.
        np.testing.assert_allclose(center, [92.0588235294, 649.4117647059], atol=1.5)

    def test_rotation_preserves_canonical_marker_corner_roles(self):
        config = profile()
        config["objects"][0]["base"]["corner"] = 2
        config["objects"][0]["top"]["corner"] = 1
        config["objects"][1]["base"]["corner"] = 3
        config["objects"][1]["top"]["corner"] = 2
        objects, _, _ = self.detect(np.rot90(scene()), config)
        # A counterclockwise rotation maps (x,y) to (y,1199-x).
        np.testing.assert_allclose(objects["reference"], [[729, 820], [200, 820]], atol=1)
        np.testing.assert_allclose(objects["target"], [[779, 449], [429, 320]], atol=1)

    def test_color_images_use_the_same_geometry_without_modifying_inputs(self):
        config = profile()
        before = copy.deepcopy(config)
        gray = scene()
        for conversion in (cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGRA):
            image = cv2.cvtColor(gray, conversion)
            original = image.copy()
            objects, _, _ = self.detect(image, config)
            np.testing.assert_allclose(objects["target"], [[750, 700], [800, 350]], atol=1)
            np.testing.assert_array_equal(image, original)
        self.assertEqual(config, before)

    def test_another_named_predefined_dictionary_is_supported(self):
        config = profile()
        config["dictionary"] = "DICT_5X5_50"
        objects, _, diagnostics = self.detect(scene(dictionary_name="DICT_5X5_50"), config)
        np.testing.assert_allclose(objects["reference"], [[300, 650], [300, 200]], atol=1)
        self.assertEqual(diagnostics["dictionary"], "DICT_5X5_50")

    def test_partial_occlusion_cannot_silently_supply_an_endpoint(self):
        image = scene()
        image[350:430, 840:880] = 255
        with self.assertRaises(MarkerError) as caught:
            self.detect(image)
        self.assertEqual(caught.exception.diagnostics["missing_ids"], [7])
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_blank_image_reports_all_required_ids(self):
        with self.assertRaises(MarkerError) as caught:
            self.detect(np.full((200, 200), 255, np.uint8))
        self.assertEqual(caught.exception.diagnostics["missing_ids"], list(range(8)))
        self.assertEqual(caught.exception.diagnostics["markers"], [])

    def test_concave_plane_layout_is_rejected(self):
        positions = dict(POSITIONS)
        positions[2] = (500, 700)
        with self.assertRaises(MarkerError) as caught:
            self.detect(scene(positions=positions))
        self.assertEqual(caught.exception.diagnostics["reason"], "invalid_plane")

    def test_distinct_parallel_vertical_axes_remain_valid(self):
        positions = dict(POSITIONS)
        positions[7] = (750, 350)
        objects, groups, _ = self.detect(scene(positions=positions))
        np.testing.assert_allclose(objects["target"], [[750, 700], [750, 350]], atol=1)
        self.assertEqual(len(groups["v"]), 2)

    def test_irrelevant_profile_dimensions_do_not_supply_measurement_scale(self):
        config = profile()
        config["target_height_m"] = 12345
        config["marker_length_m"] = 999
        config["plane_dimensions"] = [99999, 99999]
        actual = self.detect(scene(), config)
        expected = self.detect(scene())
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
