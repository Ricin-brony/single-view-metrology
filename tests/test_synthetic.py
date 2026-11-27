"""Integration checks for the independently projected synthetic scene bundles."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


SCENES = ("level_equal_depth", "pitched_staggered", "missing_marker")
REPOSITORY = Path(__file__).resolve().parents[1]


class SyntheticSceneTests(unittest.TestCase):
    def generate(self, root):
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "tools.generate_synthetic", "--output", str(root)],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return {name: root / name for name in SCENES}

    @staticmethod
    def detect(path):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(cv2.imread(str(path)))
        if ids is None:
            return {}
        return {int(marker_id): points.reshape(4, 2)
                for marker_id, points in zip(ids.ravel(), corners)}

    def test_cli_writes_complete_bundles_with_the_required_detectable_ids(self):
        # Catches tags omitted, overlapped, or rendered as lookalike decorations.
        with tempfile.TemporaryDirectory() as temporary:
            scenes = self.generate(Path(temporary) / "scenes")
            self.assertEqual({p.name for p in scenes[SCENES[0]].parent.iterdir()}, set(SCENES))
            for name, scene in scenes.items():
                with self.subTest(scene=name):
                    self.assertTrue({"image.jpg", "reference.txt", "objects.txt", "lines.txt",
                                     "truth.txt", "rig.json", "scene.json"}.issubset(
                                         {p.name for p in scene.iterdir()}))
                    expected_ids = set(range(7 if name == "missing_marker" else 8))
                    self.assertEqual(set(self.detect(scene / "image.jpg")), expected_ids)

    def test_detected_canonical_corners_match_the_manual_physical_endpoints(self):
        # Catches using marker centers or a wrong corner for the measured segment.
        with tempfile.TemporaryDirectory() as temporary:
            scenes = self.generate(Path(temporary))
            for name in SCENES[:2]:
                scene = scenes[name]
                rig = json.loads((scene / "rig.json").read_text(encoding="utf-8"))
                self.assertEqual(rig["dictionary"], "DICT_4X4_50")
                self.assertEqual(rig["plane_tag_ids"], [0, 1, 2, 3])
                self.assertEqual(rig["objects"], [
                    {"object_id": "reference", "base": {"id": 4, "corner": 0},
                     "top": {"id": 5, "corner": 0}},
                    {"object_id": "target", "base": {"id": 6, "corner": 0},
                     "top": {"id": 7, "corner": 0}},
                ])
                self.assertEqual((scene / "reference.txt").read_text().split(),
                                 ["reference", "0.20"])
                self.assertEqual((scene / "truth.txt").read_text().split(), ["target", "0.30"])
                objects = {row[0]: np.array(row[1:], dtype=float).reshape(2, 2)
                           for row in (line.split() for line in
                                       (scene / "objects.txt").read_text().splitlines())}
                detected = self.detect(scene / "image.jpg")
                for profile in rig["objects"]:
                    for endpoint_index, endpoint in enumerate(("base", "top")):
                        role = profile[endpoint]
                        np.testing.assert_allclose(
                            detected[role["id"]][role["corner"]],
                            objects[profile["object_id"]][endpoint_index], atol=1.8,
                        )

    def test_annotations_follow_an_independent_pinhole_projection_and_known_heights(self):
        # Catches inconsistent manual annotations, truth, or the nominal camera cases.
        with tempfile.TemporaryDirectory() as temporary:
            scenes = self.generate(Path(temporary))
            for name, scene in scenes.items():
                metadata = json.loads((scene / "scene.json").read_text(encoding="utf-8"))
                camera = metadata["camera"]
                intrinsic = np.asarray(camera["intrinsic_matrix"])
                rotation = np.asarray(camera["world_to_camera_rotation"])
                position = np.asarray(camera["position_world_m"])

                def project(point):
                    image_point = intrinsic @ (rotation @ (np.asarray(point) - position))
                    return image_point[:2] / image_point[2]

                world_objects = metadata["world"]["objects"]
                image_objects = {row[0]: np.array(row[1:], dtype=float).reshape(2, 2)
                                 for row in (line.split() for line in
                                             (scene / "objects.txt").read_text().splitlines())}
                for object_id, height in (("reference", 0.20), ("target", 0.30)):
                    base, top = map(np.asarray, world_objects[object_id])
                    np.testing.assert_allclose(top - base, [0, 0, height], atol=1e-12)
                    np.testing.assert_allclose(image_objects[object_id],
                                               [project(base), project(top)], atol=1e-6)

                rectangle = metadata["world"]["ground_rectangle_m"]
                expected_edges = [(0, 1, "h1"), (3, 2, "h1"),
                                  (0, 3, "h2"), (1, 2, "h2")]
                line_rows = [line.split() for line in (scene / "lines.txt").read_text().splitlines()]
                for row, (first, second, label) in zip(line_rows[:4], expected_edges):
                    self.assertEqual(row[4], label)
                    np.testing.assert_allclose(np.array(row[:4], dtype=float).reshape(2, 2),
                                               [project(rectangle[first]), project(rectangle[second])],
                                               atol=1e-6)
                self.assertEqual([row[4] for row in line_rows], ["h1", "h1", "h2", "h2", "v", "v"])
                lengths = {key: np.linalg.norm(points[1] - points[0])
                           for key, points in image_objects.items()}
                naive_height = 0.20 * lengths["target"] / lengths["reference"]
                if name == "level_equal_depth":
                    self.assertAlmostEqual(naive_height, 0.30, places=7)
                else:
                    self.assertGreater(abs(naive_height - 0.30), 0.02)

    def test_repeated_generation_is_byte_for_byte_deterministic(self):
        # Catches accidental random data, timestamps, or nondeterministic scene output.
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            self.generate(first)
            self.generate(second)
            digest = lambda root: {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in root.rglob("*") if p.is_file()}
            self.assertEqual(digest(first), digest(second))


if __name__ == "__main__":
    unittest.main()
