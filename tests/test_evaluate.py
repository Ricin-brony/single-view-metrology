import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluate import evaluate_scenes
from src.metrology import measure_scene
from tools.generate_synthetic import generate_scenes


class SyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.root = Path(work.name)
        cls.scenes = generate_scenes(cls.root / "scenes")

    def test_clean_scenes_measure_and_missing_marker_is_counted(self):
        output = self.root / "evaluation"
        scene_names = ("level_equal_depth", "pitched_staggered", "missing_marker")
        report = evaluate_scenes([self.scenes[name] for name in scene_names], output)
        self.assertEqual(report["modes"]["auto"]["attempted"], 3)
        self.assertEqual(report["modes"]["auto"]["successful"], 2)
        self.assertEqual(report["modes"]["manual"]["successful"], 3)

        with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
            rows = {(row["scene"], row["mode"]): row for row in csv.DictReader(stream)}
        for scene_name in scene_names[:2]:
            self.assertAlmostEqual(float(rows[scene_name, "manual"]["height_m"]), 0.30, places=5)
            self.assertAlmostEqual(float(rows[scene_name, "auto"]["height_m"]), 0.30, delta=0.008)
        pixel_error = float(rows["pitched_staggered", "pixel"]["absolute_error_m"])
        auto_error = float(rows["pitched_staggered", "auto"]["absolute_error_m"])
        self.assertLess(auto_error, pixel_error)
        failure = rows["missing_marker", "auto"]
        self.assertEqual(failure["height_m"], "")
        self.assertIn("7", failure["reason"])

    def test_auto_uses_only_photo_reference_and_rig(self):
        scene = self.root / "minimal"
        scene.mkdir()
        original = self.scenes["pitched_staggered"]
        for filename in ("image.jpg", "reference.txt", "rig.json"):
            (scene / filename).write_bytes((original / filename).read_bytes())
        output = self.root / "auto"
        replay = self.root / "replay"
        record = measure_scene(scene, output, replay, mode="auto")
        self.assertEqual(record["status"], "ok")
        rerun = measure_scene(replay, self.root / "manual-replay", mode="manual")
        self.assertAlmostEqual(record["measurements"][0]["height_m"], rerun["measurements"][0]["height_m"])
        self.assertFalse((replay / "truth.txt").exists())


if __name__ == "__main__":
    unittest.main()
