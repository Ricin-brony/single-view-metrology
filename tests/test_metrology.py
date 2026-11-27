from contextlib import redirect_stderr
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from src.metrology import main, measure_scene


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        root = Path(work.name)
        self.scene = root / "scene"
        self.scene.mkdir()
        self.output = root / "output"
        self.debug = root / "replay"

        # Level pinhole camera, principal point (320, 240), focal length 400 px.
        # At ground depth 4 m, every meter of height spans 100 pixels.
        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.scene / "image.jpg"), image))
        (self.scene / "reference.txt").write_text("reference 0.2\n", encoding="utf-8")
        (self.scene / "objects.txt").write_text(
            "reference 270 340 270 320\ntarget 370 340 370 310\n", encoding="utf-8"
        )
        (self.scene / "lines.txt").write_text(
            "220 340 420 340 h1\n270 290 370 290 h1\n"
            "220 340 270 290 h2\n420 340 370 290 h2\n"
            "270 340 270 320 v\n370 340 370 310 v\n",
            encoding="utf-8",
        )

    def test_manual_outputs_and_replay_without_reading_target_truth(self):
        (self.scene / "truth.txt").write_text("not valid evaluation data\n", encoding="utf-8")
        record = measure_scene(self.scene, self.output, self.debug, mode="manual")
        self.assertEqual(record["status"], "ok")
        with (self.output / "heights.csv").open(newline="", encoding="utf-8") as stream:
            row, = csv.DictReader(stream)
        self.assertEqual(row["object_id"], "target")
        self.assertAlmostEqual(float(row["height_m"]), 0.3)
        self.assertEqual(json.loads((self.output / "run.json").read_text(encoding="utf-8")), record)
        self.assertTrue((self.output / "overlay.png").is_file())
        replay = measure_scene(self.debug, self.output)
        self.assertAlmostEqual(replay["measurements"][0]["height_m"], 0.3)

    def test_invalid_geometry_writes_rejected_row(self):
        with (self.scene / "lines.txt").open("a", encoding="utf-8") as stream:
            stream.write("1 1 1 1 h1\n")
        record = measure_scene(self.scene, self.output)
        self.assertEqual(record["status"], "rejected")
        failure, = record["measurements"]
        self.assertEqual(failure["object_id"], "target")
        self.assertIsNone(failure["height_m"])
        self.assertTrue(failure["reason"])

    def test_auto_reports_missing_profile_and_nonzero_exit(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--mode", "auto", "--input", str(self.scene), "--output", str(self.output)])
        self.assertEqual(code, 1)
        self.assertIn("rig.json", stderr.getvalue())
        record = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "rejected")
        self.assertIn("rig.json", record["measurements"][0]["reason"])

    def test_auto_reports_nonfinite_profile_data(self):
        (self.scene / "rig.json").write_text('{"extra": 1e999}\n', encoding="utf-8")
        record = measure_scene(self.scene, self.output, mode="auto")
        self.assertEqual(record["status"], "rejected")
        self.assertIn("Invalid JSON number", record["measurements"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
