import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTest(unittest.TestCase):
    def test_required_assets_exist(self):
        required = [
            "data/labels/manual_keyframe_labels_file1.json",
            "data/labels/manual_keyframe_labels_file2.json",
            "data/labels/manual_keyframe_labels_file3.json",
            "results/frozen_detector/best_model.pth",
            "results/frozen_detector/summary.json",
            "results/r5/best_model.pth",
            "results/r5/summary.json",
            "results/r5/cam_scorecam_taskguided_R5_auto.png",
        ]
        for item in required:
            self.assertTrue((ROOT / item).exists(), item)

    def test_full_csv_dataset_is_present(self):
        csv_files = list((ROOT / "data" / "raw").rglob("*.CSV"))
        self.assertEqual(len(csv_files), 126)

    def test_release_summaries_use_relative_paths(self):
        detector = json.loads((ROOT / "results" / "frozen_detector" / "summary.json").read_text(encoding="utf-8"))
        residual = json.loads((ROOT / "results" / "r5" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(detector["config"]["data_root"], "data/raw")
        self.assertEqual(detector["config"]["file3_labels"], "data/labels/manual_keyframe_labels_file3.json")
        self.assertEqual(residual["config"]["run_dir"], "results/frozen_detector")
        self.assertEqual(residual["config"]["size_reg_mode"], "expected_residual")


if __name__ == "__main__":
    unittest.main()

