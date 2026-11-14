import unittest

from vision_engine import CAMERAS, SCENARIOS, WEIGHTS_PATH, build_payload, counts_at, load_checkpoint, open_cameras


class ReplayTest(unittest.TestCase):
    def test_checkpoint_replay_only_reports_supported_measurements(self):
        _payload, rows = load_checkpoint(WEIGHTS_PATH)
        counts = counts_at(rows, "baseline", 0)
        result = build_payload("single_cashier", counts, {camera["id"]: True for camera in CAMERAS})
        self.assertEqual(result["headcount"], max(row["shopper"] for row in counts.values()))
        self.assertTrue(all(field not in result for field in ("traffic", "queues", "shelf_health", "shoppers")))
        self.assertNotIn("3", SCENARIOS)

    def test_bundled_baseline_video_opens(self):
        captures = open_cameras("single_cashier")
        try:
            self.assertEqual(len(captures), 5)
        finally:
            for _camera, capture, _path in captures:
                capture.release()


if __name__ == "__main__":
    unittest.main()
