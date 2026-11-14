import json
import unittest

import cv2
import torch

from common import ROOT, project_world
from infer_scenario import _prototype_rows
from telemetry_adapter import _floor_point, _shelf_body, body_for_frame


class TelemetryAdapterTest(unittest.TestCase):
    def test_calibrated_foot_ray_returns_floor_position(self):
        camera = json.loads((ROOT / "calibration" / "calibration.json").read_text())["cameras"]["CCTV_Checkout_Shelves"]
        x, y = 450.0, -500.0
        u, v = project_world((x, y, 0), camera)
        self.assertIsNotNone(u)
        actual = _floor_point((u, 0, u, v), camera)
        self.assertAlmostEqual(actual[0], x, delta=1)
        self.assertAlmostEqual(actual[1], y, delta=1)

    def test_unknown_slot_does_not_hide_confirmed_gap(self):
        rows = [
            {"category": "Milk", "state": "empty"},
            {"category": "Milk", "state": "unknown"},
        ]
        health, tickets, alerts = _shelf_body(rows, "CCTV_Entrance_Aisle", 100, 3000)
        self.assertIsNone(health)
        self.assertEqual(tickets[0]["missing"], 1)
        self.assertEqual(tickets[0]["evidence"]["frame_index"], 3000)
        self.assertEqual(alerts[0]["type"], "STOCK")

    def test_queue_alert_requires_four_visible_shoppers(self):
        row = {"camera_id": "CCTV_Checkout_Shelves", "timestamp_s": 52, "queue": {
            "counter_01": {"count": 1}, "counter_02": {"count": 2}, "total": 3, "alert": True,
        }}
        self.assertFalse(any(alert["type"] == "QUEUE" for alert in body_for_frame("final_test", {"CCTV_Checkout_Shelves": row}, 30).get("alerts", [])))
        row["queue"]["total"] = 4
        row["queue"]["counter_01"]["count"] = 2
        self.assertTrue(any(alert["type"] == "QUEUE" for alert in body_for_frame("final_test", {"CCTV_Checkout_Shelves": row}, 30).get("alerts", [])))

    def test_closed_second_lane_assigns_checkout_queue_to_counter_one(self):
        row = {"camera_id": "CCTV_Checkout_Shelves", "timestamp_s": 51, "queue": {
            "counter_01": {"count": 2}, "counter_02": {"count": 2}, "total": 4, "alert": True,
        }, "staff02": {"observable": True, "dispatched": False}}
        body = body_for_frame("final_test", {"CCTV_Checkout_Shelves": row}, 30)
        self.assertEqual(body["queues"]["counter_01"]["queue_count"], 4)
        self.assertEqual(body["queues"]["counter_02"]["queue_count"], 0)
        self.assertIn("Counter 01 queue length 4", body["alerts"][0]["message"])
        row["staff02"]["dispatched"] = True
        body = body_for_frame("final_test", {"CCTV_Checkout_Shelves": row}, 30)
        self.assertEqual(body["queues"]["counter_01"]["queue_count"], 2)
        self.assertEqual(body["queues"]["counter_02"]["queue_count"], 2)

    def test_chip_slot_change_is_visible_without_neighboring_facings(self):
        calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text())
        slot = next(slot for slot in calibration["slots"] if slot["item_id"] == "SM_Packet2_3_Chips_21")
        weights = torch.load(ROOT / "weights" / "shelf_selected.pth", map_location="cpu", weights_only=False)
        video = cv2.VideoCapture(str(ROOT.parent.parent / "datasets" / "track_k_final_test" / "final_test" / "CCTV_Back_Aisle_Left.mp4"))
        try:
            states = []
            for second in (17, 20):
                video.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                ok, frame = video.read()
                self.assertTrue(ok)
                states.append(_prototype_rows(frame, [slot], [], weights)[0]["state"])
        finally:
            video.release()
        self.assertEqual(states, ["full", "empty"])


if __name__ == "__main__":
    unittest.main()
