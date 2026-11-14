import asyncio
import json
import os
import shutil
import subprocess
import time
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

os.environ["TRACK_K_INGEST_TOKEN"] = "test-ingest-token"
os.environ["TRACK_K_OPERATOR_USER"] = "test-operator"
os.environ["TRACK_K_OPERATOR_PASSWORD"] = "test-operator-password"
os.environ["TRACK_K_USE_SIMULATED_DATA"] = "false"

from main import advance_timeline, app, clear_results, heatmap_seconds, lifespan, registry, review_video, save_latest_report, stream, tracking_clock, zone_dwell_seconds, zone_visitor_ids  # noqa: E402


class BackendTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original = deepcopy(registry)
        self.original_dwell = zone_dwell_seconds.copy()
        self.original_visitors = deepcopy(zone_visitor_ids)
        self.original_heatmap = heatmap_seconds.copy()
        self.original_clock = tracking_clock.copy()
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        self.ingest_headers = {"Authorization": "Bearer test-ingest-token"}
        self.operator_auth = ("test-operator", "test-operator-password")

    async def asyncTearDown(self):
        registry.clear()
        registry.update(self.original)
        zone_dwell_seconds.update(self.original_dwell)
        zone_visitor_ids.update(self.original_visitors)
        heatmap_seconds[:] = self.original_heatmap
        tracking_clock.update(self.original_clock)
        await self.client.aclose()

    async def test_scene_geometry_and_auth(self):
        self.assertEqual({zone["source_id"] for zone in registry["zones"] if zone["category"] not in {"Checkout", "Threshold"}}, {"ShelfBox", "ShelfBox3", "ShelfBox4", "ShelfBox5", "ShelfBox6", "ShelfBox7", "ShelfBox8"})
        self.assertEqual(next(zone for zone in registry["zones"] if zone["id"] == "wall_boxed")["layout"], "side-wall")
        self.assertEqual((await self.client.get("/")).status_code, 401)
        self.assertEqual((await self.client.post("/api/telemetry", json={"queue_count": 4})).status_code, 401)
        self.assertEqual((await self.client.get("/media/orbit/frame-00.webp")).status_code, 401)
        denied_model = await self.client.get("/api/electronics/scene.glb")
        self.assertEqual(denied_model.status_code, 401)
        model = await self.client.get("/api/electronics/scene.glb", auth=self.operator_auth)
        self.assertEqual(model.status_code, 200, model.text)
        self.assertTrue(model.headers["content-type"].startswith("model/gltf-binary"))
        self.assertEqual(model.content[:4], b"glTF")

    async def test_ingest_dispatch_and_tasks(self):
        payload = {
            "footfall": {"in": 7, "out": 2},
            "live_occupancy": 5,
            "queue_count": 4,
            "camera_feeds": {"cam_checkout": True},
            "fps": 29.7,
            "task_queue": [
                {"id": "MILK-1", "item": "1L Milk Carton", "category": "Dairy", "location": "Aisle_1_Tier_2", "missing": 5},
                {"id": "CANS-1", "item": "Canned food", "category": "CannedFoods", "location": "Aisle_3_Tier_1", "missing": 3},
            ],
        }
        response = await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["mode"], "vision")
        self.assertEqual(state["measurement_source"], "external_vision")
        self.assertEqual(state["measurements"], {"visible_shoppers": True, "footfall": True, "queues": True, "positions": False, "shelf_gaps": True})
        self.assertEqual(state["live_occupancy"], 5)
        self.assertEqual(state["current_scenario"], "final_test")
        self.assertEqual(state["queues"]["counter_01"]["queue_count"], 4)
        self.assertEqual(state["checkout"]["counter_02"]["status"], "standby")
        self.assertIn("[QUEUE_ALERT] Counter 01 queue length 4 breaks threshold.", str(state["events"]))
        self.assertEqual(len(state["footfall_history"]), 1)
        self.assertEqual(next(item for item in state["task_queue"] if item["id"] == "MILK-1")["status"], "Open")
        self.assertEqual(next(item for item in state["task_queue"] if item["id"] == "CANS-1")["shelf"], "shelf4")

        async def patch(status):
            return await self.client.patch("/api/tasks/MILK-1", json={"status": status}, auth=self.operator_auth)
        self.assertEqual((await patch("Refilled")).status_code, 409)
        self.assertEqual((await patch("Assigned")).json()["status"], "Assigned")
        self.assertEqual((await patch("Refilled")).json()["status"], "Refilled")
        self.assertEqual((await patch("Assigned")).status_code, 409)

        repeated = (await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)).json()
        self.assertEqual(next(item for item in repeated["task_queue"] if item["id"] == "MILK-1")["status"], "Refilled")
        self.assertEqual(len([item for item in repeated["events"] if "breaks threshold" in item["message"]]), 1)
        self.assertEqual(len(repeated["footfall_history"]), 1)

    async def test_position_stream_builds_zone_visits_and_heatmap(self):
        shopper = {"id": "person_01", "x": 50, "y": 50, "zone": "shelf4"}
        first = await self.client.post("/api/telemetry", json={"shoppers": [shopper]}, headers=self.ingest_headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["occupancy"]["zones"]["shelf4"]["visits"], 1)
        tracking_clock["last"] = time.monotonic() - 2
        second = await self.client.post("/api/telemetry", json={"shoppers": [shopper]}, headers=self.ingest_headers)
        heatmap = second.json()["occupancy"]["heatmap"]
        self.assertEqual(heatmap["size"], 12)
        self.assertGreater(sum(heatmap["seconds"]), 1)
        self.assertEqual(second.json()["occupancy"]["zones"]["shelf4"]["visits"], 1)
        moved = await self.client.post("/api/telemetry", json={"shoppers": [{**shopper, "zone": "shelf2"}]}, headers=self.ingest_headers)
        self.assertEqual(moved.json()["occupancy"]["zones"]["shelf2"]["visits"], 1)

    async def test_partial_camera_visibility_preserves_store_occupancy(self):
        payload = {"headcount": 2, "live_occupancy": 2, "shoppers": [{"id": "visible_01", "x": 450, "y": -500}], "coordinate_system": "world", "playback_seconds": 1}
        response = await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["occupancy"]["count"], 2)
        self.assertEqual(len(response.json()["shoppers"]), 1)
        payload["headcount"] = payload["live_occupancy"] = 0
        self.assertEqual((await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)).status_code, 422)

    async def test_total_queue_does_not_overwrite_lane_count(self):
        response = await self.client.post("/api/telemetry", json={
            "queue_count": 4,
            "queues": {"counter_01": {"queue_count": 3}, "counter_02": {"queue_count": 1}},
            "checkout": {"counter_01": {"queue_count": 3}, "counter_02": {"queue_count": 1}},
        }, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["checkout"]["counter_01"]["queue_count"], 3)
        self.assertEqual(state["checkout"]["counter_02"]["queue_count"], 1)
        self.assertFalse(any("Counter 01 queue length 4" in event["message"] for event in state["events"]))

    async def test_queue_episode_ignores_duplicate_model_alert_and_brief_count_dip(self):
        async def post(count, second):
            response = await self.client.post("/api/telemetry", json={
                "queue_count": count,
                "playback_seconds": second,
                "alerts": [{"type": "QUEUE", "message": "[QUEUE_ALERT] Counter 01 queue length 4 breaks threshold."}],
            }, headers=self.ingest_headers)
            self.assertEqual(response.status_code, 200, response.text)
            return [event["message"] for event in response.json()["events"] if event["type"] == "QUEUE"]

        self.assertEqual(len(await post(4, 43)), 1)
        self.assertEqual(len(await post(2, 44)), 1)
        self.assertEqual(len(await post(4, 50)), 1)
        self.assertEqual(len(await post(2, 52)), 1)
        self.assertEqual(len(await post(2, 60)), 2)
        self.assertEqual(len(await post(4, 61)), 3)
        self.assertEqual([(alert["video_seconds"], alert["count"]) for alert in registry["queue_alerts"]], [(43, 4), (61, 4)])

    async def test_continuous_vision_alert_is_logged_once(self):
        payload = {"playback_seconds": 20, "alerts": [{"type": "STOCK", "message": "Dairy facing has 2 empty slots."}]}
        for second in (20, 27, 30):
            payload["playback_seconds"] = second
            response = await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len([event for event in response.json()["events"] if event["type"] == "STOCK"]), 1)

    async def test_completed_run_and_task_status_survive_restart(self):
        with TemporaryDirectory() as directory, patch("main.LATEST_REPORT_FILE", Path(directory) / "latest.json"), patch("main.RUN_FILE", Path(directory) / "runs.json"):
            registry["run_id"] = "abc123def456"
            registry["playback"] = {"status": "complete", "scenario": "final_test", "pid": None}
            registry["task_queue"] = [{"id": "MILK-1", "item": "Milk", "category": "Dairy", "location": "Aisle_1_Tier_2", "missing": 2, "status": "Open", "shelf": "shelf1"}]
            registry["queue_alerts"] = [{"time": "2026-11-11T12:00:00+00:00", "video_seconds": 43, "count": 4}]
            (Path(directory) / "runs.json").write_text(json.dumps([{"id": "abc123def456", "scenario": "final_test", "status": "complete"}]))
            save_latest_report()
            clear_results()
            registry["playback"] = {"status": "idle", "scenario": None, "pid": None}
            async with lifespan(app):
                self.assertEqual(registry["task_queue"][0]["item"], "Milk")
                self.assertEqual(registry["queue_alerts"][0]["video_seconds"], 43)
                response = await self.client.patch("/api/tasks/MILK-1", json={"status": "Assigned"}, auth=self.operator_auth)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(json.loads((Path(directory) / "latest.json").read_text())["task_queue"][0]["status"], "Assigned")

    async def test_recorded_dwell_uses_video_time(self):
        shopper = {"id": "person_01", "x": 50, "y": 50, "zone": "shelf4"}
        for second in (0, 4):
            response = await self.client.post("/api/telemetry", json={"shoppers": [shopper], "playback_seconds": second}, headers=self.ingest_headers)
            self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(sum(response.json()["occupancy"]["heatmap"]["seconds"]), 4, places=1)
        self.assertEqual(response.json()["occupancy"]["zones"]["shelf4"]["visits"], 1)

    async def test_queue_threshold_does_not_invent_staff_or_shelf_gap(self):
        payload = {
            "current_scenario": "final_test",
            "headcount": 6,
            "traffic": {"footfall_in": 9, "footfall_out": 3},
            "queues": {
                "counter_01": {"status": "ACTIVE", "queue_count": 4, "estimated_wait_time": 48},
                "counter_02": {"status": "IDLE", "queue_count": 0, "estimated_wait_time": 0},
            },
            "shelf_health": {"Dairy - 01": {"current_stock": 12, "max_capacity": 40}},
        }
        response = await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["headcount"], 6)
        self.assertEqual(state["traffic"], {"footfall_in": 9, "footfall_out": 3})
        self.assertEqual(state["queues"]["counter_02"]["status"], "IDLE")
        self.assertEqual(state["checkout"]["counter_02"]["status"], "standby")
        self.assertEqual(state["shelf_health"]["Dairy - 01"]["current_stock"], 12)
        self.assertEqual(state["inventory_health"]["shelf1"], 30)
        self.assertEqual(state["active_tickets"], [])
        self.assertNotIn("Backup cashier dispatched", str(state["log_lines"]))
        repeated = (await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)).json()
        self.assertEqual(repeated["active_tickets"], [])
        self.assertEqual(len([line for line in repeated["log_lines"] if "breaks threshold" in line]), 1)

    async def test_invalid_camera_does_not_partially_update(self):
        response = await self.client.post("/api/telemetry", json={"live_occupancy": 9, "camera_feeds": {"unknown": True}}, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(registry["live_occupancy"], 0)

    async def test_world_coordinate_ingest_and_zone_density(self):
        payload = {
            "coordinate_system": "world",
            "shoppers": [{"id": "person_01", "x": 650, "y": -600, "zone": "shelf4"}],
            "staff_02": {"x": 260, "y": -696},
            "checkout": {"counter_01": {"queue_count": 1, "wait_seconds": 3.5}},
            "inventory_health": {"shelf4": 78},
            "alerts": [{"type": "FLOW", "message": "Vision tracker synchronized"}],
        }
        response = await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["shoppers"][0]["x"], 50)
        self.assertEqual(state["shoppers"][0]["y"], 50)
        self.assertEqual(state["staff_02"], {"x": 20, "y": 42})
        self.assertEqual(state["occupancy"]["zones"]["shelf4"]["headcount"], 1)
        self.assertEqual(state["inventory_health"]["shelf4"], 78)
        self.assertEqual(state["system_logs"][0]["message"], "Vision tracker synchronized")
        invalid = await self.client.post("/api/telemetry", json={"shoppers": [{"id": "bad", "x": 140, "y": 10}]}, headers=self.ingest_headers)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(registry["shoppers"][0]["id"], "person_01")

    async def test_stream_sends_current_snapshot(self):
        class Request:
            async def is_disconnected(self):
                return False

        async def first_event():
            response = await stream(Request())
            generator = response.body_iterator
            chunk = await asyncio.wait_for(anext(generator), 1)
            await generator.aclose()
            return chunk

        chunk = await first_event()
        self.assertTrue(chunk.startswith("data: "))
        self.assertEqual(json.loads(chunk[6:].strip())["footfall"]["in"], 0)

    async def test_timeline_phases_and_task_resolution(self):
        start = await advance_timeline(0)
        self.assertEqual(start["live_occupancy"], 8)
        self.assertEqual(len(start["shoppers"]), 8)
        self.assertFalse(any(start["camera_feeds"].values()))
        stock = await advance_timeline(25)
        self.assertEqual(stock["inventory_tickets"][0]["status"], "Open")
        self.assertEqual(stock["inventory_tickets"][0]["missing"], 5)
        self.assertEqual(stock["inventory_health"]["shelf1"], 37)
        await advance_timeline(35)
        self.assertEqual({item["category"] for item in registry["task_queue"]}, {"Dairy", "Snacks"})
        queue = await advance_timeline(45)
        self.assertEqual(queue["checkout"]["counter_01"]["queue_count"], 4)
        self.assertIn("[QUEUE_ALERT]", str(queue["system_logs"]))
        dispatch = await advance_timeline(52)
        self.assertEqual(dispatch["checkout"]["counter_02"]["status"], "dispatching")
        active = await advance_timeline(55)
        self.assertEqual(active["checkout"]["counter_02"]["status"], "active")
        self.assertEqual(active["checkout"]["counter_02"]["queue_count"], 2)
        await self.client.patch("/api/tasks/MILK-001", json={"status": "Assigned"}, auth=self.operator_auth)
        self.assertEqual((await advance_timeline(56))["inventory_tickets"][1]["status"], "Assigned")
        exit_state = await advance_timeline(85)
        self.assertLess(exit_state["live_occupancy"], 8)
        self.assertGreater(exit_state["footfall"]["out"], start["footfall"]["out"])

    async def test_scenario_run_resets_results_and_starts_engine(self):
        registry["live_occupancy"] = 8
        registry["footfall"]["in"] = 12
        registry["headcount"] = 8

        class Process:
            pid = 4242
            returncode = None
            replay_log = None

            def terminate(self):
                self.returncode = -15

            async def wait(self):
                self.returncode = 0
                return 0

        process = Process()

        async def fake_spawn(*_args, **_kwargs):
            return process

        with TemporaryDirectory() as directory, patch("main.RUN_FILE", Path(directory) / "runs.json"), patch("main.LATEST_REPORT_FILE", Path(directory) / "latest.json"), patch("main.model_runtime_issue", return_value=None), patch("main.asyncio.create_subprocess_exec", fake_spawn):
            response = await self.client.post("/api/scenarios/final_test/run", auth=self.operator_auth)
            self.assertEqual((await self.client.get("/api/runs", auth=self.operator_auth)).json()["runs"][0]["id"], response.json()["run_id"])
            await asyncio.sleep(0)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "running")
        self.assertEqual(registry["current_scenario"], "final_test")
        self.assertEqual(registry["mode"], "vision")
        self.assertEqual(registry["measurement_source"], "model_inference")
        self.assertFalse(registry["measurements"]["footfall"])
        self.assertEqual(registry["playback"]["scenario"], "final_test")
        self.assertEqual(registry["live_occupancy"], 0)
        self.assertEqual(registry["footfall"]["in"], 0)
        self.assertEqual(registry["headcount"], 0)

    async def test_final_test_catalog_has_one_five_camera_take(self):
        with patch("main.model_runtime_issue", return_value=None):
            videos = (await self.client.get("/api/videos", auth=self.operator_auth)).json()["scenarios"]
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["id"], "final_test")
        self.assertEqual(videos[0]["duration_seconds"], 120)
        self.assertTrue(videos[0]["can_run"])
        self.assertTrue(all(camera["source"] == "final test" for camera in videos[0]["cameras"]))
        self.assertEqual((await self.client.post("/api/scenarios/missing/run", auth=self.operator_auth)).status_code, 404)
        self.assertEqual((await self.client.post("/api/scenarios/single_cashier/run", auth=self.operator_auth)).status_code, 404)

    async def test_video_manifest_and_upload_validation(self):
        with TemporaryDirectory() as directory, patch("main.VIDEO_DIR", Path(directory)), patch("main.FINAL_TEST_DIR", Path(directory) / "source"):
            manifest = (await self.client.get("/api/videos", auth=self.operator_auth)).json()
            self.assertEqual(len(manifest["scenarios"]), 1)
            self.assertTrue(all(len(item["cameras"]) == 5 for item in manifest["scenarios"]))
            self.assertFalse(manifest["scenarios"][0]["can_run"])
            path = "/api/videos/final_test/cam_checkout"
            self.assertEqual((await self.client.put(path, content=b"bad", headers={"Content-Type": "video/mp4"}, auth=self.operator_auth)).status_code, 422)
            self.assertEqual((await self.client.put("/api/videos/unknown/cam_checkout", content=b"bad", headers={"Content-Type": "video/mp4"}, auth=self.operator_auth)).status_code, 404)
            mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00"
            result = await self.client.put(path, content=mp4, headers={"Content-Type": "video/mp4"}, auth=self.operator_auth)
            self.assertEqual(result.status_code, 200, result.text)
            cameras = (await self.client.get("/api/videos", auth=self.operator_auth)).json()["scenarios"][0]["cameras"]
            self.assertEqual(next(item for item in cameras if item["id"] == "cam_checkout")["source"], "uploaded")
            self.assertFalse((await self.client.get("/api/videos", auth=self.operator_auth)).json()["scenarios"][0]["can_run"])
            self.assertEqual((Path(directory) / "final_test" / "cam_checkout.mp4").read_bytes(), mp4)

    async def test_camera_review_and_persistent_calibration(self):
        with TemporaryDirectory() as directory, patch("main.CALIBRATION_FILE", Path(directory) / "calibrations.json"), patch("main.FINAL_TEST_DIR", Path(directory) / "videos"), patch("main.VIDEO_DIR", Path(directory) / "uploads"):
            clip = Path(directory) / "videos" / "CCTV_Entrance_Aisle.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")
            video_path = "/api/videos/final_test/cam_entrance/file"
            self.assertEqual((await self.client.get(video_path)).status_code, 401)
            video = await review_video("final_test", "cam_entrance")
            self.assertEqual(Path(video.path), clip)
            self.assertEqual((await self.client.get("/api/videos/missing/cam_entrance/file", auth=self.operator_auth)).status_code, 404)

            calibration_path = "/api/calibrations/final_test/cam_entrance"
            self.assertEqual((await self.client.get(calibration_path, auth=self.operator_auth)).json()["regions"], {})
            regions = {"entry": {"type": "line", "x1": 10, "y1": 50, "x2": 40, "y2": 50}, "shelf1": {"type": "rect", "x1": 25, "y1": 20, "x2": 60, "y2": 70}}
            saved = await self.client.put(calibration_path, json={"regions": regions}, auth=self.operator_auth)
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual((await self.client.get(calibration_path, auth=self.operator_auth)).json()["regions"], regions)
            self.assertTrue((Path(directory) / "calibrations.json").is_file())
            bad = await self.client.put(calibration_path, json={"regions": {"entry": {"type": "rect", "x1": 10, "y1": 10, "x2": 20, "y2": 20}}}, auth=self.operator_auth)
            self.assertEqual(bad.status_code, 422)
            self.assertEqual((await self.client.get(calibration_path, auth=self.operator_auth)).json()["regions"], regions)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for evidence frames")
    async def test_evidence_frame_is_extracted_from_selected_clip(self):
        with TemporaryDirectory() as directory, patch("main.FINAL_TEST_DIR", Path(directory) / "videos"), patch("main.VIDEO_DIR", Path(directory) / "uploads"):
            clip = Path(directory) / "videos" / "CCTV_Entrance_Aisle.mp4"
            clip.parent.mkdir(parents=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-c:v", "mpeg4", "-y", str(clip)], check=True)
            path = "/api/videos/final_test/cam_entrance/frame?at=0.5"
            self.assertEqual((await self.client.get(path)).status_code, 401)
            frame = await self.client.get(path, auth=self.operator_auth)
            self.assertEqual(frame.status_code, 200, frame.text[:100] if frame.status_code != 200 else "")
            self.assertEqual(frame.headers["content-type"], "image/jpeg")
            self.assertTrue(frame.content.startswith(b"\xff\xd8"))
            self.assertEqual((await self.client.get("/api/videos/final_test/cam_entrance/frame?at=-1", auth=self.operator_auth)).status_code, 422)

    async def test_partial_replay_does_not_claim_unmeasured_results(self):
        registry["playback"] = {"status": "running", "scenario": "final_test", "pid": 42}
        response = await self.client.post("/api/telemetry", json={"headcount": 3, "playback_seconds": 119.5, "camera_feeds": {"cam_entrance": True}}, headers=self.ingest_headers)
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["playback_seconds"], 119.5)
        self.assertEqual(state["measurements"], {"visible_shoppers": True, "footfall": False, "queues": False, "positions": False, "shelf_gaps": False})
        self.assertEqual((await self.client.post("/api/telemetry", json={"playback_seconds": 120.1}, headers=self.ingest_headers)).status_code, 422)

    async def test_evidence_links_to_source_frame(self):
        evidence = {"camera_id": "cam_entrance", "video_seconds": 25.0, "frame_index": 750, "confidence": 0.91}
        payload = {"alerts": [{"type": "STOCK", "message": "Dairy facing has a persistent gap", "evidence": evidence}], "task_queue": [{"id": "MILK-2", "item": "Milk carton", "category": "Dairy", "location": "ShelfBox7", "missing": 8, "evidence": evidence}]}
        result = await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["task_queue"][0]["evidence"], evidence)
        self.assertEqual(result.json()["events"][0]["evidence"], evidence)
        payload["alerts"][0]["evidence"]["camera_id"] = "unknown"
        self.assertEqual((await self.client.post("/api/telemetry", json=payload, headers=self.ingest_headers)).status_code, 422)


if __name__ == "__main__":
    unittest.main()
