#!/usr/bin/env python3
"""Replay the completed Isaac baseline annotations as labeled reference telemetry."""

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
CAMERAS = {
    "CCTV_Entrance_Aisle": "cam_entrance",
    "CCTV_Checkout_Shelves": "cam_checkout",
    "CCTV_Back_Aisle": "cam_back_aisle",
    "CCTV_Entrance_Aisle_End": "cam_entrance_end",
    "CCTV_Back_Aisle_Left": "cam_back_aisle_left",
}


def token():
    value = os.environ.get("TRACK_K_INGEST_TOKEN", "").strip()
    if value:
        return value
    for line in (ROOT / "backend" / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("TRACK_K_INGEST_TOKEN="):
            return line.partition("=")[2].strip().strip("\"'")
    raise RuntimeError("TRACK_K_INGEST_TOKEN is missing")


def measurements(folder):
    events = [json.loads(line) for line in (folder / "events.jsonl").open(encoding="utf-8")]
    events = sorted((event for event in events if event.get("status") == "observed"), key=lambda event: event["timestamp_s"])
    entries = exits = event_index = 0
    waiting = {}
    with (folder / "annotations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if '"camera_id":"CCTV_Checkout_Shelves"' not in line:
                continue
            frame = json.loads(line)
            frame_index = frame["frame_index"]
            if frame_index % 3 and frame_index != 2699:
                continue
            second = frame["timestamp_s"]
            alerts = []
            while event_index < len(events) and events[event_index]["timestamp_s"] <= second + 0.00001:
                event = events[event_index]
                event_index += 1
                kind = event["event"]
                person = event["character_id"]
                if kind == "ENTRY":
                    entries += 1
                elif kind == "EXIT":
                    exits += 1
                elif kind == "QUEUE_JOIN":
                    waiting[person] = event["timestamp_s"]
                elif kind in {"QUEUE_LEAVE", "SERVICE_START"}:
                    waiting.pop(person, None)
                if kind in {"ENTRY", "EXIT"}:
                    alert = {"type": "FLOW", "message": f"{person} {'entered' if kind == 'ENTRY' else 'exited'} the store"}
                    visual = event.get("visual", {})
                    for camera_name, observation in visual.items():
                        box = observation.get("bbox")
                        if camera_name in CAMERAS and box and min(box[2] - box[0], box[3] - box[1]) >= 16:
                            alert["evidence"] = {"camera_id": CAMERAS[camera_name], "video_seconds": observation["timestamp_s"], "frame_index": observation["frame_index"]}
                            break
                    alerts.append(alert)
            shoppers = []
            for character in frame["characters"]:
                if character["class"] != "shopper" or character.get("phase") == "cleared":
                    continue
                x, y, _ = character["world_position_cm"]
                if 0 <= x <= 1300 and -1200 <= y <= 0:
                    shoppers.append({"id": character["character_id"], "x": x, "y": y})
            customers = [{"id": person, "wait_seconds": round(second - joined, 2)} for person, joined in waiting.items()]
            yield {
                "current_scenario": "single_cashier",
                "footfall": {"in": entries, "out": exits},
                "shoppers": shoppers,
                "coordinate_system": "world",
                "checkout": {
                    "counter_01": {"status": "active", "queue_count": len(customers), "wait_seconds": max((customer["wait_seconds"] for customer in customers), default=0), "customers": customers},
                    "counter_02": {"status": "standby", "queue_count": 0},
                },
                "task_queue": [],
                "camera_feeds": {camera: True for camera in CAMERAS.values()},
                "fps": 30,
                "playback_seconds": second,
                "alerts": alerts,
            }


def main():
    if sys.argv[1:] != ["single_cashier"]:
        raise SystemExit("Only the completed single_cashier baseline is supported")
    folder = Path(os.environ["TRACK_K_DATASET_DIR"]) / "baseline"
    endpoint = os.environ.get("TRACK_K_TELEMETRY_URL", "http://127.0.0.1:5173/api/telemetry")
    authorization = f"Bearer {token()}"
    started = time.monotonic()
    for payload in measurements(folder):
        time.sleep(max(0, started + payload["playback_seconds"] - time.monotonic()))
        request = Request(endpoint, data=json.dumps(payload).encode(), headers={"Authorization": authorization, "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Telemetry rejected with HTTP {response.status}")


if __name__ == "__main__":
    main()
