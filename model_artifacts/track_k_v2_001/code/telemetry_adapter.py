"""Map vision predictions onto the Track K telemetry body.

This module writes JSON files only. It does not open a socket and it does not
change the dashboard mode. Posting to /api/telemetry would switch the dashboard
to external vision, and that switch waits until a model-driven run is accepted.
"""

from __future__ import annotations

import json
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CAMERAS, CHECKOUT_CAMERA, ENTRANCE_CAMERA, ROOT, foot_point, line_side, point_segment_distance, write_json

# Folder name to dashboard scenario. This map is not used by the detector.
FOLDER_SCENARIO = {
    "baseline": "single_cashier",
    "dynamic_staff": "dynamic_staff",
    "final_test": "final_test",
}
CAMERA_KEYS = {
    "CCTV_Entrance_Aisle": "cam_entrance",
    "CCTV_Checkout_Shelves": "cam_checkout",
    "CCTV_Back_Aisle": "cam_back_aisle",
    "CCTV_Entrance_Aisle_End": "cam_entrance_end",
    "CCTV_Back_Aisle_Left": "cam_back_aisle_left",
}
SHELF_BINDINGS = {
    "Milk": {
        "name": "Dairy - 01",
        "category": "Dairy",
        "location": "Aisle_1_Tier_2",
        "camera_key": "cam_entrance",
        "facing_capacity": 8,
        "dashboard_seed_capacity": 40,
    },
    "Packaged Snacks": {
        "name": "Snacks - 02",
        "category": "Snacks",
        "location": "Aisle_2_Tier_1",
        "camera_key": "cam_back_aisle_left",
        "facing_capacity": 10,
        "dashboard_seed_capacity": 48,
    },
}


def _shelf_body(slots, camera_id=None, video_seconds=None, frame_index=None):
    if not slots:
        return None, [], []
    grouped = {}
    for slot in slots:
        grouped.setdefault(slot["category"], []).append(slot)
    health = {}
    tasks = []
    alerts = []
    for category, binding in SHELF_BINDINGS.items():
        rows = grouped.get(category)
        if not rows:
            continue
        empty = sum(1 for row in rows if row["state"] == "empty")
        full = sum(1 for row in rows if row["state"] == "full")
        if not any(row["state"] == "unknown" for row in rows):
            health[binding["name"]] = {"current_stock": full, "max_capacity": binding["facing_capacity"]}
        if empty >= 1:
            evidence = {"camera_id": CAMERA_KEYS[camera_id], "video_seconds": video_seconds, "frame_index": frame_index} if camera_id in CAMERA_KEYS and video_seconds is not None else None
            tasks.append(
                {
                    "id": f"{binding['category'].lower()}-gap",
                    "item": category,
                    "category": binding["category"],
                    "location": binding["location"],
                    "missing": empty,
                    **({"evidence": evidence} if evidence else {}),
                }
            )
            alerts.append(
                {
                    "type": "STOCK",
                    "message": f"{binding['name']} facing has {empty} empty slots.",
                    **({"evidence": evidence} if evidence else {}),
                }
            )
    return (health or None), tasks, alerts


def _floor_point(box, camera):
    """Intersect a detected foot ray with the calibrated USD floor (z=0 cm)."""
    matrix = camera["world_matrix_row_major"]
    fx = camera["focal_length_mm"] / camera["horizontal_aperture_mm"] * 1280
    fy = camera["focal_length_mm"] / camera["vertical_aperture_mm"] * 720
    u, v = foot_point(box)
    dx, dy = (u - 640) / fx, (360 - v) / fy
    ray = (
        dx * matrix[0] + dy * matrix[4] - matrix[8],
        dx * matrix[1] + dy * matrix[5] - matrix[9],
        dx * matrix[2] + dy * matrix[6] - matrix[10],
    )
    if ray[2] >= -1e-6:
        return None
    scale = -matrix[14] / ray[2]
    x, y = matrix[12] + scale * ray[0], matrix[13] + scale * ray[1]
    return (round(x, 2), round(y, 2)) if 0 <= x <= 1300 and -1200 <= y <= 0 else None


def _visible_positions(rows, calibration, limit):
    candidates = []
    for camera_id in CAMERAS:
        row = rows.get(camera_id) or {}
        camera = calibration["cameras"][camera_id]
        for detection in row.get("detections") or []:
            if detection["class"] != "shopper" or not detection["tracked"] or detection["bbox"][3] >= 715:
                continue
            point = _floor_point(detection["bbox"], camera)
            if point is None:
                continue
            candidates.append({"id": f"{CAMERA_KEYS[camera_id]}_{detection['track_id']}", "x": point[0], "y": point[1], "camera": camera_id, "score": detection["score"]})
    for point in candidates:
        point["support"] = len({other["camera"] for other in candidates if other["camera"] != point["camera"] and (point["x"] - other["x"]) ** 2 + (point["y"] - other["y"]) ** 2 < 100 ** 2})
    candidates.sort(key=lambda point: (point["support"], point["score"]), reverse=True)
    positions = []
    for point in candidates:
        if len(positions) >= limit:
            break
        if point["camera"] == "CCTV_Entrance_Aisle_End" and not point["support"]:
            continue
        if any((point["x"] - other["x"]) ** 2 + (point["y"] - other["y"]) ** 2 < 45 ** 2 for other in positions):
            continue
        positions.append(point)
    return [{key: point[key] for key in ("id", "x", "y")} for point in positions]


def _crossings_by_frame(entrance_rows, door):
    """Confirm short-lived entry tracks and count inside-to-outside door crossings."""
    track_frames = defaultdict(list)
    for row in entrance_rows:
        for detection in row.get("detections") or []:
            if detection["class"] == "shopper" and detection["tracked"]:
                track_frames[detection["track_id"]].append(row["frame_index"])
    crossings = defaultdict(list)
    state = defaultdict(lambda: {"far_inside": False, "inside_run": 0, "exited": False})
    for row in entrance_rows:
        frame = row["frame_index"]
        for crossing in row.get("crossings") or []:
            if crossing["kind"] != "entry":
                continue
            later = sum(frame < seen <= frame + 30 for seen in track_frames[crossing["track_id"]])
            if later >= 10:
                crossings[frame].append(crossing)
        for detection in row.get("detections") or []:
            if detection["class"] != "shopper" or not detection["tracked"]:
                continue
            foot = foot_point(detection["bbox"])
            distance = point_segment_distance(foot, door["a"], door["b"])
            inside = line_side(foot, door["a"], door["b"]) * door["outside_sign"] < 0
            track = state[detection["track_id"]]
            if inside and distance > 200:
                track["far_inside"] = True
            if distance > door["max_distance_px"]:
                continue
            if inside:
                track["inside_run"] += 1
            else:
                if track["far_inside"] and track["inside_run"] >= 2 and not track["exited"]:
                    crossings[frame].append({"kind": "exit", "track_id": detection["track_id"]})
                    track["exited"] = True
                track["inside_run"] = 0
    return crossings


def _lane(count, dispatched=False):
    if count is None:
        return None
    status = "ACTIVE" if count else "IDLE"
    checkout_status = "dispatching" if dispatched else ("active" if count else "standby")
    return {
        "queues": {"status": status, "queue_count": count},
        "checkout": {"queue_count": count, "status": checkout_status},
    }


def body_for_frame(folder_name: str, rows_by_camera: dict, fps: float | None) -> dict:
    """Build one API body from the five camera rows that share a frame index."""
    checkout = rows_by_camera.get(CHECKOUT_CAMERA) or {}
    queue = checkout.get("queue") or {}
    staff = checkout.get("staff02") or {}
    dispatched = staff.get("dispatched") is True
    first_count = queue.get("counter_01", {}).get("count") if isinstance(queue.get("counter_01"), dict) else None
    second_count = queue.get("counter_02", {}).get("count") if isinstance(queue.get("counter_02"), dict) else None
    # shortcut: this scene opens Counter 02 only after observed Staff_02 movement; use cashier occupancy for other stores.
    if staff.get("observable") is True and staff.get("dispatched") is False and queue.get("total") is not None:
        first_count, second_count = queue["total"], 0
    counter_01 = _lane(first_count)
    counter_02 = _lane(
        second_count,
        dispatched=dispatched,
    )
    body = {}
    scenario = FOLDER_SCENARIO.get(folder_name)
    if scenario:
        body["current_scenario"] = scenario
    timestamp = checkout.get("timestamp_s")
    if timestamp is None:
        timestamp = next((row.get("timestamp_s") for row in rows_by_camera.values() if row), None)
    if timestamp is not None and 0 <= float(timestamp) <= 120:
        body["playback_seconds"] = round(float(timestamp), 5)
    if fps is not None and 0 <= fps <= 240:
        body["fps"] = round(float(fps), 3)
    body["camera_feeds"] = {CAMERA_KEYS[camera]: camera in rows_by_camera for camera in CAMERAS}
    queues = {}
    checkout_body = {}
    if counter_01:
        queues["counter_01"] = counter_01["queues"]
        checkout_body["counter_01"] = counter_01["checkout"]
    if counter_02:
        queues["counter_02"] = counter_02["queues"]
        checkout_body["counter_02"] = counter_02["checkout"]
    if queues:
        body["queues"] = queues
    if checkout_body:
        body["checkout"] = checkout_body
    if queue.get("total") is not None:
        body["queue_count"] = queue["total"]
    alerts = []
    if queue.get("alert") and first_count is not None and first_count >= 4:
        alerts.append(
            {
                "type": "QUEUE",
                "message": f"[QUEUE_ALERT] Counter 01 queue length {first_count} breaks threshold.",
            }
        )
    elif queue.get("alert") and queue.get("total") is not None and queue["total"] >= 4:
        alerts.append(
            {
                "type": "QUEUE",
                "message": f"[QUEUE_ALERT] Checkout queue length {queue['total']} breaks threshold.",
            }
        )
    if dispatched:
        alerts.append(
            {
                "type": "STAFF",
                "message": "Checkout camera shows the non-cashier staff away from the frozen home position.",
            }
        )
    shelf_health = None
    tasks = []
    for row in rows_by_camera.values():
        health, row_tasks, row_alerts = _shelf_body(row.get("shelf_slots"), row.get("camera_id"), row.get("timestamp_s"), row.get("frame_index"))
        if health:
            shelf_health = {**(shelf_health or {}), **health}
        tasks.extend(row_tasks)
        alerts.extend(row_alerts)
    if shelf_health:
        body["shelf_health"] = shelf_health
    if tasks:
        body["task_queue"] = tasks[:20]
    if alerts:
        body["alerts"] = alerts[:20]
    return body


def write_payloads(scenario_dir: Path, predictions: Path, timing: dict, output: Path) -> dict:
    by_frame = {}
    with predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_frame.setdefault(int(row["frame_index"]), {})[row["camera_id"]] = row
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    entrance_rows = [rows[ENTRANCE_CAMERA] for _, rows in sorted(by_frame.items()) if ENTRANCE_CAMERA in rows]
    crossings_by_frame = _crossings_by_frame(entrance_rows, calibration["door"])
    entries = sum(crossing["kind"] == "entry" for crossings in crossings_by_frame.values() for crossing in crossings)
    exits = sum(crossing["kind"] == "exit" for crossings in crossings_by_frame.values() for crossing in crossings)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    running_in = 0
    running_out = 0
    fps = None if timing is None else timing.get("end_to_end_fps")
    with output.open("w", encoding="utf-8") as handle:
        for frame in sorted(by_frame):
            rows = by_frame[frame]
            for crossing in crossings_by_frame.get(frame, []):
                if crossing.get("kind") == "entry":
                    running_in += 1
                elif crossing.get("kind") == "exit":
                    running_out += 1
            if frame % 30 != 0 and not (rows.get(CHECKOUT_CAMERA) or {}).get("queue", {}).get("alert"):
                continue
            body = body_for_frame(scenario_dir.name, rows, fps)
            occupancy = max(0, running_in - running_out)
            shoppers = _visible_positions(rows, calibration, occupancy)
            body.update({"shoppers": shoppers, "headcount": occupancy, "live_occupancy": occupancy, "coordinate_system": "world"})
            body["traffic"] = {"footfall_in": running_in, "footfall_out": running_out}
            body["footfall"] = {"in": running_in, "out": running_out}
            handle.write(
                json.dumps(
                    {
                        "posted": False,
                        "api_path": "/api/telemetry",
                        "source_folder": str(scenario_dir),
                        "frame_index": frame,
                        "body": body,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            written += 1
    summary = {
        "posted": False,
        "payload_file": str(output),
        "payloads_written": written,
        "predicted_entries": entries,
        "predicted_exits": exits,
        "omitted": [
            "staff_02 position",
            "named customer waits",
            "cross-camera shopper identity",
        ],
        "scenario_map": FOLDER_SCENARIO,
        "facing_capacity_vs_dashboard_seed": {
            name: {"facing": spec["facing_capacity"], "dashboard_seed": spec["dashboard_seed_capacity"]}
            for name, spec in SHELF_BINDINGS.items()
        },
    }
    write_json(output.with_name(output.stem + "_summary.json"), summary)
    return summary


def write_mapping_doc(path: Path) -> None:
    path.write_text(
        """# Track K telemetry mapping

The final-test Run action reads five video streams, produces model predictions, converts them with this adapter, and posts the resulting bodies to `/api/telemetry`. The final-test annotations and events are never read during inference.

| Model output | Dashboard field | Limit |
| --- | --- | --- |
| Entrance camera crossings | `footfall.in` and `footfall.out` | Short-lived entry detections are rejected; exits require a tracked move from the store interior across the doorway. |
| Checkout polygon counts | `queues` and `checkout` | Wait and service times are not measured. |
| Queue and staff signals | `alerts` | Staff image position is not a floor coordinate. |
| Known milk and chip facing states | `shelf_health`, `task_queue` | Confirmed empty slots create tickets even when another slot is unknown. Unknown slots do not become missing items. |
| Decoded cameras and frame clock | `camera_feeds`, `playback_seconds`, `fps` | Test take is 120 seconds; FPS is processing throughput. |
| Calibrated shopper foot rays | `shoppers`, `headcount`, `occupancy.heatmap` | Camera-visible positions only. Close positions from different cameras are merged spatially; identities are not reconciled. |

The map and heatmap represent camera-visible shoppers, not a guaranteed census of every person in the store. Named customer waits remain unavailable.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_mapping_doc(ROOT / "telemetry" / "OUTPUT_MAPPING.md")
