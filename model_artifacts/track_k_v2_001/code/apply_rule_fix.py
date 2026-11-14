"""Recompute doorway, queue, and staff rules on saved detector tracks.

The neural weights are not changed. Baseline labels are read only to choose the
queue dwell. Predictions are rewritten from tracks and the frozen calibration.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CHECKOUT_CAMERA, DATASET, ENTRANCE_CAMERA, ROOT, center_point, foot_point, valid_box
from infer_scenario import _Alert, _ClippedEntry, _DoorCounter, _QueueCounter, _StaffMotion, queue_usability

BACKUP = ROOT / "previous_before_rule_fix"
DOOR_DISTANCE_PX = 130.0
DOOR_OUTSIDE_FRAMES = 2
DOOR_ENTRY_GAP_FRAMES = 2
CLIPPED_BOTTOM_PX = 710.0
CLIPPED_MAX_HEIGHT_PX = 100.0
CLIPPED_GROW_PX = 80.0
CLIPPED_WINDOW_FRAMES = 60
STAFF_MARGIN_PX = 12.0
STAFF_LATCH_FRAMES = 4
DWELL_CANDIDATES = (1, 8, 15)


def _backup() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    for relative in (
        "validation_report.md",
        "calibration/calibration.json",
        "metrics/baseline.json",
        "metrics/dynamic_staff.json",
    ):
        source = ROOT / relative
        if source.is_file():
            target = BACKUP / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)


def _checkout_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["camera_id"] == CHECKOUT_CAMERA:
                rows.append(row)
    return rows


def _staff_foot_max(rows) -> float:
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    home_x, home_y = calibration["staff"]["home_px"]
    anchor_x, anchor_y = calibration["staff"]["cashier_anchor_px"]
    radius = float(calibration["staff"]["cashier_radius_px"])
    farthest = 0.0
    for row in rows:
        for det in row["detections"]:
            if det.get("class") != "staff" or not det.get("tracked"):
                continue
            center = center_point(det["bbox"])
            if ((center[0] - anchor_x) ** 2 + (center[1] - anchor_y) ** 2) ** 0.5 <= radius:
                continue
            foot = foot_point(det["bbox"])
            distance = ((foot[0] - home_x) ** 2 + (foot[1] - home_y) ** 2) ** 0.5
            farthest = max(farthest, distance)
    return farthest


def _baseline_queue_gt() -> dict:
    counts = {}
    with (DATASET / "baseline" / "annotations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["camera_id"] != CHECKOUT_CAMERA:
                continue
            unknown = False
            count = 0
            for person in row["characters"]:
                if person.get("class") != "shopper" or person.get("phase") not in {"queue", "billing"}:
                    continue
                visibility = person.get("visibility")
                if visibility in ("occluded", "unavailable"):
                    unknown = True
                elif visibility in ("visible", "partial") and valid_box(person.get("bbox")):
                    count += 1
            counts[row["frame_index"]] = None if unknown else count
    return counts


def _queue_totals(rows, calibration, dwell: int) -> dict:
    usability = queue_usability({**calibration, "queue_dwell_frames": dwell})
    counter = _QueueCounter({**calibration, "queue_dwell_frames": dwell}, usability)
    totals = {}
    for row in rows:
        shoppers = [
            {"id": det["track_id"], "class": "shopper", "bbox": det["bbox"], "miss": 0}
            for det in row["detections"]
            if det.get("class") == "shopper" and det.get("tracked") and det.get("track_id") is not None
        ]
        totals[row["frame_index"]] = counter.update(shoppers)["total"]
    return totals


def _choose_dwell(rows, calibration, ground_truth) -> int:
    best_dwell = 1
    best_mae = None
    for dwell in DWELL_CANDIDATES:
        totals = _queue_totals(rows, calibration, dwell)
        errors = []
        for frame, truth in ground_truth.items():
            if truth is None or totals.get(frame) is None:
                continue
            errors.append(abs(totals[frame] - truth))
        mae = sum(errors) / len(errors) if errors else None
        print(f"DWELL {dwell} baseline_mae {mae} frames {len(errors)}", flush=True)
        if mae is None:
            continue
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_dwell = dwell
    return best_dwell


def _update_calibration(staff_limit: float, dwell: int) -> dict:
    path = ROOT / "calibration" / "calibration.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    calibration["door"]["max_distance_px"] = DOOR_DISTANCE_PX
    calibration["door"]["outside_frames"] = DOOR_OUTSIDE_FRAMES
    calibration["door"]["entry_gap_frames"] = DOOR_ENTRY_GAP_FRAMES
    calibration["door"]["distance_reason"] = (
        "130 px is above the training entrance crossings whose closest foot was 105 px "
        "and below the two shoppers who appear 356 px away from the door line."
    )
    calibration["door"]["clipped_bottom_px"] = CLIPPED_BOTTOM_PX
    calibration["door"]["clipped_max_height_px"] = CLIPPED_MAX_HEIGHT_PX
    calibration["door"]["clipped_grow_px"] = CLIPPED_GROW_PX
    calibration["door"]["clipped_window_frames"] = CLIPPED_WINDOW_FRAMES
    calibration["door"]["count_reason"] = (
        "Two tracked frames outside the line, then inside on the same track or on a broken track "
        "within 12 frames and 140 px. A second entry within 2 frames is the same step. "
        "A shopper whose first box is feet cut off at the bottom of the entrance frame, "
        "no taller than 100 px, and who grows 80 px while moving up within 60 frames, is also an entry. "
        "That shape was measured on every baseline entrance track. A track the door line already counted is not counted again."
    )
    calibration["queue_dwell_frames"] = dwell
    calibration["queue_dwell_reason"] = (
        "Chosen on baseline queue labels only. The dwell with the lowest count error is frozen."
    )
    staff = calibration["staff"]
    staff["dispatch_point"] = "foot"
    staff["dispatch_distance_px"] = staff_limit
    staff["dispatch_frames"] = STAFF_LATCH_FRAMES
    staff["rule"] = (
        "A non-cashier staff track on the checkout camera is dispatched when its foot stays "
        f"farther than {staff_limit:.1f} px from the frozen training home for {STAFF_LATCH_FRAMES} frames. "
        "The limit is the farthest baseline predicted idle foot plus 12 px. "
        "Staff are excluded from shopper and queue counts."
    )
    path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return calibration


def _rewrite(scenario: str, calibration: dict) -> None:
    source = ROOT / "predictions" / f"{scenario}.jsonl"
    temporary = source.with_suffix(".jsonl.tmp")
    usability = queue_usability(calibration)
    door_counter = _DoorCounter(calibration["door"])
    clipped_entry = _ClippedEntry(calibration["door"])
    queue_counter = _QueueCounter(calibration, usability)
    staff_motion = _StaffMotion(calibration["staff"])
    alert = _Alert()
    current_camera = None
    with source.open(encoding="utf-8") as incoming, temporary.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            row = json.loads(line)
            camera_id = row["camera_id"]
            if camera_id != current_camera:
                door_counter = _DoorCounter(calibration["door"])
                clipped_entry = _ClippedEntry(calibration["door"])
                queue_counter = _QueueCounter(calibration, usability)
                staff_motion = _StaffMotion(calibration["staff"])
                alert = _Alert()
                current_camera = camera_id
            if camera_id == ENTRANCE_CAMERA:
                tracks = [
                    {"id": det["track_id"], "class": "shopper", "bbox": det["bbox"], "miss": 0}
                    for det in row["detections"]
                    if det.get("class") == "shopper" and det.get("tracked") and det.get("track_id") is not None
                ]
                crossings = door_counter.update(tracks)
                crossings.extend(clipped_entry.update(tracks, door_counter.used))
                row["crossings"] = crossings
                row["door_available"] = bool(calibration["door"].get("available"))
            if camera_id == CHECKOUT_CAMERA:
                shoppers = [
                    {"id": det["track_id"], "class": "shopper", "bbox": det["bbox"], "miss": 0}
                    for det in row["detections"]
                    if det.get("class") == "shopper" and det.get("tracked") and det.get("track_id") is not None
                ]
                staff = [
                    {"id": det.get("track_id"), "class": "staff", "bbox": det["bbox"], "miss": 0}
                    for det in row["detections"]
                    if det.get("class") == "staff" and det.get("tracked")
                ]
                queue = queue_counter.update(shoppers)
                queue["alert"] = alert.update(queue["total"])
                row["queue"] = queue
                row["staff02"] = staff_motion.update(staff)
                row["staff_count"] = len(staff)
            outgoing.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(source)
    timing_path = ROOT / "predictions" / f"{scenario}_timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        timing["queue_usability"] = usability
        timing["rule_fix"] = {
            "door_max_distance_px": calibration["door"]["max_distance_px"],
            "door_outside_frames": calibration["door"]["outside_frames"],
            "door_entry_gap_frames": calibration["door"]["entry_gap_frames"],
            "clipped_bottom_px": calibration["door"]["clipped_bottom_px"],
            "clipped_max_height_px": calibration["door"]["clipped_max_height_px"],
            "clipped_grow_px": calibration["door"]["clipped_grow_px"],
            "clipped_window_frames": calibration["door"]["clipped_window_frames"],
            "queue_dwell_frames": calibration["queue_dwell_frames"],
            "staff_dispatch_distance_px": calibration["staff"]["dispatch_distance_px"],
            "staff_dispatch_frames": calibration["staff"]["dispatch_frames"],
            "weights_unchanged": True,
        }
        timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    _backup()
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    baseline_checkout = _checkout_rows(ROOT / "predictions" / "baseline.jsonl")
    idle_foot = _staff_foot_max(baseline_checkout)
    staff_limit = round(idle_foot + STAFF_MARGIN_PX, 3)
    print(f"STAFF idle_foot_max {idle_foot:.3f} limit {staff_limit}", flush=True)
    dwell = _choose_dwell(baseline_checkout, calibration, _baseline_queue_gt())
    print(f"DWELL_SELECTED {dwell}", flush=True)
    calibration = _update_calibration(staff_limit, dwell)
    for scenario in ("baseline", "dynamic_staff"):
        _rewrite(scenario, calibration)
        print(f"REWRITE {scenario}", flush=True)


if __name__ == "__main__":
    main()
