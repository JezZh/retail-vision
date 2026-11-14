"""Freeze image regions from the training split and the store camera geometry.

dynamic_staff annotations and events are not read.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CAMERAS,
    CHECKOUT_CAMERA,
    DATASET,
    ENTRANCE_CAMERA,
    FIT_FRACTION,
    FRAME_COUNTS,
    HEIGHT,
    ROOT,
    TRAIN_SCENARIOS,
    WIDTH,
    center_point,
    convex_hull,
    dilate_polygon,
    fit_cut,
    foot_point,
    line_side,
    point_in_polygon,
    project_world,
    valid_box,
    write_json,
)


def _median_box(boxes):
    arr = np.asarray(boxes, dtype=np.float64)
    med = np.median(arr, axis=0)
    return [float(v) for v in med]


def _scan_training(manifest, shelf_regions):
    entrance = manifest["cameras"][ENTRANCE_CAMERA]
    checkout = manifest["cameras"][CHECKOUT_CAMERA]
    door = manifest["doorway"]
    slot_boxes = defaultdict(list)
    queue_feet = []
    staff_samples = []
    entrance_people = defaultdict(list)
    visual_entries = []

    for scenario in TRAIN_SCENARIOS:
        event_path = DATASET / scenario / "events.jsonl"
        if event_path.stat().st_size:
            with event_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if (
                        event.get("event") == "ENTRY"
                        and event.get("status") == "observed"
                        and event.get("visual_status") == "observed"
                        and event.get("frame_index") is not None
                    ):
                        visual_entries.append(
                            {
                                "scenario": scenario,
                                "character_id": event["character_id"],
                                "frame_index": int(event["frame_index"]),
                            }
                        )
        n_frames = FRAME_COUNTS[scenario]
        cut = fit_cut(n_frames)
        facing_items = {}
        for shelf in shelf_regions.values():
            facing_items.setdefault(shelf["facing_camera"], set()).update(shelf["item_ids"])
        with (DATASET / scenario / "annotations.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                frame = int(record["frame_index"])
                camera_id = record["camera_id"]
                if frame >= cut:
                    continue
                if camera_id in facing_items and frame % 5 == 0:
                    for product in record["products"]:
                        if product["item_id"] not in facing_items[camera_id]:
                            continue
                        region = product.get("expected_region")
                        if region and len(region) == 4 and (region[2] - region[0]) > 2 and (region[3] - region[1]) > 2:
                            slot_boxes[product["item_id"]].append(region)
                if camera_id == ENTRANCE_CAMERA:
                    for person in record["characters"]:
                        if person["class"] != "shopper" or not valid_box(person.get("bbox")):
                            continue
                        if person["visibility"] not in ("visible", "partial"):
                            continue
                        entrance_people[(scenario, frame)].append(
                            {
                                "character_id": person["character_id"],
                                "foot": foot_point(person["bbox"]),
                            }
                        )
                if camera_id != CHECKOUT_CAMERA or frame % 5 != 0:
                    continue
                for person in record["characters"]:
                    box = person.get("bbox")
                    if not valid_box(box) or person["visibility"] not in ("visible", "partial"):
                        continue
                    if person["class"] == "shopper" and person.get("phase") in ("queue", "billing"):
                        queue_feet.append(foot_point(box))
                    if person["class"] == "staff":
                        staff_samples.append(
                            {
                                "center": center_point(box),
                                "foot": foot_point(box),
                                "bbox": [float(v) for v in box],
                                "height": float(box[3] - box[1]),
                            }
                        )
    return {
        "entrance": entrance,
        "checkout": checkout,
        "door": door,
        "slot_boxes": slot_boxes,
        "queue_feet": queue_feet,
        "staff_samples": staff_samples,
        "entrance_people": entrance_people,
        "visual_entries": visual_entries,
    }


def _door_line(scanned):
    camera = scanned["entrance"]
    door = scanned["door"]
    z = 0.0
    start = project_world([door["opening_x_min_cm"], door["y_cm"], z], camera)
    end = project_world([door["opening_x_max_cm"], door["y_cm"], z], camera)
    outside = project_world(
        [(door["opening_x_min_cm"] + door["opening_x_max_cm"]) * 0.5, 120.0, z],
        camera,
    )
    inside = project_world(
        [(door["opening_x_min_cm"] + door["opening_x_max_cm"]) * 0.5, -120.0, z],
        camera,
    )
    if start is None or end is None or outside is None:
        return {"available": False, "reason": "Doorway points did not project in front of the entrance camera."}
    outside_sign = 1 if line_side(outside, start, end) >= 0 else -1
    # Confirm the sign with training visual entries. Flip only if the training
    # crossings disagree with the geometric outside point.
    agree = 0
    disagree = 0
    checked = 0
    for event in scanned["visual_entries"]:
        before = scanned["entrance_people"].get((event["scenario"], max(0, event["frame_index"] - 10)), [])
        after = scanned["entrance_people"].get((event["scenario"], event["frame_index"] + 10), [])
        before = [p for p in before if p["character_id"] == event["character_id"]]
        after = [p for p in after if p["character_id"] == event["character_id"]]
        if not before or not after:
            continue
        checked += 1
        before_out = line_side(before[0]["foot"], start, end) * outside_sign >= 0
        after_out = line_side(after[0]["foot"], start, end) * outside_sign >= 0
        if before_out and not after_out:
            agree += 1
        elif (not before_out) and after_out:
            disagree += 1
    flipped = False
    if checked >= 3 and disagree > agree:
        outside_sign *= -1
        flipped = True
        agree, disagree = disagree, agree
    return {
        "available": True,
        "camera_id": ENTRANCE_CAMERA,
        "a": [float(start[0]), float(start[1])],
        "b": [float(end[0]), float(end[1])],
        "outside_sign": int(outside_sign),
        "inside_check_projected": None if inside is None else [float(inside[0]), float(inside[1])],
        "max_distance_px": 90.0,
        "sign_source": "training_entry_boxes" if flipped else "projected_outside_point",
        "training_entry_checks": checked,
        "training_entry_agree": agree,
        "training_entry_disagree": disagree,
        "reason": None,
    }


def _queue_polygons(scanned, manifest):
    camera = scanned["checkout"]
    counters = manifest["checkout_regions"]
    feet = scanned["queue_feet"]
    splits = {
        "counter_01": (-280.0, -467.0),
        "counter_02": (-467.0, -820.0),
    }
    polygons = {}
    for name, (y_near, y_far) in splits.items():
        world = [
            [250.0, y_near, 0.0],
            [560.0, y_near, 0.0],
            [560.0, y_far, 0.0],
            [250.0, y_far, 0.0],
        ]
        projected = [point for point in (project_world(point, camera) for point in world) if point is not None]
        service_name = "Counter_01" if name == "counter_01" else "Counter_02"
        anchor = project_world(counters[service_name]["service_point_cm"], camera)
        polygons[name] = {
            "camera_id": CHECKOUT_CAMERA,
            "polygon": [[float(x), float(y)] for x, y in convex_hull(projected)],
            "method": "projected_lane",
            "service_anchor_px": None if anchor is None else [float(anchor[0]), float(anchor[1])],
        }
    a1 = polygons["counter_01"]["service_anchor_px"]
    a2 = polygons["counter_02"]["service_anchor_px"]
    groups = {"counter_01": [], "counter_02": []}
    if a1 and a2:
        for foot in feet:
            d1 = (foot[0] - a1[0]) ** 2 + (foot[1] - a1[1]) ** 2
            d2 = (foot[0] - a2[0]) ** 2 + (foot[1] - a2[1]) ** 2
            groups["counter_01" if d1 <= d2 else "counter_02"].append(foot)
    for name, group in groups.items():
        polygon = [tuple(point) for point in polygons[name]["polygon"]]
        covered = (
            sum(1 for foot in group if point_in_polygon(foot[0], foot[1], polygon)) / len(group)
            if group and polygon
            else 0.0
        )
        method = polygons[name]["method"]
        if covered < 0.75 and len(group) >= 8:
            polygon = dilate_polygon(convex_hull(group), 30.0)
            covered = sum(1 for foot in group if point_in_polygon(foot[0], foot[1], polygon)) / len(group)
            method = "training_queue_feet_split_by_register"
        polygons[name].update(
            {
                "polygon": [[float(x), float(y)] for x, y in polygon],
                "method": method,
                "training_feet": len(group),
                "training_feet_coverage": float(covered),
                "available": len(polygon) >= 3 and covered >= 0.5,
            }
        )
    return polygons


def _staff_rule(scanned, polygons):
    anchor = polygons.get("counter_01", {}).get("service_anchor_px")
    samples = scanned["staff_samples"]
    if not anchor or len(samples) < 10:
        return {"available": False, "reason": "Checkout staff boxes were not available on the training split."}
    for sample in samples:
        dx = sample["center"][0] - anchor[0]
        dy = sample["center"][1] - anchor[1]
        sample["anchor_dist"] = float((dx * dx + dy * dy) ** 0.5)
    samples = sorted(samples, key=lambda sample: sample["anchor_dist"])
    # The cashier is the nearer mode. Split at the largest gap in the distance list.
    distances = [sample["anchor_dist"] for sample in samples]
    gap_at = int(np.argmax(np.diff(distances))) if len(distances) > 2 else 0
    cashier = samples[: gap_at + 1]
    other = samples[gap_at + 1 :]
    if len(other) < 5:
        return {
            "available": False,
            "reason": "A second staff image location was not stable on the training checkout camera.",
            "cashier_count": len(cashier),
            "other_count": len(other),
        }
    home = np.median(np.asarray([sample["center"] for sample in other], dtype=np.float64), axis=0)
    home_dist = [
        float(((sample["center"][0] - home[0]) ** 2 + (sample["center"][1] - home[1]) ** 2) ** 0.5)
        for sample in other
    ]
    cashier_dist = [sample["anchor_dist"] for sample in cashier]
    return {
        "available": True,
        "camera_id": CHECKOUT_CAMERA,
        "cashier_anchor_px": [float(anchor[0]), float(anchor[1])],
        "cashier_radius_px": float(np.percentile(cashier_dist, 95) + 40.0),
        "home_px": [float(home[0]), float(home[1])],
        "dispatch_distance_px": float(max(home_dist) + 40.0),
        "other_median_height_px": float(np.median([sample["height"] for sample in other])),
        "cashier_median_height_px": float(np.median([sample["height"] for sample in cashier])),
        "rule": (
            "A non-cashier staff track on the checkout camera is dispatched when its center "
            "stays farther than dispatch_distance_px from the frozen training home for "
            "15 frames. Staff are excluded from shopper and queue counts."
        ),
    }


def _slots(scanned, shelf_regions):
    slots = []
    for shelf in shelf_regions.values():
        for item_id in shelf["item_ids"]:
            boxes = scanned["slot_boxes"].get(item_id, [])
            if len(boxes) < 5:
                slots.append(
                    {
                        "item_id": item_id,
                        "shelf_id": shelf["shelf_id"],
                        "category": shelf["category"],
                        "facing_camera": shelf["facing_camera"],
                        "roi": None,
                        "available": False,
                    }
                )
                continue
            roi = _median_box(boxes)
            # One-pixel pad so the crop includes the facing edge. Still a fixed box.
            roi = [
                max(0.0, roi[0] - 1.0),
                max(0.0, roi[1] - 1.0),
                min(WIDTH - 1.0, roi[2] + 1.0),
                min(HEIGHT - 1.0, roi[3] + 1.0),
            ]
            slots.append(
                {
                    "item_id": item_id,
                    "shelf_id": shelf["shelf_id"],
                    "category": shelf["category"],
                    "facing_camera": shelf["facing_camera"],
                    "roi": [round(v, 2) for v in roi],
                    "roi_source_frames": len(boxes),
                    "available": True,
                }
            )
    return slots


def build_calibration() -> dict:
    validation = json.loads((DATASET / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise SystemExit("Refusing to calibrate because validation.passed is not true.")
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    shelf_regions = json.loads((DATASET / "shelf_regions.json").read_text(encoding="utf-8"))
    # Self-check the projection on the known baseline entrance foot.
    entrance = manifest["cameras"][ENTRANCE_CAMERA]
    pixel = project_world([411.65, -71.16, 1.8], entrance)
    known = [611, 537, 764, 719]
    # The labeled box is clipped at the bottom of the 720-pixel frame, so the
    # projected foot can sit a few pixels below y=719 while x stays inside.
    inside = (
        pixel is not None
        and known[0] <= pixel[0] <= known[2]
        and known[1] - 8 <= pixel[1] <= known[3] + 8
    )
    if not inside:
        raise RuntimeError(f"Projection self-check failed: {pixel}")
    scanned = _scan_training(manifest, shelf_regions)
    door = _door_line(scanned)
    queues = _queue_polygons(scanned, manifest)
    staff = _staff_rule(scanned, queues)
    slots = _slots(scanned, shelf_regions)
    payload = {
        "format": "track_k_v2_calibration",
        "fit_fraction": FIT_FRACTION,
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "projection": "row_vector_world_times_inverse_camera_matrix_y_flipped",
        "projection_self_check": {"world_cm": [411.65, -71.16, 1.8], "pixel": list(pixel), "inside_box": known},
        "held_out_not_used": "dynamic_staff",
        "door": door,
        "queues": queues,
        "staff": staff,
        "slots": slots,
        "cameras": {
            camera_id: {
                "focal_length_mm": manifest["cameras"][camera_id]["focal_length_mm"],
                "horizontal_aperture_mm": manifest["cameras"][camera_id]["horizontal_aperture_mm"],
                "vertical_aperture_mm": manifest["cameras"][camera_id]["vertical_aperture_mm"],
                "eye_cm": manifest["cameras"][camera_id]["eye_cm"],
                "world_matrix_row_major": manifest["cameras"][camera_id]["world_matrix_row_major"],
            }
            for camera_id in CAMERAS
        },
        "limitations": [
            "Floor occupancy and the heatmap are unavailable. Feet are not lifted to a shared floor map, and identities are not merged across cameras.",
            "Slot ROIs are the median training expected_region on the facing camera. They belong to this store and these camera poses.",
            "dynamic_staff was not used to place the door, the queue polygons, the staff home, or the slot boxes.",
        ],
    }
    write_json(ROOT / "calibration" / "calibration.json", payload)
    print(
        "CALIBRATION door",
        door.get("available"),
        "slots",
        sum(1 for slot in slots if slot["available"]),
        "queue",
        {name: item["available"] for name, item in queues.items()},
        "staff",
        staff.get("available"),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    build_calibration()
