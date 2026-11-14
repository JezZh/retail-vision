"""Score prediction JSONL against simulator labels. Scoring is the only step that reads labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CAMERAS,
    CHECKOUT_CAMERA,
    DATASET,
    ENTRANCE_CAMERA,
    FPS,
    PERSON_CLASSES,
    ROOT,
    average_precision,
    f1_value,
    iou,
    match_detections,
    pr_at_threshold,
    valid_box,
    write_json,
)

FORBIDDEN_KEYS = {
    "character_id",
    "world_position_cm",
    "expected_region",
    "visual_status",
    "occlusion_ratio",
    "prim_path",
}
IOU_GRID = [round(0.50 + 0.05 * step, 2) for step in range(10)]
QUEUE_PHASES = {"queue", "billing"}


def _walk_keys(value, found: set) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                found.add(key)
            _walk_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, found)


def _gt_shelf_state(product) -> str:
    visibility = product.get("visibility")
    active = bool(product.get("active"))
    if visibility in ("occluded", "partial", "not_in_frame", "unavailable"):
        return "unknown"
    if not active or visibility == "hidden":
        return "empty"
    if active and visibility == "visible":
        return "full"
    return "unknown"


def _person_blocks(product, characters, roi) -> bool:
    region = product.get("bbox") if valid_box(product.get("bbox")) else roi
    if not region:
        return False
    cx = (region[0] + region[2]) * 0.5
    cy = (region[1] + region[3]) * 0.5
    for person in characters:
        if person.get("class") not in ("shopper", "staff"):
            continue
        if person.get("visibility") not in ("visible", "partial"):
            continue
        box = person.get("bbox")
        if not valid_box(box):
            continue
        if iou(region, box) >= 0.15 or (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            return True
    return False


def _load_predictions(path: Path):
    rows = {}
    forbidden = set()
    crossings = {"entry": 0, "exit": 0}
    door_available = None
    staff_frames = []
    queue_frames = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            _walk_keys(row, forbidden)
            key = (row["camera_id"], int(row["frame_index"]))
            rows[key] = row
            if row["camera_id"] == ENTRANCE_CAMERA:
                door_available = row.get("door_available")
                for crossing in row.get("crossings") or []:
                    kind = crossing.get("kind")
                    if kind in crossings:
                        crossings[kind] += 1
            if row["camera_id"] == CHECKOUT_CAMERA:
                queue = row.get("queue") or {}
                queue_frames[key[1]] = queue.get("total")
                staff = row.get("staff02") or {}
                staff_frames.append((key[1], staff.get("observable"), staff.get("dispatched")))
    return rows, forbidden, crossings, door_available, staff_frames, queue_frames


def _load_events(scenario: str):
    path = DATASET / scenario / "events.jsonl"
    events = []
    if path.stat().st_size == 0:
        return events
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _class_metrics(frame_rows, class_name, threshold):
    positives = 0
    hits = []
    grid = []
    for iou_threshold in IOU_GRID:
        iou_hits = []
        iou_positives = 0
        for frame in frame_rows:
            predictions = [item for item in frame["preds"] if item["class"] == class_name]
            ground_truth = [{"bbox": box, "class": class_name} for box in frame["gts"].get(class_name, [])]
            ignores = [{"bbox": box, "class": class_name} for box in frame["ignores"].get(class_name, [])]
            matched = match_detections(predictions, ground_truth, ignores, iou_threshold)
            iou_hits.extend(matched)
            iou_positives += len(ground_truth)
        ap = average_precision([(score, hit) for score, hit, _ in iou_hits], iou_positives)
        grid.append(ap)
        if iou_threshold == 0.5:
            hits = iou_hits
            positives = iou_positives
    point = pr_at_threshold(hits, positives, threshold)
    finite = [value for value in grid if value is not None]
    return {
        "positives": positives,
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "tp": point["tp"],
        "fp": point["fp"],
        "fn": point["fn"],
        "ap50": grid[0],
        "ap50_95": (sum(finite) / len(finite)) if finite else None,
    }


def _mean_present(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _queue_alert_metrics(gt_counts, pred_totals, pred_alerts):
    frames = sorted(set(gt_counts) | set(pred_totals))
    compared = []
    skipped_unknown = 0
    for frame in frames:
        gt = gt_counts.get(frame)
        pred = pred_totals.get(frame)
        if gt is None:
            skipped_unknown += 1
            continue
        if pred is None:
            continue
        compared.append((frame, gt, pred, pred_alerts.get(frame)))
    if not compared:
        return {
            "available": False,
            "reason": "No checkout frame had both a known visible queue count and a polygon count.",
            "frames_skipped_unknown_gt": skipped_unknown,
        }
    absolute = [abs(pred - gt) for _frame, gt, pred, _alert in compared]
    signed = [pred - gt for _frame, gt, pred, _alert in compared]
    gt_flags = [gt >= 4 for _frame, gt, _pred, _alert in compared]
    sustained = []
    high = low = 0
    latched = False
    for flag in gt_flags:
        if flag:
            high += 1
            low = 0
            latched = latched or high >= 15
        else:
            low += 1
            high = 0
            if low >= 15:
                latched = False
        sustained.append(latched)
    tp = fp = fn = 0
    for flag, (_frame, _gt, _pred, alert) in zip(sustained, compared):
        if alert is None:
            continue
        if alert and flag:
            tp += 1
        elif alert and not flag:
            fp += 1
        elif flag and not alert:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    first_gt = next((frame for frame, gt, _pred, _alert in compared if gt >= 4), None)
    first_pred = next((frame for frame, _gt, _pred, alert in compared if alert), None)
    timing = None
    if first_gt is not None and first_pred is not None:
        timing = round((first_pred - first_gt) / FPS, 3)
    return {
        "available": True,
        "frames_compared": len(compared),
        "frames_skipped_unknown_gt": skipped_unknown,
        "mae": sum(absolute) / len(absolute),
        "mean_signed_error": sum(signed) / len(signed),
        "alert_precision": precision,
        "alert_recall": recall,
        "alert_f1": f1_value(precision, recall),
        "alert_tp_frames": tp,
        "alert_fp_frames": fp,
        "alert_fn_frames": fn,
        "first_gt_count_at_least_4_frame": first_gt,
        "first_predicted_alert_frame": first_pred,
        "alert_timing_error_s": timing,
        "alert_timing_note": "Predicted alert is the 15-frame latch on the summed polygon count. Timing is that frame minus the first known ground-truth frame whose visible queue count is at least 4.",
    }


def score(scenario: str, predictions: Path, timing_path: Path, output: Path) -> dict:
    rows, forbidden, crossings, door_available, staff_frames, _queue_pred = _load_predictions(predictions)
    threshold = None
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        threshold = timing.get("person_score_threshold")
    if threshold is None:
        threshold = json.loads((ROOT / "calibration" / "person_operating.json").read_text(encoding="utf-8"))["score_threshold"]
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    slots = {slot["item_id"]: slot for slot in calibration["slots"]}
    per_camera = {camera: [] for camera in CAMERAS}
    gt_queue = {}
    pred_queue = {}
    pred_alert = {}
    shelf = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "occlusion_false_alarms": 0,
        "occlusion_candidates": 0,
        "by_category": {},
    }
    missing_errors = {"Milk": [], "Packaged Snacks": []}
    active_prev = {}
    refill_transitions = []
    examples = []
    missing_predictions = 0
    annotation_path = DATASET / scenario / "annotations.jsonl"
    with annotation_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            camera_id = record["camera_id"]
            frame = int(record["frame_index"])
            prediction = rows.get((camera_id, frame))
            if prediction is None:
                missing_predictions += 1
                continue
            gts = {name: [] for name in PERSON_CLASSES if name != "background"}
            ignores = {name: [] for name in gts}
            for person in record["characters"]:
                class_name = person.get("class")
                if class_name not in gts:
                    continue
                visibility = person.get("visibility")
                box = person.get("bbox")
                if visibility in ("visible", "partial") and valid_box(box):
                    gts[class_name].append([float(v) for v in box])
                elif visibility in ("occluded", "partial", "unavailable") and valid_box(box):
                    ignores[class_name].append([float(v) for v in box])
            preds = []
            for det in prediction.get("detections") or []:
                if det.get("class") not in gts or not det.get("bbox"):
                    continue
                preds.append({"class": det["class"], "bbox": det["bbox"], "score": float(det["score"])})
            per_camera[camera_id].append({"frame": frame, "preds": preds, "gts": gts, "ignores": ignores})
            kinds = {item["kind"] for item in examples}
            if not {"detection_match", "detection_miss", "detection_false"} <= kinds:
                _consider_detection_example(examples, camera_id, frame, preds, gts, ignores, threshold)
            if camera_id == CHECKOUT_CAMERA:
                unknown = False
                count = 0
                for person in record["characters"]:
                    if person.get("class") != "shopper" or person.get("phase") not in QUEUE_PHASES:
                        continue
                    visibility = person.get("visibility")
                    if visibility in ("occluded", "unavailable"):
                        unknown = True
                    elif visibility in ("visible", "partial") and valid_box(person.get("bbox")):
                        count += 1
                gt_queue[frame] = None if unknown else count
                queue = prediction.get("queue") or {}
                pred_queue[frame] = queue.get("total")
                pred_alert[frame] = queue.get("alert")
                if not any(item["kind"] == "queue_frame" for item in examples) and gt_queue[frame] is not None and pred_queue[frame] is not None:
                    examples.append(
                        {
                            "kind": "queue_frame",
                            "camera_id": camera_id,
                            "frame_index": frame,
                            "gt_count": gt_queue[frame],
                            "pred_count": pred_queue[frame],
                        }
                    )
            facing = [slot for slot in slots.values() if slot["facing_camera"] == camera_id and slot.get("available")]
            if not facing:
                continue
            found = {product["item_id"]: product for product in record["products"]}
            pred_slots = {item["item_id"]: item for item in (prediction.get("shelf_slots") or [])}
            known_by_cat = {}
            for slot in facing:
                product = found.get(slot["item_id"])
                if product is None:
                    continue
                gt_state = _gt_shelf_state(product)
                pred_state = (pred_slots.get(slot["item_id"]) or {}).get("state", "unknown")
                category = slot["category"]
                bucket = shelf["by_category"].setdefault(category, {"tp": 0, "fp": 0, "fn": 0})
                if gt_state == "empty" and pred_state == "empty":
                    shelf["tp"] += 1
                    bucket["tp"] += 1
                    if not any(item["kind"] == "shelf_gap_match" for item in examples):
                        examples.append({"kind": "shelf_gap_match", "camera_id": camera_id, "frame_index": frame, "item_id": slot["item_id"]})
                elif gt_state == "full" and pred_state == "empty":
                    shelf["fp"] += 1
                    bucket["fp"] += 1
                    if not any(item["kind"] == "shelf_false_empty" for item in examples):
                        examples.append({"kind": "shelf_false_empty", "camera_id": camera_id, "frame_index": frame, "item_id": slot["item_id"]})
                elif gt_state == "empty" and pred_state != "empty":
                    shelf["fn"] += 1
                    bucket["fn"] += 1
                blocked = product.get("visibility") in ("occluded", "partial") or _person_blocks(product, record["characters"], slot["roi"])
                if gt_state == "unknown" and blocked:
                    shelf["occlusion_candidates"] += 1
                    if pred_state == "empty":
                        shelf["occlusion_false_alarms"] += 1
                        if not any(item["kind"] == "shelf_occlusion_false_alarm" for item in examples):
                            examples.append(
                                {
                                    "kind": "shelf_occlusion_false_alarm",
                                    "camera_id": camera_id,
                                    "frame_index": frame,
                                    "item_id": slot["item_id"],
                                }
                            )
                if gt_state in ("full", "empty"):
                    known_by_cat.setdefault(category, []).append((gt_state, pred_state))
                previous = active_prev.get(slot["item_id"])
                active = bool(product.get("active"))
                if previous is False and active is True:
                    refill_transitions.append({"item_id": slot["item_id"], "frame_index": frame, "pred_state": pred_state})
                active_prev[slot["item_id"]] = active
            for category, pairs in known_by_cat.items():
                gt_missing = sum(1 for gt_state, _pred in pairs if gt_state == "empty")
                pred_missing = sum(1 for _gt, pred_state in pairs if pred_state == "empty")
                missing_errors[category].append(abs(pred_missing - gt_missing))
    camera_metrics = {}
    pooled = []
    for camera_id, frame_rows in per_camera.items():
        pooled.extend(frame_rows)
        class_metrics = {name: _class_metrics(frame_rows, name, threshold) for name in ("shopper", "staff")}
        camera_metrics[camera_id] = {
            "shopper": class_metrics["shopper"],
            "staff": class_metrics["staff"],
            "map50": _mean_present([class_metrics["shopper"]["ap50"], class_metrics["staff"]["ap50"]]),
            "map50_95": _mean_present([class_metrics["shopper"]["ap50_95"], class_metrics["staff"]["ap50_95"]]),
        }
    pooled_classes = {name: _class_metrics(pooled, name, threshold) for name in ("shopper", "staff")}
    events = _load_events(scenario)
    entries = _count_events(events, "ENTRY")
    exits = _count_events(events, "EXIT")
    shelf_precision = shelf["tp"] / (shelf["tp"] + shelf["fp"]) if (shelf["tp"] + shelf["fp"]) else None
    shelf_recall = shelf["tp"] / (shelf["tp"] + shelf["fn"]) if (shelf["tp"] + shelf["fn"]) else None
    staff_metric = _staff_metric(events, staff_frames)
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else None
    payload = {
        "scenario": scenario,
        "dataset_validator_is_not_model_validation": True,
        "baseline_is_in_the_training_split": scenario == "baseline",
        "held_out_from_fitting": scenario != "baseline",
        "same_cast_and_store_as_training": True,
        "future_combined_final_test": False,
        "prediction_file": str(predictions),
        "forbidden_ground_truth_keys_in_predictions": sorted(forbidden),
        "predictions_include_copied_ground_truth": bool(forbidden),
        "missing_prediction_frames": missing_predictions,
        "operating_score_threshold": threshold,
        "person_detection": {
            "by_camera": camera_metrics,
            "pooled": {
                "shopper": pooled_classes["shopper"],
                "staff": pooled_classes["staff"],
                "map50": _mean_present([pooled_classes["shopper"]["ap50"], pooled_classes["staff"]["ap50"]]),
                "map50_95": _mean_present([pooled_classes["shopper"]["ap50_95"], pooled_classes["staff"]["ap50_95"]]),
            },
            "matching": "Greedy IoU. Visible and partial boxes are positives. Occluded or unavailable boxes are ignored. Tiny boxes are excluded. Staff are not shoppers.",
        },
        "entries": {
            "predicted": crossings["entry"],
            "visual_ground_truth": entries["visual"],
            "count_error": None if door_available is False or entries["visual"] is None else crossings["entry"] - entries["visual"],
            "available": bool(door_available) and entries["visual"] is not None,
            "door_available": door_available,
            "world_events_not_seen_by_entrance_camera": entries["unseen"],
            "unavailable_events_not_scored": entries["unavailable"],
        },
        "exits": {
            "predicted": crossings["exit"],
            "visual_ground_truth": exits["visual"],
            "count_error": None if door_available is False else crossings["exit"] - exits["visual"],
            "available": bool(door_available),
            "world_exits_not_seen_by_entrance_camera": exits["unseen"],
            "unavailable_events_not_scored": exits["unavailable"],
            "note": "A world exit with visual_status other than observed is flagged and is not counted as a model miss or as zero.",
        },
        "queue": _queue_alert_metrics(gt_queue, pred_queue, pred_alert),
        "named_shopper_wait": {
            "available": False,
            "reason": "Track ids are not simulator character ids, so a named service wait cannot be checked.",
        },
        "shelf": {
            "precision": shelf_precision,
            "recall": shelf_recall,
            "f1": f1_value(shelf_precision, shelf_recall),
            "tp": shelf["tp"],
            "fp": shelf["fp"],
            "fn": shelf["fn"],
            "occlusion_false_alarms": shelf["occlusion_false_alarms"],
            "occlusion_candidates": shelf["occlusion_candidates"],
            "occlusion_false_alarm_rate": (
                shelf["occlusion_false_alarms"] / shelf["occlusion_candidates"] if shelf["occlusion_candidates"] else None
            ),
            "by_category": shelf["by_category"],
            "missing_count_mae": {
                category: (sum(errors) / len(errors) if errors else None) for category, errors in missing_errors.items()
            },
            "missing_count_frames": {category: len(errors) for category, errors in missing_errors.items()},
            "refill_recovery": _refill(refill_transitions),
        },
        "staff_02": staff_metric,
        "heatmap": {"available": False, "reason": "Image-to-floor identity across cameras is not reliable enough to draw a heatmap."},
        "live_occupancy": {"available": False, "reason": "The same person is visible on more than one camera, and tracks are not joined."},
        "runtime": timing,
        "examples": examples,
    }
    write_json(output, payload)
    print(
        f"SCORE {scenario} shopper_ap50 {pooled_classes['shopper']['ap50']} shelf_f1 {payload['shelf']['f1']} forbidden {sorted(forbidden)}",
        flush=True,
    )
    return payload


def _consider_detection_example(examples, camera_id, frame, preds, gts, ignores, threshold):
    shopper_gt = [{"bbox": box, "class": "shopper"} for box in gts.get("shopper", [])]
    shopper_ign = [{"bbox": box, "class": "shopper"} for box in ignores.get("shopper", [])]
    shopper_pred = [item for item in preds if item["class"] == "shopper" and item["score"] >= threshold]
    if shopper_gt and not any(item["kind"] == "detection_match" for item in examples):
        matched = match_detections(shopper_pred, shopper_gt, shopper_ign, 0.5)
        if any(hit for _score, hit, _class in matched):
            examples.append({"kind": "detection_match", "camera_id": camera_id, "frame_index": frame})
    if shopper_gt and not any(item["kind"] == "detection_miss" for item in examples):
        matched = match_detections(shopper_pred, shopper_gt, [], 0.5)
        if not any(hit for _score, hit, _class in matched):
            examples.append({"kind": "detection_miss", "camera_id": camera_id, "frame_index": frame})
    if shopper_pred and not any(item["kind"] == "detection_false" for item in examples):
        matched = match_detections(shopper_pred, shopper_gt, shopper_ign, 0.5)
        if any(not hit for _score, hit, _class in matched):
            examples.append({"kind": "detection_false", "camera_id": camera_id, "frame_index": frame})


def _count_events(events, name):
    visual = 0
    unseen = []
    unavailable = []
    for event in events:
        if event.get("event") != name:
            continue
        status = event.get("status")
        visual_status = event.get("visual_status")
        brief = {
            "character_id": event.get("character_id"),
            "frame_index": event.get("frame_index"),
            "status": status,
            "visual_status": visual_status,
            "visual_reason": event.get("visual_reason") or event.get("reason"),
        }
        if status == "observed" and visual_status == "observed":
            visual += 1
        elif status == "observed" and visual_status != "observed":
            unseen.append(brief)
        else:
            unavailable.append(brief)
    return {"visual": visual, "unseen": unseen, "unavailable": unavailable}


def _staff_metric(events, staff_frames):
    dispatch = next((event for event in events if event.get("event") == "STAFF_02_DISPATCH" and event.get("status") == "observed"), None)
    active = next((event for event in events if event.get("event") == "STAFF_02_ACTIVE" and event.get("status") == "observed"), None)
    observable_frames = [frame for frame, observable, _dispatched in staff_frames if observable]
    dispatched_frames = [frame for frame, _observable, dispatched in staff_frames if dispatched is True]
    if not observable_frames:
        return {
            "available": False,
            "reason": "The checkout detector did not hold a stable non-cashier staff track for 15 frames, so dispatch is unobservable.",
            "ground_truth_dispatch_frame": None if dispatch is None else dispatch.get("frame_index"),
            "ground_truth_active_frame": None if active is None else active.get("frame_index"),
        }
    first = min(dispatched_frames) if dispatched_frames else None
    timing = None
    if dispatch is not None and first is not None and dispatch.get("frame_index") is not None:
        timing = round((first - int(dispatch["frame_index"])) / FPS, 3)
    return {
        "available": True,
        "predicted_dispatched": first is not None,
        "first_predicted_frame": first,
        "ground_truth_dispatch_frame": None if dispatch is None else dispatch.get("frame_index"),
        "ground_truth_active_frame": None if active is None else active.get("frame_index"),
        "timing_error_s_vs_dispatch_event": timing,
        "false_dispatch": dispatch is None and first is not None,
        "note": "Dispatch is a non-cashier checkout track leaving the home frozen from the training split. It is not a second staff appearance and it is not read from the character name.",
    }


def _refill(transitions):
    if not transitions:
        return {
            "available": False,
            "reason": "This scenario has no facing item that changes from inactive back to active. Refill recovery is not scored as zero.",
        }
    recovered = sum(1 for item in transitions if item["pred_state"] == "full")
    return {"available": True, "transitions": len(transitions), "predicted_full_on_transition_frame": recovered, "examples": transitions[:8]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.scenario, args.predictions, args.timing, args.output)


if __name__ == "__main__":
    main()
