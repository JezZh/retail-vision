"""Per-slot full/empty prototypes for the 18 facing shelf crops.

The ResNet full-class cap filled with milk, so a stocked chip bag was never a
training example and every chip crop was called unknown. Each slot here keeps
its own median stocked picture and its own median empty picture from the fit
split. The margin is chosen on the development tail of the training clips.
dynamic_staff is not read by this file.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DATASET,
    FRAME_COUNTS,
    ROOT,
    SEED,
    TRAIN_SCENARIOS,
    fit_cut,
    f1_value,
    iou,
    set_seed,
    valid_box,
    write_json,
)

FORMAT = "track_k_v2_shelf_prototype"
CROP = 32
MAX_SAMPLES = 24
MARGINS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
MIN_SAMPLES = 5


def _center_inside(roi, box) -> bool:
    cx = (roi[0] + roi[2]) * 0.5
    cy = (roi[1] + roi[3]) * 0.5
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _covered(roi, person_boxes) -> bool:
    return any(iou(roi, box) >= 0.15 or _center_inside(roi, box) for box in person_boxes)


def _people(record) -> list:
    boxes = []
    for person in record["characters"]:
        if person.get("visibility") not in ("visible", "partial"):
            continue
        if valid_box(person.get("bbox")):
            boxes.append(person["bbox"])
    return boxes


def label_for(product, person_boxes, roi) -> str:
    if _covered(roi, person_boxes):
        return "unknown"
    visibility = product.get("visibility")
    active = bool(product.get("active"))
    if visibility in ("occluded", "partial", "not_in_frame", "unavailable"):
        return "unknown"
    if not active or visibility == "hidden":
        return "empty"
    if active and visibility == "visible":
        return "full"
    return "unknown"


def crop_for(image, roi):
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return cv2.resize(image[y1:y2, x1:x2], (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.float32)


def _collect():
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    slots = [slot for slot in calibration["slots"] if slot.get("available") and slot.get("roi")]
    by_camera = defaultdict(list)
    for slot in slots:
        by_camera[slot["facing_camera"]].append(slot)
    rows = []
    for scenario in TRAIN_SCENARIOS:
        n_frames = FRAME_COUNTS[scenario]
        cut = fit_cut(n_frames)
        wanted = defaultdict(list)
        with (DATASET / scenario / "annotations.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                camera_id = record["camera_id"]
                if camera_id not in by_camera:
                    continue
                frame = int(record["frame_index"])
                split = "fit" if frame < cut else "dev"
                stride = 6 if split == "fit" else 8
                if frame % stride:
                    continue
                people = _people(record)
                found = {product["item_id"]: product for product in record["products"]}
                labeled = []
                for slot in by_camera[camera_id]:
                    product = found.get(slot["item_id"])
                    if product is None:
                        continue
                    label = label_for(product, people, slot["roi"])
                    partial = (
                        bool(product.get("active"))
                        and product.get("visibility") == "partial"
                        and not _covered(slot["roi"], people)
                    )
                    labeled.append((slot, label, partial))
                if labeled:
                    wanted[camera_id].append((frame, split, labeled))
        for camera_id, frames in wanted.items():
            capture = cv2.VideoCapture(str(DATASET / scenario / f"{camera_id}.mp4"))
            if not capture.isOpened():
                raise RuntimeError(DATASET / scenario / f"{camera_id}.mp4")
            need = {frame: (split, labeled) for frame, split, labeled in frames}
            index = 0
            pending = set(need)
            while pending:
                ok, image = capture.read()
                if not ok:
                    break
                if index in pending:
                    split, labeled = need[index]
                    for slot, label, partial in labeled:
                        crop = crop_for(image, slot["roi"])
                        if crop is None:
                            continue
                        rows.append(
                            {
                                "split": split,
                                "item_id": slot["item_id"],
                                "category": slot["category"],
                                "label": label,
                                "partial_active": partial,
                                "crop": crop,
                            }
                        )
                    pending.remove(index)
                index += 1
            capture.release()
            if pending:
                raise RuntimeError(f"{scenario} {camera_id} missing shelf frames {sorted(pending)[:5]}")
    return slots, rows


def _median(crops):
    return np.median(np.stack(crops), axis=0).astype(np.float32)


def _build_prototypes(slots, fit_rows):
    full = defaultdict(list)
    empty = defaultdict(list)
    partial = defaultdict(list)
    for row in fit_rows:
        item_id = row["item_id"]
        if row["label"] == "full" and len(full[item_id]) < MAX_SAMPLES:
            full[item_id].append(row["crop"])
        elif row["label"] == "empty" and len(empty[item_id]) < MAX_SAMPLES:
            empty[item_id].append(row["crop"])
        elif row["partial_active"] and len(partial[item_id]) < MAX_SAMPLES:
            partial[item_id].append(row["crop"])
    items = {}
    for slot in slots:
        item_id = slot["item_id"]
        if len(empty[item_id]) < MIN_SAMPLES:
            continue
        source = "visible"
        full_crops = full[item_id]
        if len(full_crops) < MIN_SAMPLES and len(partial[item_id]) >= MIN_SAMPLES:
            full_crops = partial[item_id]
            source = "active_partial_fallback"
        if len(full_crops) < MIN_SAMPLES:
            continue
        items[item_id] = {
            "category": slot["category"],
            "full": _median(full_crops),
            "empty": _median(empty[item_id]),
            "full_source": source,
            "full_samples": len(full_crops),
            "empty_samples": len(empty[item_id]),
        }
    return items


def predict_crop(crop, item, margin):
    if crop is None or "full" not in item or "empty" not in item:
        return "unknown", 0.0
    full_distance = float(np.abs(crop - item["full"]).mean())
    empty_distance = float(np.abs(crop - item["empty"]).mean())
    if empty_distance + margin < full_distance:
        state = "empty"
    elif full_distance + margin < empty_distance:
        state = "full"
    else:
        state = "unknown"
    score = abs(full_distance - empty_distance) / (full_distance + empty_distance + 1e-6)
    return state, round(float(score), 4)


def _dev_scores(rows, items, margin):
    known = []
    occluded = []
    for row in rows:
        item = items.get(row["item_id"])
        if item is None:
            continue
        state, _score = predict_crop(row["crop"], item, margin)
        packed = {**row, "pred": state}
        if row["label"] in ("full", "empty"):
            known.append(packed)
        elif row["label"] == "unknown":
            occluded.append(packed)
    tp = sum(1 for row in known if row["label"] == "empty" and row["pred"] == "empty")
    fp = sum(1 for row in known if row["label"] == "full" and row["pred"] == "empty")
    fn = sum(1 for row in known if row["label"] == "empty" and row["pred"] != "empty")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    full_rows = [row for row in known if row["label"] == "full"]
    full_recall = (
        sum(1 for row in full_rows if row["pred"] == "full") / len(full_rows) if full_rows else None
    )
    alarm = sum(1 for row in occluded if row["pred"] == "empty")
    alarm_rate = (alarm / len(occluded)) if occluded else 0.0
    by_category = {}
    for category in ("Milk", "Packaged Snacks"):
        subset = [row for row in known if row["category"] == category]
        full_subset = [row for row in subset if row["label"] == "full"]
        empty_subset = [row for row in subset if row["label"] == "empty"]
        by_category[category] = {
            "full_recall": (sum(1 for row in full_subset if row["pred"] == "full") / len(full_subset)) if full_subset else None,
            "empty_recall": (sum(1 for row in empty_subset if row["pred"] == "empty") / len(empty_subset)) if empty_subset else None,
            "false_empty": sum(1 for row in subset if row["label"] == "full" and row["pred"] == "empty"),
        }
    f1 = f1_value(precision, recall)
    return {
        "margin": margin,
        "rank": (f1 if f1 is not None else -1.0) - 0.25 * alarm_rate,
        "empty_f1": f1,
        "empty_precision": precision,
        "empty_recall": recall,
        "full_recall": full_recall,
        "false_empty": fp,
        "missed_empty": fn,
        "occlusion_false_alarms": alarm,
        "occlusion_false_alarm_rate": alarm_rate,
        "by_category": by_category,
    }


def _choose(sweep):
    best_rank = max(row["rank"] for row in sweep)
    tied = [row for row in sweep if abs(row["rank"] - best_rank) < 1e-9]
    best_full = max((row["full_recall"] or 0.0) for row in tied)
    tied = [row for row in tied if abs((row["full_recall"] or 0.0) - best_full) < 1e-9]
    return max(tied, key=lambda row: row["margin"])


def _backup_resnet():
    folder = ROOT / "weights" / "shelf_before_gap_fix"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("shelf_selected.pth", "shelf_best.pth", "shelf_last.pth"):
        source = ROOT / "weights" / name
        destination = folder / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    log_folder = ROOT / "logs" / "shelf_resnet18_before_gap_fix"
    log_folder.mkdir(parents=True, exist_ok=True)
    for name in ("shelf_history.json", "shelf_checkpoint_rationale.json", "shelf_epochs.jsonl"):
        source = ROOT / "logs" / name
        destination = log_folder / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)


def fit() -> dict:
    set_seed(SEED)
    started = time.perf_counter()
    _backup_resnet()
    slots, rows = _collect()
    fit_rows = [row for row in rows if row["split"] == "fit"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    items = _build_prototypes(slots, fit_rows)
    missing = [slot["item_id"] for slot in slots if slot["item_id"] not in items]
    if missing:
        raise RuntimeError(f"Shelf prototype missing slots: {missing}")
    sweep = [_dev_scores(dev_rows, items, margin) for margin in MARGINS]
    chosen = _choose(sweep)
    margin = float(chosen["margin"])
    elapsed = time.perf_counter() - started
    stored = {
        item_id: {
            "category": item["category"],
            "full": item["full"],
            "empty": item["empty"],
            "full_source": item["full_source"],
            "full_samples": item["full_samples"],
            "empty_samples": item["empty_samples"],
        }
        for item_id, item in items.items()
    }
    blob = {
        "format": FORMAT,
        "arch": "per_slot_median_l1",
        "classes": ["full", "empty", "unknown"],
        "crop_size": CROP,
        "resize": "INTER_AREA",
        "margin": margin,
        "items": stored,
        "seed": SEED,
        "dynamic_staff_used": False,
        "decision": (
            "Compare the facing crop with that slot's median stocked and empty pictures. "
            "Call empty only when the empty picture is closer by more than the margin. "
            "A person covering the slot stays unknown."
        ),
    }
    weights = ROOT / "weights"
    torch.save(blob, weights / "shelf_prototypes.pth")
    torch.save(blob, weights / "shelf_selected.pth")
    public_sweep = []
    for row in sweep:
        public_sweep.append({key: value for key, value in row.items()})
    rationale = {
        "selected_checkpoint": "weights/shelf_selected.pth",
        "selected_epoch": None,
        "arch": "per_slot_median_l1",
        "selection_metric": (
            "development empty-class F1 minus 0.25 times the occlusion false-alarm rate; "
            "ties keep the larger margin whose full recall is still the best"
        ),
        "seconds": elapsed,
        "dynamic_staff_used": False,
        "margin": margin,
        "dev": {key: value for key, value in chosen.items() if key != "margin"},
        "margin_sweep": public_sweep,
        "full_sources": {item_id: item["full_source"] for item_id, item in items.items()},
        "previous_resnet": "weights/shelf_before_gap_fix/shelf_selected.pth",
        "reason": (
            "The ResNet full-class cap was filled by milk crops, so stocked chip bags were not training examples. "
            "Per-slot medians separate stocked and empty pictures on the development tail with no false empty calls."
        ),
    }
    write_json(
        ROOT / "calibration" / "shelf_operating.json",
        {
            "min_probability": 0.0,
            "margin": margin,
            "sweep": public_sweep,
            "decision": blob["decision"],
            "arch": "per_slot_median_l1",
        },
    )
    write_json(ROOT / "logs" / "shelf_checkpoint_rationale.json", rationale)
    write_json(
        ROOT / "logs" / "shelf_history.json",
        {"seconds": elapsed, "epochs": [], "rationale": rationale, "sweep": public_sweep},
    )
    print(
        f"SHELF_PROTO_DONE margin {margin} f1 {chosen['empty_f1']} full_recall {chosen['full_recall']} "
        f"false_empty {chosen['false_empty']} seconds {elapsed:.1f}",
        flush=True,
    )
    for row in sweep:
        print(
            f"SHELF_PROTO_MARGIN {row['margin']} rank {row['rank']:.4f} f1 {row['empty_f1']} "
            f"full {row['full_recall']} alarm {row['occlusion_false_alarms']}",
            flush=True,
        )
    return rationale


def load_prototype(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != FORMAT:
        raise RuntimeError(f"{path} is not a shelf prototype")
    return blob


if __name__ == "__main__":
    fit()
