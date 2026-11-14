"""Run the trained person and shelf models on a five-camera folder.

This file reads video pixels and the frozen calibration only. It does not open
label or event logs. Predictions are detector, tracker, and shelf-head output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ALERT_COUNT,
    ALERT_FRAMES,
    CAMERAS,
    CHECKOUT_CAMERA,
    DATASET,
    DOOR_CROSS_FRAMES,
    ENTRANCE_CAMERA,
    FPS,
    HEIGHT,
    ROOT,
    TRACK_IOU,
    TRACK_MAX_MISS,
    WIDTH,
    center_point,
    foot_point,
    iou,
    line_side,
    point_in_polygon,
    point_segment_distance,
    set_seed,
)
from shelf_prototype import FORMAT as SHELF_PROTOTYPE_FORMAT
from shelf_prototype import crop_for as shelf_crop_for
from shelf_prototype import predict_crop as shelf_predict_crop
from train_person import PERSON_CLASSES, load_person
from train_shelf import INPUT, MEAN, STD, SHELF_CLASSES, load_shelf

SCORE_FLOOR = 0.05


def _segments_intersect(a, b, c, d) -> bool:
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0


def polygons_overlap(poly_a, poly_b) -> bool:
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False
    first = [tuple(point) for point in poly_a]
    second = [tuple(point) for point in poly_b]
    for i, start in enumerate(first):
        end = first[(i + 1) % len(first)]
        for j, other in enumerate(second):
            if _segments_intersect(start, end, other, second[(j + 1) % len(second)]):
                return True
    return point_in_polygon(first[0][0], first[0][1], second) or point_in_polygon(
        second[0][0], second[0][1], first
    )


def queue_usability(calibration: dict) -> dict:
    queues = calibration["queues"]
    overlap = polygons_overlap(queues["counter_01"]["polygon"], queues["counter_02"]["polygon"])
    usable = {}
    for name in ("counter_01", "counter_02"):
        usable[name] = bool(queues[name].get("available"))
    usable["overlap"] = overlap
    usable["dwell_frames"] = int(calibration.get("queue_dwell_frames", 1))
    return usable


class _Tracker:
    def __init__(self):
        self._next_id = 1
        self.tracks = []

    def update(self, detections):
        pairs = []
        for track_index, track in enumerate(self.tracks):
            for det_index, det in enumerate(detections):
                if track["class"] != det["class"]:
                    continue
                overlap = iou(track["bbox"], det["bbox"])
                if overlap >= TRACK_IOU:
                    pairs.append((overlap, track_index, det_index))
        pairs.sort(reverse=True)
        used_tracks, used_dets = set(), set()
        for _overlap, track_index, det_index in pairs:
            if track_index in used_tracks or det_index in used_dets:
                continue
            used_tracks.add(track_index)
            used_dets.add(det_index)
            track = self.tracks[track_index]
            track["bbox"] = detections[det_index]["bbox"]
            track["score"] = detections[det_index]["score"]
            track["miss"] = 0
            detections[det_index]["track_id"] = track["id"]
        for index, track in enumerate(self.tracks):
            if index not in used_tracks:
                track["miss"] += 1
        self.tracks = [track for track in self.tracks if track["miss"] <= TRACK_MAX_MISS]
        for index, det in enumerate(detections):
            if index in used_dets:
                continue
            self.tracks.append(
                {
                    "id": self._next_id,
                    "class": det["class"],
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "miss": 0,
                    "door_sides": [],
                    "inside": None,
                }
            )
            det["track_id"] = self._next_id
            self._next_id += 1


class _Alert:
    def __init__(self):
        self.high = 0
        self.low = 0
        self.latched = False

    def update(self, count):
        if count is None:
            return None
        if count >= ALERT_COUNT:
            self.high += 1
            self.low = 0
            if self.high >= ALERT_FRAMES:
                self.latched = True
        else:
            self.low += 1
            self.high = 0
            if self.low >= ALERT_FRAMES:
                self.latched = False
        return self.latched


class _StaffMotion:
    def __init__(self, rule):
        self.rule = rule
        self.seen = 0
        self.far_run = 0
        self.dispatched = False

    def update(self, staff_tracks):
        if not self.rule.get("available"):
            return {"observable": False, "dispatched": None, "reason": self.rule.get("reason")}
        anchor = self.rule["cashier_anchor_px"]
        radius = float(self.rule["cashier_radius_px"])
        home = self.rule["home_px"]
        limit = float(self.rule["dispatch_distance_px"])
        need = int(self.rule.get("dispatch_frames", ALERT_FRAMES))
        use_foot = self.rule.get("dispatch_point") == "foot"
        far = False
        saw_other = False
        for track in staff_tracks:
            center = center_point(track["bbox"])
            if ((center[0] - anchor[0]) ** 2 + (center[1] - anchor[1]) ** 2) ** 0.5 <= radius:
                continue
            saw_other = True
            point = foot_point(track["bbox"]) if use_foot else center
            distance = ((point[0] - home[0]) ** 2 + (point[1] - home[1]) ** 2) ** 0.5
            if distance > limit:
                far = True
        if saw_other:
            self.seen += 1
            self.far_run = self.far_run + 1 if far else 0
            if self.far_run >= need:
                self.dispatched = True
        else:
            self.far_run = 0
        observable = self.seen >= ALERT_FRAMES
        return {"observable": observable, "dispatched": self.dispatched if observable else None}


class _DoorCounter:
    """Count a shopper who is outside the door line, then inside it.

    The same track can finish the crossing. A new track can finish it when the
    outside track breaks within 12 frames and 140 pixels. The frozen calibration
    sets how many outside frames are required. A second entry within
    entry_gap_frames is the same step after an id break, so it is not counted.
    """

    def __init__(self, door):
        self.door = door
        self.limit = float(door.get("max_distance_px", 90.0))
        self.sign = int(door.get("outside_sign", -1))
        self.need = int(door.get("outside_frames", DOOR_CROSS_FRAMES))
        self.entry_gap = int(door.get("entry_gap_frames", 0))
        self.start = tuple(door.get("a", (0.0, 0.0)))
        self.end = tuple(door.get("b", (0.0, 0.0)))
        self.frame = -1
        self.last_entry = -999
        self.state = {}
        self.used = set()

    def update(self, tracks):
        if not self.door.get("available"):
            return []
        self.frame += 1
        events = []
        pending = [
            item
            for item in self.state.values()
            if item["out_run"] >= self.need
            and self.frame - item["frame"] <= 12
            and item["id"] not in self.used
        ]
        for track in tracks:
            if track.get("miss", 0) != 0 or track.get("class") != "shopper":
                continue
            foot = foot_point(track["bbox"])
            if point_segment_distance(foot, self.start, self.end) > self.limit:
                continue
            side = "out" if line_side(foot, self.start, self.end) * self.sign >= 0 else "in"
            state = self.state.setdefault(
                track["id"],
                {"id": track["id"], "out_run": 0, "inside": False, "frame": -999, "foot": foot},
            )
            previous_out = state["out_run"]
            state["frame"] = self.frame
            state["foot"] = foot
            if side == "out":
                state["out_run"] = previous_out + 1
                continue
            state["out_run"] = 0
            if state["inside"]:
                continue
            donor = None
            if previous_out >= self.need:
                donor = state
            else:
                for item in pending:
                    if item["id"] == track["id"] or item["id"] in self.used:
                        continue
                    dx = foot[0] - item["foot"][0]
                    dy = foot[1] - item["foot"][1]
                    if (dx * dx + dy * dy) ** 0.5 <= 140.0:
                        donor = item
                        break
            if donor is None:
                continue
            state["inside"] = True
            self.used.add(donor["id"])
            self.used.add(track["id"])
            if self.entry_gap and self.frame - self.last_entry <= self.entry_gap:
                continue
            self.last_entry = self.frame
            events.append({"kind": "entry", "track_id": track["id"]})
        return events


class _ClippedEntry:
    """Count a shopper whose feet are cut off by the bottom of the entrance frame.

    The body then grows and moves up into the store. That is the visible part of a
    crossing the door line never sees. A track the door line already counted is skipped.
    """

    def __init__(self, door):
        self.bottom = float(door.get("clipped_bottom_px", 710))
        self.max_height = float(door.get("clipped_max_height_px", 100))
        self.grow = float(door.get("clipped_grow_px", 80))
        self.window = int(door.get("clipped_window_frames", 60))
        self.frame = -1
        self.born = {}
        self.done = set()

    def update(self, tracks, already):
        self.frame += 1
        events = []
        for track in tracks:
            if track.get("miss", 0) != 0 or track.get("class") != "shopper":
                continue
            track_id = track["id"]
            box = track["bbox"]
            height = float(box[3] - box[1])
            bottom = float(box[3])
            if track_id not in self.born:
                self.born[track_id] = {
                    "frame": self.frame,
                    "height": height,
                    "bottom": bottom,
                    "ok": bottom >= self.bottom and height <= self.max_height,
                }
            born = self.born[track_id]
            if track_id in self.done or not born["ok"]:
                continue
            if self.frame - born["frame"] > self.window:
                self.done.add(track_id)
                continue
            grown = height >= born["height"] + self.grow and bottom <= born["bottom"] - self.grow
            if not grown:
                continue
            self.done.add(track_id)
            if track_id in already:
                continue
            events.append({"kind": "entry", "track_id": track_id, "source": "clipped_feet"})
        return events


class _QueueCounter:
    """Count shoppers whose feet stay inside a checkout polygon.

    Overlapping polygons stay usable. A foot inside both is given to the nearer register.
    """

    def __init__(self, calibration, usability):
        self.queues = calibration["queues"]
        self.usability = usability
        self.dwell = int(usability.get("dwell_frames", 1))
        self.state = {}

    def update(self, shopper_tracks):
        counts = {}
        for name in ("counter_01", "counter_02"):
            if not self.usability.get(name):
                counts[name] = {"count": None, "available": False}
            else:
                counts[name] = {"count": 0, "available": True}
        if not self.usability.get("counter_01") and not self.usability.get("counter_02"):
            return {
                "counter_01": counts["counter_01"],
                "counter_02": counts["counter_02"],
                "total": None,
                "alert": None,
                "overlap": self.usability.get("overlap"),
            }
        seen = set()
        for track in shopper_tracks:
            foot = foot_point(track["bbox"])
            hits = []
            for name in ("counter_01", "counter_02"):
                if not self.usability.get(name):
                    continue
                polygon = self.queues[name]["polygon"]
                if not point_in_polygon(foot[0], foot[1], polygon):
                    continue
                anchor = self.queues[name].get("service_anchor_px")
                if not anchor:
                    continue
                distance = (foot[0] - anchor[0]) ** 2 + (foot[1] - anchor[1]) ** 2
                hits.append((distance, name))
            state = self.state.setdefault(track["id"], {"name": None, "run": 0})
            seen.add(track["id"])
            if not hits:
                state["name"] = None
                state["run"] = 0
                continue
            name = min(hits)[1]
            state["run"] = state["run"] + 1 if state["name"] == name else 1
            state["name"] = name
            if state["run"] >= self.dwell:
                counts[name]["count"] += 1
        for track_id, state in self.state.items():
            if track_id not in seen:
                state["name"] = None
                state["run"] = 0
        known = [counts[name]["count"] for name in ("counter_01", "counter_02") if counts[name]["available"]]
        return {
            "counter_01": counts["counter_01"],
            "counter_02": counts["counter_02"],
            "total": sum(known),
            "alert": None,
            "overlap": bool(self.usability.get("overlap")),
        }


def _covers(roi, box) -> bool:
    if iou(roi, box) >= 0.15:
        return True
    cx = (roi[0] + roi[2]) * 0.5
    cy = (roi[1] + roi[3]) * 0.5
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _crop_tensor(frame, roi):
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = cv2.resize(frame[y1:y2, x1:x2], (INPUT, INPUT), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def load_shelf_any(path: Path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") == SHELF_PROTOTYPE_FORMAT:
        return {"kind": "prototype", "blob": blob}
    model, resnet_blob = load_shelf(path, device)
    return {"kind": "resnet", "model": model, "blob": resnet_blob}


def _prototype_rows(frame, slots, person_boxes, blob):
    items = blob["items"]
    margin = float(blob["margin"])
    rows = []
    for slot in slots:
        if not slot.get("available"):
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": "unknown",
                    "score": 0.0,
                    "reason": "roi_unavailable",
                }
            )
            continue
        if any(_covers(slot["roi"], box) for box in person_boxes):
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": "unknown",
                    "score": 0.0,
                    "reason": "person_cover",
                }
            )
            continue
        item = items.get(slot["item_id"])
        if item is None:
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": "unknown",
                    "score": 0.0,
                    "reason": "no_prototype",
                }
            )
            continue
        crop = shelf_crop_for(frame, slot["roi"])
        if crop is not None and slot["category"] == "Packaged Snacks":
            # Chip slot ROIs overlap; compare their centers to avoid neighboring facings.
            crop = crop[8:24, 8:24]
            item = {**item, "full": item["full"][8:24, 8:24], "empty": item["empty"][8:24, 8:24]}
        state, score = shelf_predict_crop(crop, item, margin)
        rows.append(
            {
                "item_id": slot["item_id"],
                "shelf_id": slot["shelf_id"],
                "category": slot["category"],
                "state": state,
                "score": score,
                "reason": "prototype",
            }
        )
    rows.sort(key=lambda row: row["item_id"])
    return rows


def _shelf_rows(frame, slots, person_boxes, model, device, min_probability):
    tensors = []
    metas = []
    rows = []
    for slot in slots:
        if not slot.get("available"):
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": "unknown",
                    "score": 0.0,
                    "reason": "roi_unavailable",
                }
            )
            continue
        if any(_covers(slot["roi"], box) for box in person_boxes):
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": "unknown",
                    "score": 0.0,
                    "reason": "person_cover",
                }
            )
            continue
        tensor = _crop_tensor(frame, slot["roi"])
        if tensor is None:
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": "unknown",
                    "score": 0.0,
                    "reason": "crop_failed",
                }
            )
            continue
        tensors.append(tensor)
        metas.append(slot)
    if tensors:
        logits = model(torch.stack(tensors).to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for slot, prob in zip(metas, probs):
            choice = int(prob.argmax())
            state = SHELF_CLASSES[choice]
            reason = "model"
            if float(prob.max()) < min_probability:
                state = "unknown"
                reason = "low_probability"
            rows.append(
                {
                    "item_id": slot["item_id"],
                    "shelf_id": slot["shelf_id"],
                    "category": slot["category"],
                    "state": state,
                    "score": round(float(prob.max()), 4),
                    "reason": reason,
                }
            )
    rows.sort(key=lambda row: row["item_id"])
    return rows


def _open_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != WIDTH or height != HEIGHT:
        capture.release()
        raise RuntimeError(f"{path} is {width}x{height}; this model expects {WIDTH}x{HEIGHT}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 1:
        fps = float(FPS)
    return capture, fps


def _operating(calibration_path: Path, score_threshold, min_probability):
    folder = calibration_path.parent
    if score_threshold is None:
        person_path = folder / "person_operating.json"
        score_threshold = json.loads(person_path.read_text(encoding="utf-8"))["score_threshold"]
    if min_probability is None:
        shelf_path = folder / "shelf_operating.json"
        min_probability = json.loads(shelf_path.read_text(encoding="utf-8"))["min_probability"]
    return float(score_threshold), float(min_probability)


def infer(scenario_dir: Path, calibration_path: Path, person_weights: Path, shelf_weights: Path, output: Path, score_threshold=None, min_probability=None):
    scenario_dir = scenario_dir.resolve()
    output = output.resolve()
    if DATASET.resolve() in output.parents or output == DATASET.resolve():
        raise RuntimeError("Refusing to write predictions into the source dataset")
    videos = []
    for camera_id in CAMERAS:
        path = scenario_dir / f"{camera_id}.mp4"
        if not path.is_file():
            raise RuntimeError(f"Missing required video {path.name}")
        videos.append((camera_id, path))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    score_threshold, min_probability = _operating(calibration_path, score_threshold, min_probability)
    usability = queue_usability(calibration)
    slots_by_camera = {}
    for slot in calibration["slots"]:
        slots_by_camera.setdefault(slot["facing_camera"], []).append(slot)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Inference requires the CUDA interpreter.")
    person, _person_blob = load_person(person_weights, device)
    shelf = load_shelf_any(shelf_weights, device)
    person.roi_heads.score_thresh = SCORE_FLOOR
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model_seconds = 0.0
    latencies = []
    frames = 0
    started = time.perf_counter()
    source_folder = str(scenario_dir)
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for camera_id, video_path in videos:
            capture, fps = _open_video(video_path)
            tracker = _Tracker()
            alert = _Alert()
            staff_motion = _StaffMotion(calibration["staff"])
            door = calibration["door"]
            door_counter = _DoorCounter(door)
            clipped_entry = _ClippedEntry(door)
            queue_counter = _QueueCounter(calibration, usability)
            index = 0
            try:
                while True:
                    frame_started = time.perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
                    torch.cuda.synchronize()
                    model_started = time.perf_counter()
                    output_row = person([tensor.to(device)])[0]
                    boxes = output_row["boxes"].detach().cpu().numpy()
                    labels = output_row["labels"].detach().cpu().numpy()
                    scores = output_row["scores"].detach().cpu().numpy()
                    high = []
                    low = []
                    for box, label, score in zip(boxes, labels, scores):
                        label = int(label)
                        if label <= 0 or label >= len(PERSON_CLASSES):
                            continue
                        item = {
                            "class": PERSON_CLASSES[label],
                            "bbox": [round(float(v), 2) for v in box],
                            "score": round(float(score), 4),
                        }
                        if item["score"] >= score_threshold:
                            high.append(item)
                        else:
                            low.append(item)
                    tracker.update(high)
                    live = [track for track in tracker.tracks if track["miss"] == 0]
                    person_boxes = [track["bbox"] for track in live]
                    shelf_slots = None
                    if camera_id in slots_by_camera:
                        if shelf["kind"] == "prototype":
                            shelf_slots = _prototype_rows(
                                frame,
                                slots_by_camera[camera_id],
                                person_boxes,
                                shelf["blob"],
                            )
                        else:
                            shelf_slots = _shelf_rows(
                                frame,
                                slots_by_camera[camera_id],
                                person_boxes,
                                shelf["model"],
                                device,
                                min_probability,
                            )
                    torch.cuda.synchronize()
                    model_seconds += time.perf_counter() - model_started
                    crossings = []
                    queue = None
                    staff02 = None
                    if camera_id == ENTRANCE_CAMERA:
                        crossings = door_counter.update(tracker.tracks)
                        crossings.extend(clipped_entry.update(tracker.tracks, door_counter.used))
                    if camera_id == CHECKOUT_CAMERA:
                        shoppers = [track for track in live if track["class"] == "shopper"]
                        queue = queue_counter.update(shoppers)
                        queue["alert"] = alert.update(queue["total"])
                        staff_tracks = [track for track in live if track["class"] == "staff"]
                        staff02 = staff_motion.update(staff_tracks)
                    detections = []
                    for item in high:
                        detections.append({**item, "tracked": True, "track_id": item["track_id"]})
                    for item in low:
                        detections.append({**item, "tracked": False, "track_id": None})
                    detections.sort(key=lambda item: item["score"], reverse=True)
                    record = {
                        "source_folder": source_folder,
                        "camera_id": camera_id,
                        "frame_index": index,
                        "timestamp_s": round(index / fps, 5),
                        "detections": detections,
                        "shelf_slots": shelf_slots,
                        "queue": queue,
                        "staff_count": sum(1 for track in live if track["class"] == "staff"),
                        "staff02": staff02,
                        "crossings": crossings,
                    }
                    if camera_id == ENTRANCE_CAMERA:
                        record["door_available"] = bool(door.get("available"))
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    latencies.append(time.perf_counter() - frame_started)
                    frames += 1
                    index += 1
                    if index % 500 == 0:
                        print(f"INFER {scenario_dir.name} {camera_id} frame {index}", flush=True)
            finally:
                capture.release()
            print(f"INFER_CAMERA_DONE {scenario_dir.name} {camera_id} frames {index}", flush=True)
    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)
    def percentile(fraction):
        if not ordered:
            return None
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[position]

    timing = {
        "source_folder": source_folder,
        "frames": frames,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
        "model_seconds": round(model_seconds, 3),
        "model_fps": round(frames / model_seconds, 3) if model_seconds else None,
        "end_to_end_seconds": round(elapsed, 3),
        "end_to_end_fps": round(frames / elapsed, 3) if elapsed else None,
        "latency_ms_mean": round(1000 * sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_ms_p50": round(1000 * percentile(0.50), 3) if ordered else None,
        "latency_ms_p95": round(1000 * percentile(0.95), 3) if ordered else None,
        "person_score_threshold": score_threshold,
        "shelf_min_probability": min_probability,
        "score_floor_written": SCORE_FLOOR,
        "queue_usability": usability,
    }
    timing_path = output.with_name(output.stem + "_timing.json")
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(
        f"INFER_DONE {scenario_dir.name} frames {frames} model_fps {timing['model_fps']} e2e_fps {timing['end_to_end_fps']}",
        flush=True,
    )
    return timing


def main():
    parser = argparse.ArgumentParser(description="Infer a new five-camera scenario without reading its labels")
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration" / "calibration.json")
    parser.add_argument("--person-weights", type=Path, default=ROOT / "weights" / "person_selected.pth")
    parser.add_argument("--shelf-weights", type=Path, default=ROOT / "weights" / "shelf_selected.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--shelf-min-probability", type=float, default=None)
    args = parser.parse_args()
    infer(
        args.scenario_dir,
        args.calibration,
        args.person_weights,
        args.shelf_weights,
        args.output,
        args.score_threshold,
        args.shelf_min_probability,
    )


if __name__ == "__main__":
    main()
