"""Replace shelf_slots in saved predictions using the prototype and the video.

Detections, crossings, queue, and staff fields stay as they are. This file does
not open annotations or events.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATASET, EVAL_SCENARIOS, ROOT
from infer_scenario import _prototype_rows
from shelf_prototype import load_prototype


def _person_boxes(row):
    boxes = []
    for det in row.get("detections") or []:
        if det.get("tracked") and det.get("bbox"):
            boxes.append(det["bbox"])
    return boxes


def refresh(scenario: str, blob, slots_by_camera) -> None:
    source = ROOT / "predictions" / f"{scenario}.jsonl"
    temporary = source.with_suffix(".jsonl.tmp")
    captures = {}
    written = 0
    with source.open(encoding="utf-8") as incoming, temporary.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            row = json.loads(line)
            camera_id = row["camera_id"]
            slots = slots_by_camera.get(camera_id)
            if slots:
                capture = captures.get(camera_id)
                if capture is None:
                    capture = cv2.VideoCapture(str(DATASET / scenario / f"{camera_id}.mp4"))
                    if not capture.isOpened():
                        raise RuntimeError(DATASET / scenario / f"{camera_id}.mp4")
                    captures[camera_id] = capture
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"{scenario} {camera_id} ended before frame {row['frame_index']}")
                if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1 != int(row["frame_index"]):
                    raise RuntimeError(
                        f"{scenario} {camera_id} prediction frame {row['frame_index']} "
                        f"does not match the video read"
                    )
                row["shelf_slots"] = _prototype_rows(frame, slots, _person_boxes(row), blob)
            outgoing.write(json.dumps(row, allow_nan=False) + "\n")
            written += 1
    for capture in captures.values():
        capture.release()
    temporary.replace(source)
    timing_path = ROOT / "predictions" / f"{scenario}_timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        timing["shelf_refit"] = {
            "arch": "per_slot_median_l1",
            "margin": blob["margin"],
            "weights": "weights/shelf_selected.pth",
            "person_weights_unchanged": True,
            "queue_and_door_unchanged": True,
        }
        timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(f"SHELF_REFRESH {scenario} rows {written}", flush=True)


def main() -> None:
    blob = load_prototype(ROOT / "weights" / "shelf_selected.pth")
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    slots_by_camera = {}
    for slot in calibration["slots"]:
        if slot.get("available") and slot.get("roi"):
            slots_by_camera.setdefault(slot["facing_camera"], []).append(slot)
    for scenario in EVAL_SCENARIOS:
        refresh(scenario, blob, slots_by_camera)


if __name__ == "__main__":
    main()
