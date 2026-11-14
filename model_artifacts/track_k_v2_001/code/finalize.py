"""Write the manifest, validation report, overlays, and telemetry files."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CAMERAS,
    DATASET,
    EVAL_SCENARIOS,
    FPS,
    HELDOUT_SCENARIO,
    OLD_PTH,
    ROOT,
    SEED,
    TRAIN_SCENARIOS,
    USD,
    sha256_file,
    write_json,
)
from telemetry_adapter import write_mapping_doc, write_payloads

METADATA = Path("metadata.json")


def _fmt(value):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_needed(path: Path, keys: set):
    found = {}
    if not path.is_file() or not keys:
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("camera_id"), row.get("frame_index"))
            if key in keys:
                found[key] = row
            if len(found) == len(keys):
                break
    return found


def _read_video_frames(video: Path, indices: set):
    found = {}
    pending = set(indices)
    capture = cv2.VideoCapture(str(video))
    index = 0
    while pending:
        ok, frame = capture.read()
        if not ok:
            break
        if index in pending:
            found[index] = frame
            pending.remove(index)
        index += 1
    capture.release()
    return found


def _draw(frame, prediction, truth, title):
    canvas = frame.copy()
    if truth:
        for person in truth.get("characters") or []:
            box = person.get("bbox")
            if not box or len(box) != 4:
                continue
            color = (40, 180, 40) if person.get("class") == "shopper" else (200, 180, 40)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if prediction:
        for det in prediction.get("detections") or []:
            if not det.get("tracked"):
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox"]]
            color = (40, 140, 255) if det["class"] == "shopper" else (220, 40, 220)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{det['class']} {det['score']:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, title[:140], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _slot_overlay(canvas, calibration, prediction, truth):
    if not prediction or not prediction.get("shelf_slots"):
        return canvas
    rois = {slot["item_id"]: slot["roi"] for slot in calibration["slots"] if slot.get("roi")}
    gt_state = {}
    if truth:
        for product in truth.get("products") or []:
            visibility = product.get("visibility")
            active = bool(product.get("active"))
            if visibility in ("occluded", "partial", "not_in_frame", "unavailable"):
                state = "unknown"
            elif not active or visibility == "hidden":
                state = "empty"
            elif active and visibility == "visible":
                state = "full"
            else:
                state = "unknown"
            gt_state[product.get("item_id")] = state
    colors = {"full": (40, 180, 40), "empty": (40, 40, 220), "unknown": (160, 160, 160)}
    for slot in prediction["shelf_slots"]:
        roi = rois.get(slot["item_id"])
        if not roi:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colors.get(slot["state"], (255, 255, 255)), 1)
        cv2.putText(
            canvas,
            f"{slot['state'][:1]}/{gt_state.get(slot['item_id'], '?')[:1]}",
            (x1, max(12, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colors.get(slot["state"], (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_overlays(calibration: dict) -> list:
    overlay_dir = ROOT / "overlays"
    clip_dir = overlay_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scenario in EVAL_SCENARIOS:
        metrics_path = ROOT / "metrics" / f"{scenario}.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        examples = metrics.get("examples") or []
        keys = set()
        clip_examples = []
        for example in examples:
            keys.add((example["camera_id"], int(example["frame_index"])))
            if example["kind"] in ("detection_match", "detection_miss", "shelf_gap_match", "shelf_occlusion_false_alarm", "shelf_false_empty") and len(clip_examples) < 2:
                center = int(example["frame_index"])
                window = range(max(0, center - 20), center + 21)
                for frame in window:
                    keys.add((example["camera_id"], frame))
                clip_examples.append(example)
        truth = _load_needed(DATASET / scenario / "annotations.jsonl", keys)
        prediction = _load_needed(ROOT / "predictions" / f"{scenario}.jsonl", keys)
        by_camera = {}
        for camera_id, frame in keys:
            by_camera.setdefault(camera_id, set()).add(frame)
        frames = {}
        for camera_id, indices in by_camera.items():
            video = DATASET / scenario / f"{camera_id}.mp4"
            for frame_index, image in _read_video_frames(video, indices).items():
                frames[(camera_id, frame_index)] = image
        for example in examples:
            key = (example["camera_id"], int(example["frame_index"]))
            image = frames.get(key)
            if image is None:
                continue
            title = f"{scenario} {example['kind']} {example['camera_id']} f{example['frame_index']}"
            canvas = _draw(image, prediction.get(key), truth.get(key), title)
            canvas = _slot_overlay(canvas, calibration, prediction.get(key), truth.get(key))
            destination = overlay_dir / f"{scenario}_{example['kind']}_{example['camera_id']}_{example['frame_index']}.jpg"
            cv2.imwrite(str(destination), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            written.append(str(destination))
        for example in clip_examples:
            center = int(example["frame_index"])
            camera_id = example["camera_id"]
            with tempfile.TemporaryDirectory(prefix="track_k_clip_") as temporary:
                count = 0
                for offset, frame_index in enumerate(range(max(0, center - 20), center + 21)):
                    image = frames.get((camera_id, frame_index))
                    if image is None:
                        continue
                    key = (camera_id, frame_index)
                    canvas = _slot_overlay(
                        _draw(image, prediction.get(key), truth.get(key), f"{scenario} {example['kind']} f{frame_index}"),
                        calibration,
                        prediction.get(key),
                        truth.get(key),
                    )
                    cv2.imwrite(str(Path(temporary) / f"{count:04d}.jpg"), canvas)
                    count += 1
                if count < 2:
                    continue
                destination = clip_dir / f"{scenario}_{example['kind']}_{camera_id}.mp4"
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        "12",
                        "-i",
                        str(Path(temporary) / "%04d.jpg"),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(destination),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    written.append(str(destination))
                else:
                    (clip_dir / f"{scenario}_{example['kind']}_ffmpeg_error.txt").write_text(result.stderr[-2000:], encoding="utf-8")
    write_json(overlay_dir / "index.json", {"files": written})
    return written


def _file_record(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "sha256": sha256_file(path)}


def _versions() -> dict:
    import torchvision

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_requirements(versions: dict) -> None:
    lines = [
        f"torch=={versions['torch']}",
        f"torchvision=={versions['torchvision']}",
        f"opencv-python=={versions['opencv']}",
        f"numpy=={versions['numpy']}",
        "# Install these into the CUDA environment used for training.",
        "# Do not replace that environment's CUDA torch build.",
        f"# python {versions['python']}",
    ]
    (ROOT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ROOT / "logs" / "dependency_versions.json", versions)


def write_manifest(versions: dict) -> dict:
    dataset_files = [
        DATASET / "manifest.json",
        DATASET / "validation.json",
        DATASET / "shelf_regions.json",
        DATASET / "events.jsonl",
        METADATA,
    ]
    for scenario in (*TRAIN_SCENARIOS, HELDOUT_SCENARIO):
        dataset_files.append(DATASET / scenario / "annotations.jsonl")
        dataset_files.append(DATASET / scenario / "events.jsonl")
        for camera_id in CAMERAS:
            dataset_files.append(DATASET / scenario / f"{camera_id}.mp4")
    weight_names = [
        "person_selected.pth",
        "person_best.pth",
        "person_last.pth",
        "shelf_selected.pth",
        "shelf_best.pth",
        "shelf_last.pth",
    ]
    person_rationale = json.loads((ROOT / "logs" / "person_checkpoint_rationale.json").read_text(encoding="utf-8"))
    shelf_rationale = json.loads((ROOT / "logs" / "shelf_checkpoint_rationale.json").read_text(encoding="utf-8"))
    person_op = json.loads((ROOT / "calibration" / "person_operating.json").read_text(encoding="utf-8"))
    shelf_op = json.loads((ROOT / "calibration" / "shelf_operating.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    code_files = sorted((ROOT / "code").glob("*.py"))
    manifest = {
        "format": "track_k_v2_model_manifest",
        "seed": SEED,
        "source_dataset": {
            "path": str(DATASET),
            "dataset_validator_passed": json.loads((DATASET / "validation.json").read_text(encoding="utf-8")).get("passed"),
            "dataset_validator_note": "The dataset validator checks alignment and measured events. It does not validate these models.",
            "files": [_file_record(path) for path in dataset_files],
        },
        "excluded_annotation_tape": {
            "path": str(OLD_PTH),
            "sha256": sha256_file(OLD_PTH),
            "note": "Simulator box counts from an older recorder. Not loaded and not renamed.",
        },
        "protected_stage": {"path": str(USD), "bytes": USD.stat().st_size},
        "weights": {name: _file_record(ROOT / "weights" / name) for name in weight_names},
        "input": {
            "video": "1280x720 BGR decoded to RGB. Person model uses the torchvision resize, short side 320, long side at most 640.",
            "shelf_crop": "Frozen slot ROI, cubic resize to 128x128, ImageNet mean and std.",
            "not_model_inputs": [
                "annotations",
                "events",
                "simulator world positions",
                "item active flags",
                "scenario names",
                "scripted timestamps",
            ],
        },
        "thresholds": {
            "person_score": person_op["score_threshold"],
            "person_score_floor_saved_for_ap": 0.05,
            "shelf_min_probability": shelf_op["min_probability"],
            "track_iou": 0.30,
            "track_max_miss_frames": 10,
            "alert_count": 4,
            "alert_frames": 15,
            "door_max_distance_px": calibration["door"].get("max_distance_px"),
            "door_outside_frames": calibration["door"].get("outside_frames"),
            "door_entry_gap_frames": calibration["door"].get("entry_gap_frames"),
            "clipped_bottom_px": calibration["door"].get("clipped_bottom_px"),
            "clipped_max_height_px": calibration["door"].get("clipped_max_height_px"),
            "clipped_grow_px": calibration["door"].get("clipped_grow_px"),
            "clipped_window_frames": calibration["door"].get("clipped_window_frames"),
            "queue_dwell_frames": calibration.get("queue_dwell_frames"),
            "staff_dispatch_distance_px": calibration["staff"].get("dispatch_distance_px"),
            "staff_dispatch_frames": calibration["staff"].get("dispatch_frames"),
        },
        "classes": {
            "person": ["background", "shopper", "staff"],
            "shelf": ["full", "empty", "unknown"],
        },
        "split": {
            "train_scenarios": list(TRAIN_SCENARIOS),
            "fit_fraction": 0.80,
            "development_frames": "last 20 percent of each training scenario, stride 30",
            "held_out_evaluation": HELDOUT_SCENARIO,
            "future_final_test": "A new combined five-camera take with different people, paths, and event timing. It does not exist. Do not train or tune on it.",
            "baseline_note": "baseline is inside the training scenarios by request. Full-clip baseline scores are not a pure holdout.",
            "person_split_file": "calibration/person_split.json",
            "shelf_split_file": "calibration/shelf_split.json",
            "calibration_file": "calibration/calibration.json",
        },
        "selected_checkpoints": {
            "person": person_rationale,
            "shelf": shelf_rationale,
        },
        "output_schema": {
            "file": "JSONL, one object per camera frame",
            "fields": [
                "source_folder",
                "camera_id",
                "frame_index",
                "timestamp_s",
                "detections[{track_id,class,bbox,score,tracked}]",
                "shelf_slots[{item_id,shelf_id,category,state,score,reason}] or null",
                "queue[{counter_01,counter_02,total,alert}] or null",
                "staff_count",
                "staff02[{observable,dispatched}] or null",
                "crossings[{kind,track_id}]",
            ],
        },
        "code": [_file_record(path) for path in code_files],
        "dependency_versions": versions,
        "inference_command": (
            "python -u "
            f"{ROOT / 'code' / 'infer_scenario.py'} "
            "--scenario-dir NEW_FIVE_CAMERA_FOLDER "
            f"--calibration {ROOT / 'calibration' / 'calibration.json'} "
            f"--person-weights {ROOT / 'weights' / 'person_selected.pth'} "
            f"--shelf-weights {ROOT / 'weights' / 'shelf_selected.pth'} "
            "--output predictions.jsonl"
        ),
        "limitations": [
            "Not certified for the future combined final test. dynamic_staff uses the same cast and the same store.",
            "Baseline metrics are partly in-sample.",
            "Heatmap, live floor occupancy, and cross-camera identity are unavailable.",
            "World exits that the entrance camera did not see are not scored as model misses.",
            "Slot ROIs and the staff home are frozen from this store and these camera poses.",
            "Named shopper waits are unavailable because tracks are not simulator ids.",
            "An entrance count is a tracked crossing of the door line, or a shopper who first appears as feet cut off by the bottom of the entrance frame and then grows into a full body. A thin strip that never grows is not an entry.",
            "The shelf head compares each facing crop with that slot's own stocked and empty pictures from the training clips. The margin was chosen on the development tail. dynamic_staff was scored once after that choice. The earlier ResNet filled its full-class cap with milk and never saw a stocked chip bag.",
            "Checkout polygons still contain shoppers who are not in the queue. dynamic_staff four-person alert precision stays low. Queue dwell was frozen from baseline and was not retuned on dynamic_staff.",
            "Dashboard telemetry was written and not posted.",
        ],
        "ready_for_future_combined_final_test": False,
    }
    write_json(ROOT / "model_manifest.json", manifest)
    return manifest


def _camera_lines(metrics: dict) -> list:
    lines = ["| Camera | Shopper P | Shopper R | Shopper AP50 | Shopper AP50:95 | Staff P | Staff R | Staff AP50 | mAP50 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for camera_id in CAMERAS:
        row = metrics["person_detection"]["by_camera"][camera_id]
        shopper = row["shopper"]
        staff = row["staff"]
        lines.append(
            "| "
            + " | ".join(
                [
                    camera_id,
                    _fmt(shopper["precision"]),
                    _fmt(shopper["recall"]),
                    _fmt(shopper["ap50"]),
                    _fmt(shopper["ap50_95"]),
                    _fmt(staff["precision"]),
                    _fmt(staff["recall"]),
                    _fmt(staff["ap50"]),
                    _fmt(row["map50"]),
                ]
            )
            + " |"
        )
    return lines


def _scenario_section(scenario: str, metrics: dict) -> list:
    person = metrics["person_detection"]["pooled"]
    queue = metrics["queue"]
    shelf = metrics["shelf"]
    runtime = metrics.get("runtime") or {}
    lines = [
        f"## {scenario}",
        "",
        "Baseline scores are not a pure holdout." if scenario == "baseline" else "dynamic_staff was not used to fit weights, thresholds, the door sign, queue polygons, the staff home, or slot boxes. It is still the same cast and the same store.",
        "",
        "### Person detection",
        "",
        *_camera_lines(metrics),
        "",
        f"Pooled shopper precision {_fmt(person['shopper']['precision'])}, recall {_fmt(person['shopper']['recall'])}, AP50 {_fmt(person['shopper']['ap50'])}, AP50:95 {_fmt(person['shopper']['ap50_95'])}.",
        f"Pooled staff precision {_fmt(person['staff']['precision'])}, recall {_fmt(person['staff']['recall'])}, AP50 {_fmt(person['staff']['ap50'])}.",
        f"Pooled mAP50 {_fmt(person['map50'])}, mAP50:95 {_fmt(person['map50_95'])}.",
        "",
        "### Entrance counts",
        "",
        f"Entries predicted {_fmt(metrics['entries']['predicted'])}, visually observed ground truth {_fmt(metrics['entries']['visual_ground_truth'])}, error {_fmt(metrics['entries']['count_error'])}.",
        f"Exits predicted {_fmt(metrics['exits']['predicted'])}, visually observed ground truth {_fmt(metrics['exits']['visual_ground_truth'])}, error {_fmt(metrics['exits']['count_error'])}.",
        f"World exits the entrance camera did not observe: {len(metrics['exits']['world_exits_not_seen_by_entrance_camera'])}.",
    ]
    for event in metrics["exits"]["world_exits_not_seen_by_entrance_camera"]:
        lines.append(f"- {event.get('character_id')} frame {event.get('frame_index')}: {event.get('visual_reason')}")
    lines.extend(
        [
            f"Unavailable exit events, not scored: {len(metrics['exits']['unavailable_events_not_scored'])}.",
            "",
            "### Queue",
            "",
        ]
    )
    if not queue.get("available"):
        lines.append(f"Queue metrics unavailable. {queue.get('reason')}")
    else:
        lines.append(
            f"Count MAE {_fmt(queue['mae'])} over {queue['frames_compared']} frames. Signed error {_fmt(queue['mean_signed_error'])}."
        )
        lines.append(
            f"Four-person alert precision {_fmt(queue['alert_precision'])}, recall {_fmt(queue['alert_recall'])}, timing error {_fmt(queue['alert_timing_error_s'])} seconds."
        )
        lines.append(f"Frames skipped because an occluded queue shopper made the count unknown: {queue['frames_skipped_unknown_gt']}.")
    lines.extend(
        [
            "Named shopper waits are unavailable. Track ids are not Shopper_01.",
            "",
            "### Shelves",
            "",
            f"Empty-slot precision {_fmt(shelf['precision'])}, recall {_fmt(shelf['recall'])}.",
            f"Occlusion false alarms {shelf['occlusion_false_alarms']} of {shelf['occlusion_candidates']} blocked slots, rate {_fmt(shelf['occlusion_false_alarm_rate'])}.",
            f"Milk missing-count MAE {_fmt(shelf['missing_count_mae'].get('Milk'))}. Chip missing-count MAE {_fmt(shelf['missing_count_mae'].get('Packaged Snacks'))}.",
            f"Refill recovery: {_fmt(shelf['refill_recovery'].get('reason') if not shelf['refill_recovery'].get('available') else shelf['refill_recovery'])}.",
            "",
            "### Staff_02",
            "",
        ]
    )
    staff = metrics["staff_02"]
    if not staff.get("available"):
        lines.append(f"Unavailable. {staff.get('reason')}")
    else:
        lines.append(
            f"Predicted dispatched {staff['predicted_dispatched']} at frame {staff['first_predicted_frame']}. "
            f"Ground-truth dispatch frame {staff['ground_truth_dispatch_frame']}, active frame {staff['ground_truth_active_frame']}, "
            f"timing error {_fmt(staff['timing_error_s_vs_dispatch_event'])} seconds. False dispatch {staff['false_dispatch']}."
        )
    lines.extend(
        [
            "",
            "### Runtime",
            "",
            f"GPU {runtime.get('gpu')}. Model FPS {_fmt(runtime.get('model_fps'))}. End-to-end FPS {_fmt(runtime.get('end_to_end_fps'))}. Mean latency {_fmt(runtime.get('latency_ms_mean'))} ms. Peak GPU memory {_fmt(runtime.get('gpu_peak_allocated_mb'))} MB.",
            "",
        ]
    )
    return lines


def write_report(versions: dict, overlays: list) -> None:
    stages = {}
    stage_path = ROOT / "logs" / "stage_times.json"
    if stage_path.is_file():
        stages = json.loads(stage_path.read_text(encoding="utf-8"))
    person_history = json.loads((ROOT / "logs" / "person_history.json").read_text(encoding="utf-8"))
    shelf_history = json.loads((ROOT / "logs" / "shelf_history.json").read_text(encoding="utf-8"))
    lines = [
        "# Track K v2 model evaluation",
        "",
        "The Isaac dataset validator passing does not mean these models passed. The numbers below are the pixel pipeline measured against simulator labels. Labels were not inputs to inference.",
        "",
        "Person weights were not retrained. The shelf head was refit on training-folder slot crops after the ResNet full-class cap was found to have dropped every stocked chip. Door, queue, and staff rules were left as they were. dynamic_staff was not used to choose the shelf margin, the dwell, the door distance, the outside-frame count, or the staff latch.",
        "",
        f"Person training took {_fmt(person_history.get('seconds'))} seconds. Shelf training took {_fmt(shelf_history.get('seconds'))} seconds. Stage seconds: `{json.dumps(stages)}`.",
        f"Interpreter `python`. Torch {versions['torch']}, torchvision {versions['torchvision']}, GPU {versions['gpu']}. Seed {SEED}.",
        "",
        "Train folders were baseline, train_occlusion, train_gap, train_refilled, and train_queue. The checkpoint and thresholds used the last 20 percent of those clips. dynamic_staff was held out of fitting. The future combined five-camera test does not exist and was not used.",
        "",
        "Heatmap, live occupancy, and cross-camera identity are unavailable. They are omitted rather than filled with a floor coordinate.",
        "",
    ]
    failures = []
    for scenario in EVAL_SCENARIOS:
        metrics = json.loads((ROOT / "metrics" / f"{scenario}.json").read_text(encoding="utf-8"))
        lines.extend(_scenario_section(scenario, metrics))
        if metrics.get("predictions_include_copied_ground_truth"):
            failures.append(f"{scenario} predictions contain ground-truth keys {metrics['forbidden_ground_truth_keys_in_predictions']}.")
        if metrics["entries"]["count_error"] is None:
            failures.append(f"{scenario} entry count is unavailable.")
        if not metrics["queue"].get("available"):
            failures.append(f"{scenario} queue metrics are unavailable.")
        if not metrics["shelf"]["refill_recovery"].get("available"):
            failures.append(f"{scenario} refill recovery is unavailable: {metrics['shelf']['refill_recovery'].get('reason')}")
        if not metrics["staff_02"].get("available"):
            failures.append(f"{scenario} Staff_02 dispatch is unavailable: {metrics['staff_02'].get('reason')}")
        unseen = metrics["exits"]["world_exits_not_seen_by_entrance_camera"]
        if unseen:
            failures.append(f"{scenario} has {len(unseen)} world exits that no entrance camera observed. They are flagged, not charged as misses.")
    lines.extend(["## Failures and unavailable metrics", ""])
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- No required metric was left unavailable. Scores can still be poor; the tables above are the measurement.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"Weights: `{ROOT / 'weights'}`.",
            f"Predictions: `{ROOT / 'predictions'}`.",
            f"Metrics: `{ROOT / 'metrics'}`.",
            f"Overlays: {len(overlays)} files under `{ROOT / 'overlays'}`.",
            f"Manifest: `{ROOT / 'model_manifest.json'}`.",
            "",
            "## Inference on a new five-camera folder",
            "",
            "The command reads the five CCTV mp4 files and the frozen calibration. It does not read label or event logs from the new folder. Do not retune thresholds on that folder.",
            "",
            "```",
            "python -u model_artifacts/track_k_v2_001/code/infer_scenario.py \\",
            "  --scenario-dir NEW_FIVE_CAMERA_FOLDER \\",
            "  --calibration model_artifacts/track_k_v2_001/calibration/calibration.json \\",
            "  --person-weights model_artifacts/track_k_v2_001/weights/person_selected.pth \\",
            "  --shelf-weights model_artifacts/track_k_v2_001/weights/shelf_selected.pth \\",
            "  --output /tmp/new_scenario_predictions.jsonl",
            "```",
            "",
            "Telemetry mapping is in `telemetry/OUTPUT_MAPPING.md`. Payloads are stored with `posted: false`. The dashboard was not switched to Vision analysis.",
            "",
            "## Ready for the later combined final test?",
            "",
            "No. dynamic_staff is the same people in the same store. Baseline was used for fitting. The shelf head sees only these 18 facings. The future take, with different people, paths, and event timing, has not been recorded and has not been run.",
            "",
        ]
    )
    (ROOT / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def finalize() -> None:
    versions = _versions()
    write_requirements(versions)
    calibration = json.loads((ROOT / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    write_mapping_doc(ROOT / "telemetry" / "OUTPUT_MAPPING.md")
    for scenario in EVAL_SCENARIOS:
        timing_path = ROOT / "predictions" / f"{scenario}_timing.json"
        timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
        write_payloads(
            DATASET / scenario,
            ROOT / "predictions" / f"{scenario}.jsonl",
            timing,
            ROOT / "telemetry" / f"{scenario}_payloads.jsonl",
        )
    overlays = write_overlays(calibration)
    write_manifest(versions)
    write_report(versions, overlays)
    print(f"FINALIZE_DONE overlays {len(overlays)}", flush=True)


if __name__ == "__main__":
    finalize()
