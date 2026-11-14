"""Train, infer, score, and write the Track K v2 artifacts.

A new five-camera folder is not read here. dynamic_staff is scored after the
weights and thresholds are frozen. Set TRACK_K_RESUME=1 to reuse weights when
person_best.pth, shelf_best.pth, and both operating files already exist.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import build_calibration
from common import (
    DATASET,
    EVAL_SCENARIOS,
    OLD_PTH,
    ROOT,
    USD,
    ensure_dirs,
    file_stat,
    write_json,
)
from finalize import finalize
from infer_scenario import infer
from score import score
from train_person import train as train_person
from train_shelf import train as train_shelf

METADATA = Path("metadata.json")
DEMO = Path("demo_videos")


def _snapshot() -> dict:
    protected = {}
    for path in (OLD_PTH, USD, METADATA, DATASET / "validation.json", DATASET / "manifest.json", DATASET / "shelf_regions.json"):
        protected[str(path)] = file_stat(path)
    demo = []
    for path in sorted(DEMO.iterdir()):
        if path.is_file():
            info = path.stat()
            demo.append((path.name, info.st_size, info.st_mtime_ns))
    protected["demo_videos"] = demo
    return protected


def _assert_unchanged(before: dict) -> None:
    after = _snapshot()
    if after != before:
        changed = [key for key in before if before[key] != after.get(key)]
        raise RuntimeError(f"A protected file changed: {changed}")


def _resume_ready() -> bool:
    if os.environ.get("TRACK_K_RESUME") != "1":
        return False
    needed = [
        ROOT / "weights" / "person_best.pth",
        ROOT / "weights" / "shelf_best.pth",
        ROOT / "calibration" / "person_operating.json",
        ROOT / "calibration" / "shelf_operating.json",
    ]
    return all(path.is_file() for path in needed)


def main() -> None:
    started = time.perf_counter()
    stages = {}
    before = None
    ensure_dirs()
    validation = __import__("json").loads((DATASET / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise SystemExit("Refusing to train because validation.passed is not true.")
    try:
        before = _snapshot()
        mark = time.perf_counter()
        calibration = build_calibration()
        stages["calibrate_s"] = round(time.perf_counter() - mark, 3)
        available = sum(1 for slot in calibration["slots"] if slot.get("available"))
        if available != 18:
            raise RuntimeError(f"Expected 18 shelf slots, found {available}. Training was not started.")
        door = calibration["door"]
        print(
            "CALIBRATION_DETAIL",
            "agree",
            door.get("training_entry_agree"),
            "disagree",
            door.get("training_entry_disagree"),
            "sign",
            door.get("outside_sign"),
            "queue",
            {name: (item.get("available"), round(item.get("training_feet_coverage", 0), 3)) for name, item in calibration["queues"].items()},
            "staff_home",
            calibration["staff"].get("home_px"),
            "dispatch_px",
            calibration["staff"].get("dispatch_distance_px"),
            flush=True,
        )
        if _resume_ready():
            print("RESUME existing weights", flush=True)
            stages["person_train_s"] = 0
            stages["shelf_train_s"] = 0
        else:
            mark = time.perf_counter()
            train_person()
            stages["person_train_s"] = round(time.perf_counter() - mark, 3)
            gc.collect()
            torch.cuda.empty_cache()
            mark = time.perf_counter()
            train_shelf(calibration)
            stages["shelf_train_s"] = round(time.perf_counter() - mark, 3)
            gc.collect()
            torch.cuda.empty_cache()
        shutil.copyfile(ROOT / "weights" / "person_best.pth", ROOT / "weights" / "person_selected.pth")
        shutil.copyfile(ROOT / "weights" / "shelf_best.pth", ROOT / "weights" / "shelf_selected.pth")
        write_json(ROOT / "logs" / "stage_times.json", stages)
        for scenario in EVAL_SCENARIOS:
            mark = time.perf_counter()
            infer(
                DATASET / scenario,
                ROOT / "calibration" / "calibration.json",
                ROOT / "weights" / "person_selected.pth",
                ROOT / "weights" / "shelf_selected.pth",
                ROOT / "predictions" / f"{scenario}.jsonl",
            )
            stages[f"infer_{scenario}_s"] = round(time.perf_counter() - mark, 3)
            gc.collect()
            torch.cuda.empty_cache()
            mark = time.perf_counter()
            score(
                scenario,
                ROOT / "predictions" / f"{scenario}.jsonl",
                ROOT / "predictions" / f"{scenario}_timing.json",
                ROOT / "metrics" / f"{scenario}.json",
            )
            stages[f"score_{scenario}_s"] = round(time.perf_counter() - mark, 3)
            write_json(ROOT / "logs" / "stage_times.json", stages)
        mark = time.perf_counter()
        finalize()
        stages["finalize_s"] = round(time.perf_counter() - mark, 3)
        stages["total_s"] = round(time.perf_counter() - started, 3)
        write_json(ROOT / "logs" / "stage_times.json", stages)
        _assert_unchanged(before)
        print("PIPELINE_OK", flush=True)
    except Exception:
        traceback.print_exc()
        stages["total_s"] = round(time.perf_counter() - started, 3)
        write_json(ROOT / "logs" / "stage_times.json", stages)
        try:
            if before is not None:
                _assert_unchanged(before)
        except Exception as protected_error:
            print(f"PROTECTED_FILE_CHANGED {protected_error}", flush=True)
        print("PIPELINE_FAIL", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
