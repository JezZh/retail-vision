#!/usr/bin/env python3
"""Replay Track K camera-visible annotation counts into the telemetry server.

The checkpoint marketgen_tracking_model.pth is an annotation tape, not a
detector. Each row is one camera at one whole second and stores bounding-box
counts for shopper, staff, shelf_structure, and grocery_item. Rows exist for
the baseline take and the dispatch take. The shelf_deletion videos have no
matching checkpoint rows.

Recorded samples line up with frames 1..2700 at 30 FPS: frame 1 is t=0.0 and
the row is held for the next 29 frames. The tape has no shopper coordinates,
entry crossings, per-counter queue lengths, or shelf-specific item counts.
"""

import os
import sys
import time

import cv2
import requests
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "demo_videos"))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "marketgen_tracking_model.pth")
DEFAULT_ENDPOINT = "http://127.0.0.1:5173/api/telemetry"
FRAME_COUNT = 2700
FRAMES_PER_SECOND = 30
FRAME_PERIOD_SECONDS = 0.033
LABEL_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 2.0

CAMERAS = (
    {"id": "cam_entrance", "file": "CCTV_Entrance_Aisle.mp4", "label": "CCTV_Entrance_Aisle"},
    {"id": "cam_checkout", "file": "CCTV_Checkout_Shelves.mp4", "label": "CCTV_Checkout_Shelves"},
    {"id": "cam_back_aisle", "file": "CCTV_Back_Aisle.mp4", "label": "CCTV_Back_Aisle"},
    {"id": "cam_entrance_end", "file": "CCTV_Entrance_Aisle_End.mp4", "label": "CCTV_Entrance_Aisle_End"},
    {"id": "cam_back_aisle_left", "file": "CCTV_Back_Aisle_Left.mp4", "label": "CCTV_Back_Aisle_Left"},
)

SCENARIOS = {
    "1": {
        "folder": "single_cashier",
        "api_name": "single_cashier",
        "label_name": "baseline",
        "detail": "single_cashier uses the checkpoint scenario named baseline.",
    },
    "2": {
        "folder": "dynamic_staff",
        "api_name": "dynamic_staff",
        "label_name": "dispatch",
        "detail": "dynamic_staff uses the checkpoint scenario named dispatch.",
    },
}


def log(message):
    print(message, flush=True)


def load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint root is {type(payload).__name__}, expected a dictionary.")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Checkpoint samples list is empty.")
    registry = {}
    for row_number, row in enumerate(samples, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Sample {row_number} is not a dictionary.")
        for field in ("scenario", "camera", "t", "counts"):
            if field not in row:
                raise ValueError(f"Sample {row_number} is missing {field}.")
        counts = row["counts"]
        if not isinstance(counts, dict):
            raise ValueError(f"Sample {row_number} counts are not a dictionary.")
        for name in ("shopper", "staff", "shelf_structure", "grocery_item"):
            if name not in counts:
                raise ValueError(f"Sample {row_number} has no {name} count.")
        camera = str(row["camera"]).rstrip("/").split("/")[-1]
        second = int(round(float(row["t"])))
        key = (str(row["scenario"]), camera, second)
        if key in registry:
            raise ValueError(f"Duplicate label row for {key}.")
        registry[key] = {
            "shopper": int(counts["shopper"]),
            "staff": int(counts["staff"]),
            "shelf_structure": int(counts["shelf_structure"]),
            "grocery_item": int(counts["grocery_item"]),
        }
    return payload, registry


def require_timeline(registry, label_name):
    missing = []
    for camera in CAMERAS:
        for second in range(LABEL_SECONDS):
            key = (label_name, camera["label"], second)
            if key not in registry:
                missing.append(key)
                if len(missing) >= 8:
                    break
        if len(missing) >= 8:
            break
    if missing:
        shown = ", ".join(f"{scenario}/{camera}/t={second}" for scenario, camera, second in missing)
        raise ValueError(f"Checkpoint is missing label rows: {shown}")


def counts_at(registry, label_name, second):
    second = min(LABEL_SECONDS - 1, max(0, int(second)))
    return {
        camera["label"]: registry[(label_name, camera["label"], second)]
        for camera in CAMERAS
    }


def build_payload(api_name, camera_counts, feeds_open, playback_seconds=0):
    visible = [row["shopper"] for row in camera_counts.values()]
    headcount = max(visible) if visible else 0
    return {
        "current_scenario": api_name,
        "headcount": headcount,
        "live_occupancy": headcount,
        "camera_feeds": feeds_open,
        "fps": float(FRAMES_PER_SECOND),
        "playback_seconds": playback_seconds,
    }


def read_token():
    token = os.environ.get("TRACK_K_INGEST_TOKEN", "").strip()
    if token:
        return token
    env_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "backend", ".env"))
    if not os.path.isfile(env_path):
        return ""
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("TRACK_K_INGEST_TOKEN="):
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def endpoint_url():
    override = os.environ.get("TRACK_K_TELEMETRY_URL", "").strip()
    return override or DEFAULT_ENDPOINT


def scenario_from_args():
    if len(sys.argv) < 2:
        return None
    requested = sys.argv[1].strip()
    if requested in SCENARIOS:
        return SCENARIOS[requested]
    for choice in SCENARIOS.values():
        if requested in {choice["api_name"], choice["folder"], choice["label_name"]}:
            return choice
    log("Unknown scenario. Use single_cashier or dynamic_staff.")
    raise SystemExit(2)


def prompt_scenario():
    log("Track K ground-truth replay")
    log("  1  single_cashier")
    log("  2  dynamic_staff")
    while True:
        try:
            choice = input("Select scenario [1-3]: ").strip()
        except EOFError:
            log("No selection received.")
            raise SystemExit(1)
        if choice in SCENARIOS:
            return SCENARIOS[choice]
        log("Enter 1 or 2.")


def open_cameras(folder):
    captures = []
    try:
        for camera in CAMERAS:
            path = os.path.join(VIDEO_ROOT, folder, camera["file"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing video: {path}")
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise RuntimeError(f"OpenCV could not open {path}")
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            log(
                f"video {camera['file']} frames={frames} fps={fps:.2f} "
                f"path={os.path.relpath(path, SCRIPT_DIR)}"
            )
            if frames < FRAME_COUNT:
                raise RuntimeError(f"{camera['file']} has {frames} frames, expected {FRAME_COUNT}.")
            captures.append((camera, capture, path))
        return captures
    except Exception:
        for _camera, capture, _path in captures:
            capture.release()
        raise


def advance_cameras(captures):
    feeds = {}
    for camera, capture, path in captures:
        ok, _frame = capture.read()
        feeds[camera["id"]] = bool(ok)
        if not ok:
            log(f"video underrun {os.path.basename(path)} at the current frame")
    return feeds


def post_payload(session, url, token, payload):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    started = time.perf_counter()
    try:
        response = session.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return None, elapsed_ms, str(exc)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    detail = ""
    if response.status_code >= 400:
        detail = response.text.replace("\n", " ")[:240]
    return response.status_code, elapsed_ms, detail


def replay(choice, registry, url, token):
    require_timeline(registry, choice["label_name"])
    captures = open_cameras(choice["folder"])
    session = requests.Session()
    failures = 0
    started = time.monotonic()
    next_tick = started
    log(
        f"pipeline connect {url} token={'present' if token else 'missing'} "
        f"scenario={choice['api_name']} label={choice['label_name']}"
    )
    try:
        for frame_index in range(1, FRAME_COUNT + 1):
            feeds = advance_cameras(captures)
            written = frame_index - 1
            second = min(LABEL_SECONDS - 1, written // FRAMES_PER_SECOND)
            camera_counts = counts_at(registry, choice["label_name"], second)
            payload = build_payload(choice["api_name"], camera_counts, feeds, written / FRAMES_PER_SECOND)
            status, elapsed_ms, detail = post_payload(session, url, token, payload)
            checkout = camera_counts["CCTV_Checkout_Shelves"]
            state = "up" if status == 200 else "down"
            log(
                f"frame={frame_index:04d}/{FRAME_COUNT} scenario={choice['api_name']} "
                f"label={choice['label_name']} t={written / FRAMES_PER_SECOND:05.2f}s "
                f"headcount={payload['headcount']} "
                f"checkout_shoppers={checkout['shopper']} checkout_staff={checkout['staff']} "
                f"link={state} http={status} {elapsed_ms}ms"
            )
            if detail:
                log(f"pipeline response {detail}")
            if status == 200:
                failures = 0
            else:
                failures += 1
                if status in (401, 403, 422) or failures >= 5:
                    log("pipeline stopped after a rejected or unreachable telemetry post.")
                    return 1
            next_tick += FRAME_PERIOD_SECONDS
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif frame_index % FRAMES_PER_SECOND == 0:
                log(f"pipeline cadence behind by {int(-remaining * 1000)}ms at frame {frame_index}")
    finally:
        session.close()
        for _camera, capture, _path in captures:
            capture.release()
    elapsed = time.monotonic() - started
    log(f"pipeline complete frames={FRAME_COUNT} elapsed={elapsed:.1f}s endpoint={url}")
    return 0


def main():
    log(f"videos ../demo_videos -> {VIDEO_ROOT}")
    log(f"labels ./marketgen_tracking_model.pth -> {WEIGHTS_PATH}")
    payload, registry = load_checkpoint(WEIGHTS_PATH)
    classes = payload.get("classes")
    log(
        f"checkpoint format={payload.get('format')} samples={len(payload.get('samples', []))} "
        f"indexed_rows={len(registry)} classes={classes}"
    )
    note = payload.get("note")
    if note:
        log(f"checkpoint note: {note}")
    log("label rows have camera-visible counts only; no entry crossings, queue tracks, or shelf-specific gaps.")
    choice = scenario_from_args()
    if choice is None:
        choice = prompt_scenario()
    log(choice["detail"])
    token = read_token()
    url = endpoint_url()
    if not token:
        log("TRACK_K_INGEST_TOKEN is not set and backend/.env has no ingest token.")
        return 1
    try:
        return replay(choice, registry, url, token)
    except KeyboardInterrupt:
        log("pipeline interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
