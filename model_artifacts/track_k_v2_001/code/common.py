"""Shared constants, camera projection, and detection scoring.

Inference code may import geometry that is already frozen in calibration.json.
It must not read annotations or events.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 17
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT.parent.parent / "datasets" / "track_k_dataset_v2"
OLD_PTH = Path("isaac_sim_scripts/marketgen_tracking_model.pth")
USD = Path("ConvenienceStore01.usd")
PYTHON = "python"

TRAIN_SCENARIOS = (
    "baseline",
    "train_occlusion",
    "train_gap",
    "train_refilled",
    "train_queue",
)
EVAL_SCENARIOS = ("baseline", "dynamic_staff")
HELDOUT_SCENARIO = "dynamic_staff"
FIT_FRACTION = 0.80

CAMERAS = (
    "CCTV_Checkout_Shelves",
    "CCTV_Entrance_Aisle",
    "CCTV_Entrance_Aisle_End",
    "CCTV_Back_Aisle",
    "CCTV_Back_Aisle_Left",
)
ENTRANCE_CAMERA = "CCTV_Entrance_Aisle"
CHECKOUT_CAMERA = "CCTV_Checkout_Shelves"
WIDTH = 1280
HEIGHT = 720
FPS = 30

PERSON_CLASSES = ("background", "shopper", "staff")
SHELF_CLASSES = ("full", "empty", "unknown")
MIN_SIDE = 12
MIN_AREA = 144
ALERT_FRAMES = 15
ALERT_COUNT = 4
TRACK_IOU = 0.30
TRACK_MAX_MISS = 10
DOOR_CROSS_FRAMES = 3

# Deterministic frame strides. Adjacent frames are not shuffled into a random split.
FIT_STRIDE = {
    "baseline": 30,
    "train_occlusion": 10,
    "train_gap": 5,
    "train_refilled": 4,
    "train_queue": 12,
}
DEV_STRIDE = 30
ENTRANCE_DENSE_SECONDS = 15
ENTRANCE_DENSE_STRIDE = 5

FRAME_COUNTS = {
    "baseline": 2700,
    "dynamic_staff": 2700,
    "train_occlusion": 720,
    "train_gap": 150,
    "train_refilled": 120,
    "train_queue": 1080,
}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dirs() -> None:
    for name in (
        "code",
        "weights",
        "logs",
        "predictions",
        "metrics",
        "overlays",
        "telemetry",
        "cache/person",
        "cache/shelf",
        "calibration",
    ):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2) + "\n", encoding="utf-8")


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def assert_protected_unchanged(before: dict) -> None:
    after = {key: file_stat(Path(key)) for key in before}
    if after != before:
        raise RuntimeError(f"A protected file changed: {after}")


def fit_cut(n_frames: int) -> int:
    return int(n_frames * FIT_FRACTION)


def split_name(frame_index: int, n_frames: int) -> str:
    return "fit" if frame_index < fit_cut(n_frames) else "dev"


def valid_box(box) -> bool:
    if not box or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    if not all(np.isfinite([x1, y1, x2, y2])):
        return False
    width = x2 - x1
    height = y2 - y1
    return width >= MIN_SIDE and height >= MIN_SIDE and width * height >= MIN_AREA


def clip_box(box, width: int = WIDTH, height: int = HEIGHT):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), width - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    return [x1, y1, x2, y2]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area += max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    area -= inter
    return float(inter / area) if area > 0 else 0.0


def foot_point(box):
    return ((box[0] + box[2]) * 0.5, box[3])


def center_point(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def project_world(world_cm, camera: dict, width: int = WIDTH, height: int = HEIGHT):
    """Row-vector USD camera: local = world_h @ inverse(world_matrix).

    The camera looks down local -Z. Image v is flipped so image y grows downward.
    Returns None when the point is behind the camera.
    """
    matrix = np.asarray(camera["world_matrix_row_major"], dtype=np.float64).reshape(4, 4)
    world = np.array([world_cm[0], world_cm[1], world_cm[2], 1.0], dtype=np.float64)
    local = world @ np.linalg.inv(matrix)
    if local[2] >= -1e-6:
        return None
    fx = float(camera["focal_length_mm"]) / float(camera["horizontal_aperture_mm"]) * width
    fy = float(camera["focal_length_mm"]) / float(camera["vertical_aperture_mm"]) * height
    depth = -local[2]
    u = fx * (local[0] / depth) + width * 0.5
    v_up = fy * (local[1] / depth) + height * 0.5
    v = height - v_up
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return float(u), float(v)


def convex_hull(points):
    unique = sorted({(round(float(x), 2), round(float(y), 2)) for x, y in points})
    if len(unique) <= 2:
        return unique

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def dilate_polygon(polygon, pixels: float):
    if len(polygon) < 3:
        return polygon
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    dilated = []
    for x, y in polygon:
        dx, dy = x - cx, y - cy
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        dilated.append((x + pixels * dx / norm, y + pixels * dy / norm))
    return dilated


def point_in_polygon(x: float, y: float, polygon) -> bool:
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def line_side(point, start, end) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def point_segment_distance(point, start, end) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return float(((px - ax) ** 2 + (py - ay) ** 2) ** 0.5)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx, qy = ax + t * dx, ay + t * dy
    return float(((px - qx) ** 2 + (py - qy) ** 2) ** 0.5)


def average_precision(scores_and_hits, n_positive: int):
    """All-point AP. scores_and_hits is a list of (score, is_true_positive)."""
    if n_positive <= 0:
        return None
    ordered = sorted(scores_and_hits, key=lambda item: item[0], reverse=True)
    if not ordered:
        return 0.0
    true_positive = np.cumsum([1.0 if hit else 0.0 for _, hit in ordered])
    false_positive = np.cumsum([0.0 if hit else 1.0 for _, hit in ordered])
    recall = true_positive / float(n_positive)
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-9)
    recall_pad = np.concatenate(([0.0], recall, [1.0]))
    precision_pad = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision_pad) - 1, 0, -1):
        precision_pad[index - 1] = max(precision_pad[index - 1], precision_pad[index])
    changes = np.where(recall_pad[1:] != recall_pad[:-1])[0]
    return float(np.sum((recall_pad[changes + 1] - recall_pad[changes]) * precision_pad[changes + 1]))


def match_detections(predictions, ground_truth, ignores, iou_threshold: float):
    """Greedy score-ordered match. A miss against an ignore box is neither TP nor FP."""
    order = sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True)
    used = set()
    hits = []
    for index in order:
        pred = predictions[index]
        best_i = -1
        best_u = 0.0
        for gt_i, gt in enumerate(ground_truth):
            if gt_i in used or gt["class"] != pred["class"]:
                continue
            overlap = iou(pred["bbox"], gt["bbox"])
            if overlap > best_u:
                best_u = overlap
                best_i = gt_i
        if best_i >= 0 and best_u >= iou_threshold:
            used.add(best_i)
            hits.append((pred["score"], True, pred["class"]))
            continue
        ignored = False
        for box in ignores:
            if box["class"] != pred["class"]:
                continue
            if iou(pred["bbox"], box["bbox"]) >= iou_threshold:
                ignored = True
                break
        if not ignored:
            hits.append((pred["score"], False, pred["class"]))
    return hits


def pr_at_threshold(hits, n_positive: int, threshold: float):
    selected = [hit for hit in hits if hit[0] >= threshold]
    tp = sum(1 for _, hit, _ in selected if hit)
    fp = sum(1 for _, hit, _ in selected if not hit)
    fn = max(0, n_positive - tp)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def f1_value(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)
