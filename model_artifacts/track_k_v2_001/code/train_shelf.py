"""Train a 3-class full / empty / unknown classifier on frozen slot crops."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DATASET,
    DEV_STRIDE,
    FRAME_COUNTS,
    ROOT,
    SEED,
    SHELF_CLASSES,
    TRAIN_SCENARIOS,
    fit_cut,
    f1_value,
    iou,
    set_seed,
    valid_box,
    write_json,
)

EPOCHS = 6
BATCH = 64
INPUT = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Deterministic subsampling of nearly identical crops.
SHELF_STRIDE = {"fit": 6, "dev": 8}
MAX_PER_CLASS = 700


def build_shelf(pretrained: bool) -> torch.nn.Module:
    model = resnet18(weights="DEFAULT" if pretrained else None)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(model.fc.in_features, len(SHELF_CLASSES)))
    return model


def _label_for(product, person_boxes):
    roi = product["roi"]
    if any(iou(roi, box) >= 0.15 or _center_inside(roi, box) for box in person_boxes):
        return "unknown"
    visibility = product["visibility"]
    if visibility in ("occluded", "partial", "not_in_frame", "unavailable"):
        return "unknown"
    if not product["active"] or visibility == "hidden":
        return "empty"
    if product["active"] and visibility == "visible":
        return "full"
    return "unknown"


def _center_inside(roi, box) -> bool:
    cx = (roi[0] + roi[2]) * 0.5
    cy = (roi[1] + roi[3]) * 0.5
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def extract_crops(calibration: dict):
    slots = [slot for slot in calibration["slots"] if slot["available"]]
    by_camera = {}
    for slot in slots:
        by_camera.setdefault(slot["facing_camera"], []).append(slot)
    records = []
    for scenario in TRAIN_SCENARIOS:
        n_frames = FRAME_COUNTS[scenario]
        cut = fit_cut(n_frames)
        with (DATASET / scenario / "annotations.jsonl").open(encoding="utf-8") as handle:
            products_by_frame = {}
            for line in handle:
                record = json.loads(line)
                camera_id = record["camera_id"]
                if camera_id not in by_camera:
                    continue
                frame = int(record["frame_index"])
                split = "fit" if frame < cut else "dev"
                stride = SHELF_STRIDE[split]
                if frame % stride:
                    continue
                people = []
                for person in record["characters"]:
                    if person["visibility"] not in ("visible", "partial"):
                        continue
                    if valid_box(person.get("bbox")):
                        people.append(person["bbox"])
                labeled = []
                found = {product["item_id"]: product for product in record["products"]}
                for slot in by_camera[camera_id]:
                    product = found.get(slot["item_id"])
                    if product is None:
                        continue
                    label = _label_for(
                        {
                            "roi": slot["roi"],
                            "visibility": product.get("visibility"),
                            "active": bool(product.get("active")),
                        },
                        people,
                    )
                    labeled.append((slot, label))
                if labeled:
                    products_by_frame.setdefault(camera_id, []).append((frame, split, labeled))
        for camera_id, frames in products_by_frame.items():
            video = DATASET / scenario / f"{camera_id}.mp4"
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(video)
            wanted = {frame: (split, labeled) for frame, split, labeled in frames}
            index = 0
            pending = set(wanted)
            while pending:
                ok, image = capture.read()
                if not ok:
                    break
                if index in pending:
                    split, labeled = wanted[index]
                    for slot, label in labeled:
                        x1, y1, x2, y2 = [int(round(v)) for v in slot["roi"]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
                        if x2 - x1 < 4 or y2 - y1 < 4:
                            continue
                        crop = cv2.resize(image[y1:y2, x1:x2], (INPUT, INPUT), interpolation=cv2.INTER_CUBIC)
                        name = f"{scenario}__{camera_id}__{index:05d}__{slot['item_id']}.jpg"
                        path = ROOT / "cache" / "shelf" / split / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if not path.exists():
                            cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        records.append(
                            {
                                "image": str(path),
                                "label": label,
                                "split": split,
                                "scenario": scenario,
                                "camera_id": camera_id,
                                "frame_index": index,
                                "item_id": slot["item_id"],
                                "category": slot["category"],
                            }
                        )
                    pending.remove(index)
                index += 1
            capture.release()
            if pending:
                raise RuntimeError(f"{video} missing shelf frames {sorted(pending)[:5]}")
    kept = _cap(records)
    write_json(ROOT / "calibration" / "shelf_split.json", {"records": [
        {key: item[key] for key in ("split", "scenario", "camera_id", "frame_index", "item_id", "label", "category")}
        for item in kept
    ]})
    return kept


def _cap(records):
    """Keep each category separately.

    A single cap on "full" filled up with the first milk crops and dropped every
    stocked chip bag, so the ResNet called every chip unknown.
    """
    kept = []
    seen = {split: {} for split in ("fit", "dev")}
    for record in records:
        key = (record.get("category", ""), record["label"])
        count = seen[record["split"]].get(key, 0)
        if count >= MAX_PER_CLASS:
            continue
        seen[record["split"]][key] = count + 1
        kept.append(record)
    print("SHELF_COUNTS", {split: dict(counts) for split, counts in seen.items()}, flush=True)
    return kept


class ShelfCrops(Dataset):
    def __init__(self, records, train: bool):
        self.records = records
        self.train = train
        self.index = {name: i for i, name in enumerate(SHELF_CLASSES)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        bgr = cv2.imread(record["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.train and np.random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
        array = rgb.astype(np.float32) / 255.0
        array = (array - MEAN) / STD
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
        return tensor, self.index[record["label"]]


@torch.no_grad()
def _predict_split(model, records, device):
    model.eval()
    results = []
    loader_records = records
    if not loader_records:
        return results
    batch = []
    metas = []

    def flush():
        if not batch:
            return
        logits = model(torch.stack(batch).to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for meta, prob in zip(metas, probs):
            results.append({**meta, "prob": prob.tolist(), "pred": SHELF_CLASSES[int(prob.argmax())]})
        batch.clear()
        metas.clear()

    for record in loader_records:
        tensor, _ = ShelfCrops([record], train=False)[0]
        batch.append(tensor)
        metas.append(record)
        if len(batch) == 128:
            flush()
    flush()
    model.train()
    return results


def _dev_scores(results):
    # Gap class on slots whose truth is full or empty. Occlusion false alarms are separate.
    known = [row for row in results if row["label"] in ("full", "empty")]
    occluded = [row for row in results if row["label"] == "unknown"]
    tp = sum(1 for row in known if row["label"] == "empty" and row["pred"] == "empty")
    fp = sum(1 for row in known if row["label"] == "full" and row["pred"] == "empty")
    fn = sum(1 for row in known if row["label"] == "empty" and row["pred"] != "empty")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    full_correct = sum(1 for row in known if row["label"] == "full" and row["pred"] == "full")
    full_count = sum(1 for row in known if row["label"] == "full")
    unknown_correct = sum(1 for row in occluded if row["pred"] == "unknown")
    false_alarm = sum(1 for row in occluded if row["pred"] == "empty")
    return {
        "empty_precision": precision,
        "empty_recall": recall,
        "empty_f1": f1_value(precision, recall),
        "full_recall": (full_correct / full_count) if full_count else None,
        "unknown_recall": (unknown_correct / len(occluded)) if occluded else None,
        "occlusion_false_alarm_rate": (false_alarm / len(occluded)) if occluded else None,
        "known": len(known),
        "unknown": len(occluded),
    }


def _select_margin(raw_rows):
    best_margin = 0.0
    best_rank = -1.0
    sweep = []
    for margin in (0.0, 0.40, 0.50, 0.60, 0.70):
        adjusted = []
        for row in raw_rows:
            probs = row["prob"]
            pred = SHELF_CLASSES[int(np.argmax(probs))]
            if max(probs) < margin:
                pred = "unknown"
            adjusted.append({**row, "pred": pred})
        scores = _dev_scores(adjusted)
        f1 = scores["empty_f1"] if scores["empty_f1"] is not None else -1.0
        alarm = scores["occlusion_false_alarm_rate"] if scores["occlusion_false_alarm_rate"] is not None else 1.0
        rank = f1 - 0.25 * alarm
        sweep.append({"min_probability": margin, "rank": rank, **scores})
        if rank > best_rank:
            best_rank = rank
            best_margin = margin
    print(f"SHELF_MARGIN {best_margin}", flush=True)
    return best_margin, sweep


def train(calibration: dict) -> dict:
    set_seed(SEED)
    (ROOT / "logs" / "shelf_epochs.jsonl").write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = extract_crops(calibration)
    fit = [row for row in records if row["split"] == "fit"]
    dev = [row for row in records if row["split"] == "dev"]
    if not fit or not dev:
        raise RuntimeError(f"Shelf split is empty fit={len(fit)} dev={len(dev)}")
    counts = [sum(1 for row in fit if row["label"] == name) for name in SHELF_CLASSES]
    if min(counts) == 0:
        raise RuntimeError(f"Shelf fit split is missing a class: {dict(zip(SHELF_CLASSES, counts))}")
    weights = torch.tensor([sum(counts) / count for count in counts], dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    loader = DataLoader(
        ShelfCrops(fit, train=True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = build_shelf(True).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    history = []
    best_f1 = -1.0
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        steps = 0
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite shelf loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            steps += 1
        dev_rows = _predict_split(model, dev, device)
        scores = _dev_scores(dev_rows)
        f1 = scores["empty_f1"] if scores["empty_f1"] is not None else -1.0
        # Prefer a checkpoint that does not call blocked facings empty when F1 is close.
        alarm = scores["occlusion_false_alarm_rate"] if scores["occlusion_false_alarm_rate"] is not None else 1.0
        rank = f1 - 0.25 * alarm
        row = {"epoch": epoch, "train_loss": total / max(1, steps), "dev": scores, "rank": rank}
        history.append(row)
        _save(model, ROOT / "weights" / "shelf_last.pth", epoch, scores, "last")
        if rank > best_f1:
            best_f1 = rank
            best_epoch = epoch
            _save(model, ROOT / "weights" / "shelf_best.pth", epoch, scores, "best_dev_empty_f1_minus_occlusion_alarm")
        print(f"SHELF_EPOCH_DONE {epoch} loss {row['train_loss']:.4f} empty_f1 {f1:.4f} alarm {alarm:.4f}", flush=True)
        with (ROOT / "logs" / "shelf_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    best_model, _blob = load_shelf(ROOT / "weights" / "shelf_best.pth", device)
    raw_rows = _predict_split(best_model, dev, device)
    margin, margin_sweep = _select_margin(raw_rows)
    elapsed = time.perf_counter() - started
    rationale = {
        "selected_checkpoint": "weights/shelf_best.pth",
        "selected_epoch": best_epoch,
        "selection_metric": "dev empty-class F1 minus 0.25 times the occlusion false-alarm rate",
        "seconds": elapsed,
        "dynamic_staff_used": False,
        "class_counts_fit": dict(zip(SHELF_CLASSES, counts)),
        "min_probability": margin,
        "margin_sweep": margin_sweep,
    }
    write_json(
        ROOT / "calibration" / "shelf_operating.json",
        {"min_probability": margin, "sweep": margin_sweep, "decision": "argmax, then unknown when the top probability is below min_probability"},
    )
    write_json(ROOT / "logs" / "shelf_checkpoint_rationale.json", rationale)
    write_json(ROOT / "logs" / "shelf_history.json", {"seconds": elapsed, "epochs": history, "rationale": rationale})
    print(f"SHELF_TRAIN_DONE seconds {elapsed:.1f} best_epoch {best_epoch}", flush=True)
    return rationale


def _save(model, path, epoch, scores, role):
    torch.save(
        {
            "format": "track_k_v2_shelf_slot",
            "arch": "resnet18",
            "classes": list(SHELF_CLASSES),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "dev": scores,
            "role": role,
            "input_size": INPUT,
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
            "seed": SEED,
        },
        path,
    )


def load_shelf(path: Path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_shelf(False)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob


if __name__ == "__main__":
    from calibrate import build_calibration

    train(build_calibration() if not (ROOT / "calibration" / "calibration.json").exists() else json.loads((ROOT / "calibration" / "calibration.json").read_text()))
