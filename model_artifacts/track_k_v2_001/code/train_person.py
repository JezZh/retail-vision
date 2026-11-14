"""Fine-tune a COCO Faster R-CNN on shopper and staff boxes from the train split."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CAMERAS,
    DATASET,
    DEV_STRIDE,
    ENTRANCE_CAMERA,
    ENTRANCE_DENSE_SECONDS,
    ENTRANCE_DENSE_STRIDE,
    FIT_STRIDE,
    FPS,
    FRAME_COUNTS,
    PERSON_CLASSES,
    ROOT,
    SEED,
    TRAIN_SCENARIOS,
    WIDTH,
    average_precision,
    clip_box,
    fit_cut,
    match_detections,
    pr_at_threshold,
    set_seed,
    valid_box,
    write_json,
)

EPOCHS = 5
BATCH = 4
PERSON_INDEX = {"shopper": 1, "staff": 2}


def build_person(pretrained: bool) -> torch.nn.Module:
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, len(PERSON_CLASSES))
    return model


def _wanted_frames():
    fit, dev = [], []
    for scenario in TRAIN_SCENARIOS:
        n_frames = FRAME_COUNTS[scenario]
        cut = fit_cut(n_frames)
        stride = FIT_STRIDE[scenario]
        for camera_id in CAMERAS:
            for frame in range(0, cut, stride):
                fit.append((scenario, camera_id, frame))
            if camera_id == ENTRANCE_CAMERA:
                dense_end = min(cut, ENTRANCE_DENSE_SECONDS * FPS)
                for frame in range(0, dense_end, ENTRANCE_DENSE_STRIDE):
                    fit.append((scenario, camera_id, frame))
            for frame in range(cut, n_frames, DEV_STRIDE):
                dev.append((scenario, camera_id, frame))
    fit = sorted(set(fit))
    dev = sorted(set(dev))
    return fit, dev


def _collect_labels(frame_keys):
    wanted = set(frame_keys)
    labels = {key: [] for key in wanted}
    by_scenario = {}
    for scenario, camera_id, frame in wanted:
        by_scenario.setdefault(scenario, set()).add((camera_id, frame))
    for scenario, keys in by_scenario.items():
        with (DATASET / scenario / "annotations.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = (record["camera_id"], int(record["frame_index"]))
                if key not in keys:
                    continue
                boxes = []
                for person in record["characters"]:
                    if person["class"] not in PERSON_INDEX:
                        continue
                    if person["visibility"] not in ("visible", "partial"):
                        continue
                    if not valid_box(person.get("bbox")):
                        continue
                    boxes.append((clip_box(person["bbox"]), PERSON_INDEX[person["class"]]))
                labels[(scenario, key[0], key[1])] = boxes
    return labels


def _extract(split_name, frame_keys, labels):
    out_dir = ROOT / "cache" / "person" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    grouped = {}
    for scenario, camera_id, frame in frame_keys:
        grouped.setdefault((scenario, camera_id), []).append(frame)
    for (scenario, camera_id), frames in grouped.items():
        needed = set(frames)
        video = DATASET / scenario / f"{camera_id}.mp4"
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video}")
        index = 0
        while needed:
            ok, frame = capture.read()
            if not ok:
                break
            if index in needed:
                name = f"{scenario}__{camera_id}__{index:05d}.jpg"
                path = out_dir / name
                if not path.exists():
                    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                boxes = labels[(scenario, camera_id, index)]
                records.append(
                    {
                        "image": str(path),
                        "boxes": [box for box, _ in boxes],
                        "labels": [label for _, label in boxes],
                        "scenario": scenario,
                        "camera_id": camera_id,
                        "frame_index": index,
                        "split": split_name,
                    }
                )
                needed.remove(index)
            index += 1
        capture.release()
        if needed:
            raise RuntimeError(f"{video} ended before frames {sorted(needed)[:5]}")
    return records


class PersonFrames(Dataset):
    def __init__(self, records, train: bool):
        self.records = records
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        bgr = cv2.imread(record["image"])
        if bgr is None:
            raise RuntimeError(record["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(record["labels"], dtype=np.int64)
        if self.train and len(boxes) and np.random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            width = rgb.shape[1]
            flipped = boxes.copy()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            boxes = flipped
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        target = {"boxes": torch.from_numpy(np.ascontiguousarray(boxes)), "labels": torch.from_numpy(labels.copy())}
        return tensor, target


def _collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def _evaluate(model, records, device):
    model.eval()
    model.roi_heads.score_thresh = 0.05
    per_class_hits = {1: [], 2: []}
    per_class_pos = {1: 0, 2: 0}
    for record in records:
        bgr = cv2.imread(record["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        output = model([tensor.to(device)])[0]
        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        predictions = [
            {"bbox": boxes[i].tolist(), "class": int(labels[i]), "score": float(scores[i])}
            for i in range(len(scores))
        ]
        ground_truth = [
            {"bbox": box, "class": int(label)}
            for box, label in zip(record["boxes"], record["labels"])
        ]
        for class_id in (1, 2):
            class_pred = [item for item in predictions if item["class"] == class_id]
            class_gt = [item for item in ground_truth if item["class"] == class_id]
            hits = match_detections(class_pred, class_gt, [], 0.5)
            per_class_hits[class_id].extend(hits)
            per_class_pos[class_id] += len(class_gt)
    metrics = {}
    for class_id, name in ((1, "shopper"), (2, "staff")):
        ap = average_precision([(score, hit) for score, hit, _ in per_class_hits[class_id]], per_class_pos[class_id])
        point = pr_at_threshold(per_class_hits[class_id], per_class_pos[class_id], 0.40)
        metrics[name] = {"ap50": ap, "f1_at_0.40": point["f1"], "positives": per_class_pos[class_id]}
    return metrics


@torch.no_grad()
def _select_threshold(model, records, device):
    """Pick the shopper score cutoff on the training temporal tail only."""
    model.eval()
    model.roi_heads.score_thresh = 0.05
    hits = []
    positives = 0
    for record in records:
        bgr = cv2.imread(record["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        output = model([tensor.to(device)])[0]
        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        predictions = [
            {"bbox": boxes[i].tolist(), "class": "shopper", "score": float(scores[i])}
            for i in range(len(scores))
            if int(labels[i]) == 1
        ]
        ground_truth = [
            {"bbox": box, "class": "shopper"}
            for box, label in zip(record["boxes"], record["labels"])
            if int(label) == 1
        ]
        hits.extend(match_detections(predictions, ground_truth, [], 0.5))
        positives += len(ground_truth)
    sweep = []
    best_thr = 0.40
    best_key = (-1.0, -1.0, 0.0)
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        point = pr_at_threshold(hits, positives, threshold)
        sweep.append({"threshold": threshold, **point})
        key = (point["f1"] if point["f1"] is not None else -1.0, point["precision"] if point["precision"] is not None else -1.0, threshold)
        if key > best_key:
            best_key = key
            best_thr = threshold
    print(f"PERSON_THRESHOLD {best_thr} sweep {sweep}", flush=True)
    return best_thr, sweep


def train() -> dict:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Person training requires the CUDA interpreter at python.")
    fit_keys, dev_keys = _wanted_frames()
    labels = _collect_labels(fit_keys + dev_keys)
    print(f"PERSON_EXTRACT fit {len(fit_keys)} dev {len(dev_keys)}", flush=True)
    fit_records = _extract("fit", fit_keys, labels)
    dev_records = _extract("dev", dev_keys, labels)
    write_json(
        ROOT / "calibration" / "person_split.json",
        {
            "seed": SEED,
            "fit_frames": [
                {"scenario": r["scenario"], "camera_id": r["camera_id"], "frame_index": r["frame_index"]}
                for r in fit_records
            ],
            "dev_frames": [
                {"scenario": r["scenario"], "camera_id": r["camera_id"], "frame_index": r["frame_index"]}
                for r in dev_records
            ],
            "note": "Dev frames are the last 20% of each training scenario. dynamic_staff is absent.",
        },
    )
    loader = DataLoader(
        PersonFrames(fit_records, train=True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = build_person(True).to(device)
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("roi_heads") or name.startswith("rpn"):
            head.append(param)
        else:
            backbone.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 2e-5}, {"params": head, "lr": 2e-4}],
        weight_decay=1e-4,
    )
    history = []
    best_ap = -1.0
    best_epoch = 0
    started = time.perf_counter()
    weights_dir = ROOT / "weights"
    (ROOT / "logs" / "person_epochs.jsonl").write_text("", encoding="utf-8")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        steps = 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite person loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            steps += 1
            if steps % 25 == 0:
                print(f"PERSON_EPOCH {epoch} step {steps} loss {total / steps:.4f}", flush=True)
        metrics = _evaluate(model, dev_records, device)
        shopper_ap = metrics["shopper"]["ap50"] if metrics["shopper"]["ap50"] is not None else -1.0
        row = {"epoch": epoch, "train_loss": total / max(1, steps), "dev": metrics}
        history.append(row)
        _save(model, weights_dir / "person_last.pth", epoch, shopper_ap, "last")
        if shopper_ap > best_ap:
            best_ap = shopper_ap
            best_epoch = epoch
            _save(model, weights_dir / "person_best.pth", epoch, shopper_ap, "best_dev_shopper_ap50")
        print(f"PERSON_EPOCH_DONE {epoch} loss {row['train_loss']:.4f} shopper_ap50 {shopper_ap:.4f}", flush=True)
        with (ROOT / "logs" / "person_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    best_model, _ = load_person(weights_dir / "person_best.pth", device)
    threshold, sweep = _select_threshold(best_model, dev_records, device)
    elapsed = time.perf_counter() - started
    rationale = {
        "selected_checkpoint": "weights/person_best.pth",
        "selected_epoch": best_epoch,
        "selection_metric": "shopper AP at IoU 0.50 on the temporal dev tail of the training scenarios",
        "dev_shopper_ap50": best_ap,
        "epochs": EPOCHS,
        "seconds": elapsed,
        "why_not_last": "The last epoch is kept, but the selected weights are the epoch with the highest dev shopper AP50.",
        "dynamic_staff_used": False,
        "operating_threshold": threshold,
        "threshold_sweep": sweep,
    }
    write_json(ROOT / "logs" / "person_checkpoint_rationale.json", rationale)
    write_json(ROOT / "calibration" / "person_operating.json", {"score_threshold": threshold, "sweep": sweep})
    write_json(ROOT / "logs" / "person_history.json", {"seconds": elapsed, "epochs": history, "rationale": rationale})
    print(f"PERSON_TRAIN_DONE seconds {elapsed:.1f} best_epoch {best_epoch} ap {best_ap:.4f}", flush=True)
    return rationale


def _save(model, path: Path, epoch: int, ap: float, role: str) -> None:
    torch.save(
        {
            "format": "track_k_v2_person_detector",
            "arch": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "classes": list(PERSON_CLASSES),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "dev_shopper_ap50": ap,
            "role": role,
            "input": "RGB float image; torchvision resizes the short side to 320 and caps the long side at 640",
            "seed": SEED,
        },
        path,
    )


def load_person(path: Path, device) -> torch.nn.Module:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_person(False)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob


if __name__ == "__main__":
    train()
