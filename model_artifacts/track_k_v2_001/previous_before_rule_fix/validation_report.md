# Track K v2 model evaluation

The Isaac dataset validator passing does not mean these models passed. The numbers below are the pixel pipeline measured against simulator labels. Labels were not inputs to inference.

Person training took 149.395 seconds. Shelf training took 8.731 seconds. Stage seconds: `{"calibrate_s": 0.838, "person_train_s": 173.879, "shelf_train_s": 15.694, "infer_baseline_s": 242.635, "score_baseline_s": 10.252, "infer_dynamic_staff_s": 239.503, "score_dynamic_staff_s": 9.514}`.
Interpreter `python`. Torch 2.11.0+cu128, torchvision 0.26.0+cu128, GPU NVIDIA GeForce RTX 3090. Seed 17.

Train folders were baseline, train_occlusion, train_gap, train_refilled, and train_queue. The checkpoint and thresholds used the last 20 percent of those clips. dynamic_staff was held out of fitting. The future combined five-camera test does not exist and was not used.

Heatmap, live occupancy, and cross-camera identity are unavailable. They are omitted rather than filled with a floor coordinate.

## baseline

Baseline scores are not a pure holdout.

### Person detection

| Camera | Shopper P | Shopper R | Shopper AP50 | Shopper AP50:95 | Staff P | Staff R | Staff AP50 | mAP50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCTV_Checkout_Shelves | 0.802 | 0.779 | 0.813 | 0.455 | 0.935 | 0.938 | 0.986 | 0.899 |
| CCTV_Entrance_Aisle | 0.824 | 0.676 | 0.756 | 0.409 | unavailable | unavailable | unavailable | 0.756 |
| CCTV_Entrance_Aisle_End | 0.954 | 0.890 | 0.940 | 0.692 | unavailable | unavailable | unavailable | 0.940 |
| CCTV_Back_Aisle | 0.598 | 0.825 | 0.816 | 0.361 | unavailable | unavailable | unavailable | 0.816 |
| CCTV_Back_Aisle_Left | 0.770 | 0.400 | 0.601 | 0.189 | 0.910 | 0.941 | 0.941 | 0.771 |

Pooled shopper precision 0.779, recall 0.671, AP50 0.755, AP50:95 0.388.
Pooled staff precision 0.926, recall 0.939, AP50 0.969.
Pooled mAP50 0.862, mAP50:95 0.491.

### Entrance counts

Entries predicted 1, visually observed ground truth 8, error -7.
Exits predicted 0, visually observed ground truth 0, error 0.
World exits the entrance camera did not observe: 1.
- Shopper_01 frame 1433: No entrance-camera box within 90 frames of the world crossing.
Unavailable exit events, not scored: 7.

### Queue

Queue metrics unavailable. No checkout frame had both a known visible queue count and a polygon count.
Named shopper waits are unavailable. Track ids are not Shopper_01.

### Shelves

Empty-slot precision unavailable, recall unavailable.
Occlusion false alarms 0 of 2700 blocked slots, rate 0.000.
Milk missing-count MAE 0.000. Chip missing-count MAE 0.000.
Refill recovery: This scenario has no facing item that changes from inactive back to active. Refill recovery is not scored as zero..

### Staff_02

Predicted dispatched False at frame None. Ground-truth dispatch frame None, active frame None, timing error unavailable seconds. False dispatch False.

### Runtime

GPU NVIDIA GeForce RTX 3090. Model FPS 74.247. End-to-end FPS 55.720. Mean latency 17.941 ms. Peak GPU memory 292.300 MB.

## dynamic_staff

dynamic_staff was not used to fit weights, thresholds, the door sign, queue polygons, the staff home, or slot boxes. It is still the same cast and the same store.

### Person detection

| Camera | Shopper P | Shopper R | Shopper AP50 | Shopper AP50:95 | Staff P | Staff R | Staff AP50 | mAP50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCTV_Checkout_Shelves | 0.654 | 0.732 | 0.709 | 0.361 | 0.974 | 0.777 | 0.787 | 0.748 |
| CCTV_Entrance_Aisle | 0.838 | 0.610 | 0.681 | 0.303 | 0.000 | unavailable | unavailable | 0.681 |
| CCTV_Entrance_Aisle_End | 0.941 | 0.689 | 0.704 | 0.429 | unavailable | unavailable | unavailable | 0.704 |
| CCTV_Back_Aisle | 0.654 | 0.686 | 0.723 | 0.294 | unavailable | unavailable | unavailable | 0.723 |
| CCTV_Back_Aisle_Left | 0.603 | 0.247 | 0.375 | 0.107 | 0.929 | 0.970 | 0.992 | 0.684 |

Pooled shopper precision 0.717, recall 0.577, AP50 0.637, AP50:95 0.288.
Pooled staff precision 0.955, recall 0.842, AP50 0.855.
Pooled mAP50 0.746, mAP50:95 0.418.

### Entrance counts

Entries predicted 3, visually observed ground truth 8, error -5.
Exits predicted 0, visually observed ground truth 0, error 0.
World exits the entrance camera did not observe: 5.
- Shopper_01 frame 1434: No entrance-camera box within 90 frames of the world crossing.
- Shopper_02 frame 1902: No entrance-camera box within 90 frames of the world crossing.
- Shopper_04 frame 2087: No entrance-camera box within 90 frames of the world crossing.
- Shopper_03 frame 2219: No entrance-camera box within 90 frames of the world crossing.
- Shopper_05 frame 2409: No entrance-camera box within 90 frames of the world crossing.
Unavailable exit events, not scored: 3.

### Queue

Queue metrics unavailable. No checkout frame had both a known visible queue count and a polygon count.
Named shopper waits are unavailable. Track ids are not Shopper_01.

### Shelves

Empty-slot precision 1.000, recall 0.887.
Occlusion false alarms 0 of 1200 blocked slots, rate 0.000.
Milk missing-count MAE 1.277. Chip missing-count MAE 0.000.
Refill recovery: This scenario has no facing item that changes from inactive back to active. Refill recovery is not scored as zero..

### Staff_02

Predicted dispatched False at frame None. Ground-truth dispatch frame 1500, active frame 1578, timing error unavailable seconds. False dispatch False.

### Runtime

GPU NVIDIA GeForce RTX 3090. Model FPS 75.793. End-to-end FPS 56.456. Mean latency 17.707 ms. Peak GPU memory 292.300 MB.

## Failures and unavailable metrics

- baseline queue metrics are unavailable.
- baseline refill recovery is unavailable: This scenario has no facing item that changes from inactive back to active. Refill recovery is not scored as zero.
- baseline has 1 world exits that no entrance camera observed. They are flagged, not charged as misses.
- dynamic_staff queue metrics are unavailable.
- dynamic_staff refill recovery is unavailable: This scenario has no facing item that changes from inactive back to active. Refill recovery is not scored as zero.
- dynamic_staff has 5 world exits that no entrance camera observed. They are flagged, not charged as misses.

## Artifacts

Weights: `model_artifacts/track_k_v2_001/weights`.
Predictions: `model_artifacts/track_k_v2_001/predictions`.
Metrics: `model_artifacts/track_k_v2_001/metrics`.
Overlays: 11 files under `model_artifacts/track_k_v2_001/overlays`.
Manifest: `model_artifacts/track_k_v2_001/model_manifest.json`.

## Inference on a new five-camera folder

The command reads the five CCTV mp4 files and the frozen calibration. It does not read label or event logs from the new folder. Do not retune thresholds on that folder.

```
python -u model_artifacts/track_k_v2_001/code/infer_scenario.py \
  --scenario-dir NEW_FIVE_CAMERA_FOLDER \
  --calibration model_artifacts/track_k_v2_001/calibration/calibration.json \
  --person-weights model_artifacts/track_k_v2_001/weights/person_selected.pth \
  --shelf-weights model_artifacts/track_k_v2_001/weights/shelf_selected.pth \
  --output /tmp/new_scenario_predictions.jsonl
```

Telemetry mapping is in `telemetry/OUTPUT_MAPPING.md`. Payloads are stored with `posted: false`. The dashboard was not switched to Vision analysis.

## Ready for the later combined final test?

No. dynamic_staff is the same people in the same store. Baseline was used for fitting. The shelf head sees only these 18 facings. The future take, with different people, paths, and event timing, has not been recorded and has not been run.
