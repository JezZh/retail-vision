# Vision-model status — 2026-11-10

The dashboard's **Run reference** path reads simulator annotations from `track_k_dataset_v2/baseline/annotations.jsonl` and `events.jsonl`. The legacy **Replay counts** path reads `marketgen_tracking_model.pth`; `vision_engine.py` advances video frames but does not pass their pixels into a detector. The `.pth` file is a 67 KB dictionary with 900 per-second annotation samples and no learned detector tensor weights. Neither path currently measures its dashboard readings by analyzing video pixels.

A separate local `retail-intelligence` project has a pretrained `yolo11n.pt` person detector and ByteTrack worker. It is not connected to this dashboard. To check model feasibility, the local weights were run on five sampled frames each from the completed Isaac baseline's entrance and checkout cameras at 0, 14, 30, 43, and 60 seconds. At confidence 0.15 and image size 640, predicted person boxes were greedily matched to visible, non-tiny Isaac character boxes at IoU ≥ 0.5:

| Camera | Reference people across sampled frames | Predicted boxes | Matched boxes |
| --- | ---: | ---: | ---: |
| Checkout shelves | 34 | 29 | 25 |
| Entrance aisle | 20 | 14 | 8 |

This ten-frame smoke check includes staff and shoppers and is too small to claim accuracy. It shows the pretrained detector can find checkout people but misses many entrance-view characters. It does not establish unique cross-camera IDs, entry/exit counts, queue membership, or shelf-gap detection. The local shelf monitor uses image differencing against an empty reference; it is not calibrated to the Track K cameras and cannot count named missing milk or chip products yet.

The next meaningful test is a separate **Analyze video** run: process each of the five clips with a detector and camera-scoped tracker, calibrate the entry line and queue/shelf regions, derive events from detections, and compare predictions with the held-out Isaac annotations. Keep simulator truth and model predictions as separate sources in the dashboard. The completed dynamic-staff package is needed to evaluate backup dispatch and milk/chip gaps; at this check its events and validation files were still absent.
