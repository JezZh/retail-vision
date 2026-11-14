# Baseline dataset audit — 2026-11-10

Source: `datasets/track_k_dataset_v2/baseline`. This is simulator ground truth aligned with recorded videos, not computer-vision inference. The six-frame overlay contact sheet is [`baseline-contact.jpg`](baseline-contact.jpg).

## Package and alignment

- Five H.264 camera videos: each 1280×720, 30 fps, 2,700 frames, 90.0 seconds.
- `annotations.jsonl`: 13,500 rows, exactly one row per camera per frame, frame indices 0–2699, and timestamps 0.0–89.96667 seconds. All checked character and product boxes lie within their image bounds.
- Sampled entrance, checkout, and back-aisle overlays align with visible shoppers and the intact dairy/snack facings. This is a spot check, not a full-frame labeling review.
- Staff_02 remains at the backroom position in the annotations, but part of the character is visible at the far right of the checkout camera. The baseline therefore does not provide a fully hidden backup-staff visual.
- `shelf_regions.json` identifies eight milk item IDs on ShelfBox7 and ten chip item IDs on ShelfBox6. All 18 IDs exist in `metadata.json` under the matching shelf, and all remain active throughout the baseline. The baseline is a negative example for shelf gaps; it cannot validate missing-item detection.
- The root `manifest.json` and `validation.json` are not present yet while the dynamic run is ongoing. Isaac's `scenario_complete` helper could not run in this environment because its module imports `pxr`; the independent frame, timestamp, and video checks above passed.

## Measured baseline events

`events.jsonl` contains 34 records: 16 observed and 18 explicitly unavailable. Only observed records should contribute to readings.

| Reading | Observed result | Evidence limit |
| --- | --- | --- |
| Entry crossings | 8, at 0.47–28.13 seconds | Four entry evidence boxes are clipped to 8–15 px in height; useful as simulator ground truth, weak as visual detector examples. |
| Exit crossings | 1, Shopper_01 at 47.77 seconds | World-coordinate crossing is observed, but the event has `visual_status: unavailable` and no entrance-camera box. |
| Occupancy | 8 shoppers in-store by 30 seconds; 7 at the end | Derived from stable simulator IDs and world positions, not a trained camera tracker. |
| Counter 01 queue | Four waiting by 42.8 seconds, still waiting at the end | The dashboard's queue threshold is four, so the baseline would raise a congestion alert. Counter 02 must remain standby in this take. |
| Service | Shopper_01 starts service at 33.07 seconds, with recorded wait of 0 seconds | Later service records are marked unavailable; do not invent their waits. |
| Shelf gaps | None | Milk and chip targets remain active in the baseline. |

## Integration decisions

1. Keep the baseline separate from live vision measurements. If replayed into the dashboard for review, label it **Isaac ground truth** rather than **Vision live**.
2. Filter `status: unavailable` records. Do not turn them into zero-valued counts or events.
3. Derive positions from one per-frame camera row by stable `character_id`, filtering people outside the store; the same ten characters are repeated in all five camera rows.
4. Treat the four-person baseline queue as a real threshold breach unless the desired baseline scenario or alert policy is changed. Do not dispatch Staff_02 from a baseline alert.
5. For a camera-only entry/exit model, improve exit coverage at the front door or annotate a camera-visible crossing. The world-coordinate exit is suitable as evaluation truth but does not prove a visual exit detector can see it.
6. Filter tiny or clipped boxes before detector training. Across all camera frames, 12,201 of 72,347 character boxes have width or height below 16 px; `occlusion_ratio: -1` is an unknown/sentinel value for character boxes, not a physical occlusion fraction.

## Dashboard integration check

The completed baseline is now the source for **Run reference** when `TRACK_K_DATASET_DIR` points to this dataset. The replay uses the simulator's per-frame stable IDs and observed events and keeps the video clips aligned with frame evidence. An in-process FastAPI run accepted all 901 sampled updates and ended at 8 entries, 1 exit, 7 shoppers in store, 4 waiting at Counter 01, Counter 02 standby, and zero shelf tickets. The running server on port 5173 also completed the 90-second HTTP replay; `/api/runs` reported `complete`, and SSE ended at the same totals. Backend tests and frontend smoke/build checks passed. Browser and camera-model inference checks remain separate; no detection model was evaluated by this replay.
