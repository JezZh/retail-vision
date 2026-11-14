# Final-test inference

The dashboard runs `code/run_final_test.py` against the five-camera test take in `../../datasets/track_k_final_test/final_test/`. It reads video pixels, `weights/person_selected.pth`, `weights/shelf_selected.pth`, and frozen files in `calibration/`. It does not read final-test annotations or events. Predictions are written to `predictions/final_test.jsonl`, with timing beside them; model-derived dashboard payloads are written to `telemetry/final_test_payloads.jsonl` and posted to the local FastAPI server.

This requires CUDA. The resulting footfall counts are camera-visible crossings, and checkout counts are detections inside frozen polygons. Shelf health covers only calibrated milk and chip facings; occluded slots are unknown. Whole-store occupancy, cross-camera positions, dwell and heatmaps are unavailable from this model.
