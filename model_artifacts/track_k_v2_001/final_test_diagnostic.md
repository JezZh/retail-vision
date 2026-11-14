# Final-test video diagnostic — 2026-11-11

The dashboard run `c20289c60775` completed on the five 120-second camera videos. Runtime inference reads the videos, selected model artifacts, and frozen camera calibration. It does not read `annotations.jsonl`, `events.jsonl`, or `validation.json`. Those files were read separately for this diagnostic comparison.

| Measurement | Isaac annotation | Video pipeline | Assessment |
| --- | ---: | ---: | --- |
| Entries | 8 | 8 | Counts agree; predicted entry times lag by up to 1.6 seconds. |
| Exits | 4 | 4 | Counts agree; predicted exit times differ by up to 0.37 seconds. |
| Counter 01 queue alert | 4 shoppers joined by 47.00 s | Four assigned to Counter 01 at 50.93 s | The model sees four total at that frame. Staff_02 is observed at the backroom position, so the closed second lane is assigned zero. The alert is about 3.9 s after the fourth annotated join. After Staff_02 moves, lane assignment remains uncertain. |
| Staff_02 dispatch | 51.03 s | 52.37 s | Movement signal detected; precise staff floor position and service start are unavailable. |
| Snack slots removed | 4 at 18 s | 1 confirmed from 19 s | Partial detection. Other changed slots remain uncertain. |
| Milk slots removed | 3 at 100 s | 2 confirmed from 100 s | Partial detection. One slot remains uncertain. |

| Removed product slot in Isaac annotation | Model state after removal | Result |
| --- | --- | --- |
| `SM_MilkPacket01_1` | empty | Detected |
| `SM_MilkPacket01_19` | empty | Detected |
| `SM_MilkPacket02_73` | unknown | Missed; not counted as empty |
| `SM_Packet2_3_Chips_21` | empty | Detected |
| `SM_Packet2_3_Chips_6` | unknown | Missed; not counted as empty |
| `SM_Packet2_3_Chips_31` | unknown | Missed; not counted as empty |
| `SM_Packet2_3_Chips_41` | unknown | Missed; not counted as empty |

The final-test annotation contains eight observed `ENTRY` events and four observed `EXIT` events. It also lists four later exits as unavailable, not as four additional observed exits. The video pipeline detects eight entries and four exits; its camera-local track IDs cannot be matched reliably to the simulator's named shoppers, so these are event-count and timing comparisons rather than person-identity matches.

The floor map projects detected foot points through the saved camera matrices onto the shop floor. Its heatmap accumulates those camera-visible positions; nearby detections from different cameras are merged spatially. It is an estimate of visible movement, not a reliable cross-camera identity track. Headcount is the cumulative detected entry count minus exits, assuming the shop starts empty. The run ended with 4 shoppers and a nonzero heatmap.

Cross-camera identity was tested separately and was not enabled. A position-and-time handoff created 104 track IDs for the eight annotated shoppers and matched the wrong shopper often enough to be unsafe for visitor counts. A color-histogram handoff and pretrained ResNet-18 image features also confused similarly dressed shoppers. A color-histogram classifier trained on the current baseline crops reached 86.5% accuracy on final-test crops; fine-tuning the last ResNet-18 layer on 1,127 labeled training crops reached 92.3% on 731 final-test crops. Both classifiers assume the same eight known actors and still misidentify some crops, so neither establishes reliable IDs for new customers. The dashboard therefore keeps camera-local IDs and leaves unique zone visits unavailable. Store-wide identity needs a dedicated person re-identification model, synchronized camera handoff calibration, more identities and outfit variation in training, and a new held-out take scored for ID switches and false merges.

The final-test annotations were inspected while correcting the doorway rule, shelf crop, and adapter. **This take is now a diagnostic set, not an untouched held-out evaluation.** An independent new take is required to measure generalization after these changes. More partial-gap training examples are needed before the dashboard can report all removed milk and snack items accurately; missing counts must not be filled from annotation events at runtime.
