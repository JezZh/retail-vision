# Track K telemetry mapping

The final-test Run action reads five video streams, produces model predictions, converts them with this adapter, and posts the resulting bodies to `/api/telemetry`. The final-test annotations and events are never read during inference.

| Model output | Dashboard field | Limit |
| --- | --- | --- |
| Entrance camera crossings | `footfall.in` and `footfall.out` | Camera-visible tracked crossings only. |
| Checkout polygon counts | `queues` and `checkout` | Wait and service times are not measured. |
| Queue and staff signals | `alerts` | Staff image position is not a floor coordinate. |
| Known milk and chip facing states | `shelf_health`, `task_queue` | Unknown or occluded slots do not become missing items. Gap tickets include a source camera frame. |
| Decoded cameras and frame clock | `camera_feeds`, `playback_seconds`, `fps` | Test take is 120 seconds; FPS is processing throughput. |

Whole-store headcount, shopper positions, zone dwell, and floor heatmap are unavailable from these models.
