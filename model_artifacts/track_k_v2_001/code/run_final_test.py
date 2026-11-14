"""Infer the five camera videos, then stream model-derived readings to Track K."""

import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from common import ROOT
from infer_scenario import infer
from telemetry_adapter import write_payloads


def main():
    source = Path(os.environ["TRACK_K_FINAL_TEST_DIR"])
    output = ROOT / "predictions" / "final_test.jsonl"
    infer(source, ROOT / "calibration" / "calibration.json", ROOT / "weights" / "person_selected.pth", ROOT / "weights" / "shelf_selected.pth", output)
    timing = json.loads(output.with_name("final_test_timing.json").read_text(encoding="utf-8"))
    payloads = ROOT / "telemetry" / "final_test_payloads.jsonl"
    write_payloads(source, output, timing, payloads)
    endpoint = os.environ.get("TRACK_K_TELEMETRY_URL", "http://127.0.0.1:5173/api/telemetry")
    token = os.environ["TRACK_K_INGEST_TOKEN"]
    started = time.monotonic()
    last_alerts = set()
    with payloads.open(encoding="utf-8") as handle:
        for line in handle:
            body = json.loads(line)["body"]
            seconds = body["playback_seconds"]
            alerts = [item for item in body.get("alerts", []) if item["type"] != "STOCK"]
            current_alerts = {(item["type"], item["message"]) for item in alerts}
            body["alerts"] = [item for item in alerts if (item["type"], item["message"]) not in last_alerts]
            last_alerts = current_alerts
            time.sleep(max(0, started + seconds - time.monotonic()))
            request = Request(endpoint, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"Telemetry rejected: HTTP {response.status}")


if __name__ == "__main__":
    main()
