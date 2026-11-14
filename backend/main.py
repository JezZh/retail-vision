import asyncio
import base64
import json
import math
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal


from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT_DIR = Path(__file__).resolve().parent.parent
SCENE_DIR = Path(os.environ.get(
    "TRACK_K_SCENE_DIR",
    str(ROOT_DIR),
))
DIST_DIR = ROOT_DIR / "frontend" / "dist"
FINAL_TEST_DIR = ROOT_DIR / "datasets" / "track_k_final_test" / "final_test"
QUEUE_THRESHOLD = 4
MAX_EVENTS = 100
MAX_HISTORY = 120
USE_SIMULATED_DATA = os.environ.get("TRACK_K_USE_SIMULATED_DATA", "true").lower() in {"1", "true", "yes"}
AUTOPLAY = os.environ.get("TRACK_K_AUTOPLAY", "false").lower() in {"1", "true", "yes"}
REPLAY_LOG = Path(os.environ.get("TRACK_K_REPLAY_LOG", "/tmp/track-k-replay.log"))
TIMELINE_SECONDS = 90
TEST_SECONDS = 120
ZONE_CAPACITY = {"entry": 8, "lane1": 5, "lane2": 5, "shelf1": 8, "shelf2": 8, "shelf3": 6, "shelf4": 8, "wall_boxed": 8, "wall_cleaning": 6, "wall_biscuits": 6}
HEATMAP_SIZE = 12
VIDEO_DIR = Path(__file__).resolve().parent / "uploads" / "videos"
CALIBRATION_FILE = Path(__file__).resolve().parent / "data" / "calibrations.json"
RUN_FILE = Path(__file__).resolve().parent / "data" / "runs.json"
LATEST_REPORT_FILE = Path(__file__).resolve().parent / "data" / "latest_report.json"
VIDEO_LIMIT_BYTES = 500 * 1024 * 1024
VIDEO_SCENARIOS = {"final_test": "Final test · five camera angles"}
ELECTRONICS_MODEL = ROOT_DIR / "assets" / "electronics" / "ElectronicsShop_2F.glb"
SHELF_CATALOG = {
    "Dairy - 01": {"zone_id": "shelf1", "category": "Dairy", "location": "Aisle_1_Tier_2", "max_capacity": 40},
    "Snacks - 02": {"zone_id": "shelf2", "category": "Snacks", "location": "Aisle_2_Tier_1", "max_capacity": 48},
    "Canned foods - 03": {"zone_id": "shelf4", "category": "CannedFoods", "location": "Aisle_3_Tier_1", "max_capacity": 36},
    "Beverages - wall": {"zone_id": "shelf3", "category": "Beverages", "location": "Beverage_Wall", "max_capacity": 60},
}
VIDEO_CAMERAS = {
    "cam_entrance": ("Entrance aisle", "CCTV_Entrance_Aisle.mp4"),
    "cam_checkout": ("Checkout shelves", "CCTV_Checkout_Shelves.mp4"),
    "cam_back_aisle": ("Back aisle", "CCTV_Back_Aisle.mp4"),
    "cam_entrance_end": ("Entrance aisle end", "CCTV_Entrance_Aisle_End.mp4"),
    "cam_back_aisle_left": ("Back aisle left", "CCTV_Back_Aisle_Left.mp4"),
}
CALIBRATION_REGIONS = {"entry", "lane1", "lane2", "shelf1", "shelf2", "shelf3", "shelf4", "wall_boxed", "wall_cleaning", "wall_biscuits"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Footfall(StrictModel):
    in_count: int = Field(alias="in", ge=0)
    out_count: int = Field(alias="out", ge=0)


class EvidenceIn(StrictModel):
    camera_id: str
    video_seconds: float = Field(ge=0, le=86400, allow_inf_nan=False)
    frame_index: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def known_camera(self):
        if self.camera_id not in VIDEO_CAMERAS:
            raise ValueError("Evidence camera is not configured")
        return self


class TaskIn(StrictModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    item: str = Field(min_length=1, max_length=120)
    category: Literal["Dairy", "Snacks", "Beverages", "CannedFoods", "Boxed", "CleaningSupplies", "Biscuits"]
    location: str = Field(min_length=1, max_length=120)
    missing: int = Field(ge=1, le=10000)
    evidence: EvidenceIn | None = None


class ShopperIn(StrictModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    zone: str | None = None


class PositionIn(StrictModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class QueueCustomerIn(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    wait_seconds: float = Field(ge=0, allow_inf_nan=False)


class CounterIn(StrictModel):
    queue_count: int | None = Field(default=None, ge=0)
    status: Literal["active", "standby", "dispatching"] | None = None
    wait_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    response_delay_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    customers: list[QueueCustomerIn] | None = Field(default=None, max_length=100)


class CheckoutIn(StrictModel):
    counter_01: CounterIn | None = None
    counter_02: CounterIn | None = None


class QueueLaneIn(StrictModel):
    status: Literal["IDLE", "ACTIVE"] | None = None
    queue_count: int | None = Field(default=None, ge=0)
    estimated_wait_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class QueuesIn(StrictModel):
    counter_01: QueueLaneIn | None = None
    counter_02: QueueLaneIn | None = None


class ShelfStockIn(StrictModel):
    current_stock: int = Field(ge=0)
    max_capacity: int = Field(ge=1)


class TrafficIn(StrictModel):
    footfall_in: int = Field(ge=0)
    footfall_out: int = Field(ge=0)


class VisionAlertIn(StrictModel):
    type: Literal["QUEUE", "STOCK", "STAFF", "FLOW", "SYSTEM"]
    message: str = Field(min_length=1, max_length=500)
    evidence: EvidenceIn | None = None


class TelemetryIn(StrictModel):
    current_scenario: Literal["final_test"] | None = None
    headcount: int | None = Field(default=None, ge=0)
    traffic: TrafficIn | None = None
    queues: QueuesIn | None = None
    shelf_health: dict[str, ShelfStockIn] | None = None
    footfall: Footfall | None = None
    live_occupancy: int | None = Field(default=None, ge=0)
    queue_count: int | None = Field(default=None, ge=0)
    checkout: CheckoutIn | None = None
    shoppers: list[ShopperIn] | None = Field(default=None, max_length=200)
    staff_02: PositionIn | None = None
    coordinate_system: Literal["map", "world"] = "map"
    camera_feeds: dict[str, bool] | None = None
    fps: float | None = Field(default=None, ge=0, le=240)
    playback_seconds: float | None = Field(default=None, ge=0, le=TEST_SECONDS, allow_inf_nan=False)
    task_queue: list[TaskIn] | None = Field(default=None, max_length=200)
    inventory_health: dict[str, float] | None = None
    alerts: list[VisionAlertIn] | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def require_update(self):
        if all(value is None for value in (
            self.current_scenario, self.headcount, self.traffic, self.queues, self.shelf_health,
            self.footfall, self.live_occupancy, self.queue_count,
            self.checkout, self.shoppers, self.staff_02, self.camera_feeds, self.fps, self.playback_seconds, self.task_queue, self.inventory_health, self.alerts,
        )):
            raise ValueError("At least one telemetry field is required")
        return self


class TaskPatch(StrictModel):
    status: Literal["Assigned", "Refilled"]


class CalibrationRegion(StrictModel):
    type: Literal["line", "rect"]
    x1: float = Field(ge=0, le=100, allow_inf_nan=False)
    y1: float = Field(ge=0, le=100, allow_inf_nan=False)
    x2: float = Field(ge=0, le=100, allow_inf_nan=False)
    y2: float = Field(ge=0, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def nonempty(self):
        if abs(self.x2 - self.x1) + abs(self.y2 - self.y1) < 1:
            raise ValueError("Calibration region must span at least 1% of the image")
        if self.type == "rect" and (abs(self.x2 - self.x1) < 1 or abs(self.y2 - self.y1) < 1):
            raise ValueError("Calibration rectangle must be at least 1% wide and tall")
        return self


class CalibrationIn(StrictModel):
    regions: dict[str, CalibrationRegion] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_regions(self):
        if set(self.regions) - CALIBRATION_REGIONS:
            raise ValueError("Unknown calibration region")
        if any((name == "entry") != (region.type == "line") for name, region in self.regions.items()):
            raise ValueError("Entry must be a line; checkout and shelf regions must be rectangles")
        return self


def stamp():
    return datetime.now(timezone.utc).isoformat()


def load_zones():
    metadata = json.loads((SCENE_DIR / "metadata.json").read_text())
    zone_types = {
        "Milk": ("shelf1", "Dairy", "Shelf Row 01"),
        "Packaged Snacks": ("shelf2", "Snacks", "Shelf Row 02"),
        "CannedFoods": ("shelf4", "Canned foods", "Shelf Row 03"),
        "Softdrinks": ("shelf3", "Beverages", "Beverage Wall"),
        "Boxed": ("wall_boxed", "Boxed goods", "Boxed Goods Wall"),
        "CleaningSupplies": ("wall_cleaning", "Cleaning", "Cleaning Wall Shelves"),
        "Biscuits": ("wall_biscuits", "Biscuits", "Biscuits Wall Shelf"),
    }
    zones = []
    for box in metadata["ShelfBoxes"]:
        shelves = box["Shelves"]
        if not shelves or shelves[0].get("GoodsCategory") not in zone_types:
            continue
        zone_id, category, title = zone_types[shelves[0]["GoodsCategory"]]
        xs = [point[0] for shelf in shelves for point in shelf["bbox"]]
        ys = [point[1] for shelf in shelves for point in shelf["bbox"]]
        bounds = {
            "left": round(min(xs) / 1300 * 100, 2),
            "top": round((1200 + min(ys)) / 1200 * 100, 2),
            "width": round((max(xs) - min(xs)) / 1300 * 100, 2),
            "height": round((max(ys) - min(ys)) / 1200 * 100, 2),
        }
        layout = "side-wall" if zone_id in {"wall_boxed", "wall_cleaning"} else "rear-wall" if zone_id == "shelf3" else "front-wall" if zone_id == "wall_biscuits" else "aisle"
        # A wall fixture is drawn as a narrow rack at its measured outer edge; its full product bounds remain available.
        display = {**bounds, "left": round(max(xs) / 1300 * 100 - 6, 2), "width": 6} if layout == "side-wall" else bounds
        zones.append({
            "id": zone_id, "title": title, "category": category,
            **display, "source_bounds": bounds, "layout": layout,
            "source_id": box["ID"],
        })
    if len(zones) != len(zone_types):
        raise RuntimeError("Scene metadata must contain all seven monitored shelf boxes")
    # Register and staff positions are the calibrated stand points in marketgen_complete_flow.py.
    zones.extend([
        {"id": "lane1", "title": "Counter 01", "category": "Checkout", "left": 11.5, "top": 68.5, "width": 12, "height": 9, "source_id": "staff1_register"},
        {"id": "lane2", "title": "Counter 02", "category": "Checkout", "left": 11.5, "top": 48.5, "width": 12, "height": 9, "source_id": "staff2_register"},
        {"id": "entry", "title": "ENTRY", "category": "Threshold", "left": 26, "top": 92.5, "width": 8, "height": 5, "source_id": "entrance_threshold"},
    ])
    return zones


def event(kind, message):
    return {"time": stamp(), "type": kind, "message": message}


# shortcut: one process owns this registry; use a shared store for multiple workers or durable history.
registry = {
    "mode": "simulation" if USE_SIMULATED_DATA and AUTOPLAY else "idle",
    "measurement_source": "simulation" if USE_SIMULATED_DATA and AUTOPLAY else "none",
    "measurements": {name: bool(USE_SIMULATED_DATA and AUTOPLAY) for name in ("visible_shoppers", "footfall", "queues", "positions", "shelf_gaps")},
    "footfall": {"in": 0, "out": 0, "hourly": [], "peak_occupancy": 0},
    "live_occupancy": 0,
    "occupancy": {"count": 0, "capacity": 24, "density_pct": 0, "zones": {}},
    "checkout": {
        "counter_01": {"staff": "Staff_01", "queue_count": 0, "status": "standby", "wait_seconds": 0, "response_delay_seconds": 0, "customers": []},
        "counter_02": {"staff": "Staff_02", "queue_count": 0, "status": "standby", "wait_seconds": 0, "response_delay_seconds": 0, "customers": []},
    },
    "task_queue": [],
    "inventory_health": {zone_id: 0 for zone_id in ZONE_CAPACITY if zone_id not in {"entry", "lane1", "lane2"}},
    "footfall_history": [],
    "events": [],
    "event_index": 0,
    "shoppers": [],
    "staff_02": {"x": 26, "y": 42},
    "camera_feeds": {name: False for name in (
        "cam_entrance", "cam_checkout", "cam_back_aisle",
        "cam_entrance_end", "cam_back_aisle_left",
    )},
    "fps": None,
    "zones": load_zones(),
    "timeline": {"cycle": -1, "elapsed_seconds": 0, "phase": "initializing", "events_seen": []},
    "updated_at": None,
    "current_scenario": "final_test",
    "headcount": 0,
    "traffic": {"footfall_in": 0, "footfall_out": 0},
    "queues": {
        "counter_01": {"status": "IDLE", "queue_count": 0, "estimated_wait_time": 0},
        "counter_02": {"status": "IDLE", "queue_count": 0, "estimated_wait_time": 0},
    },
    "shelf_health": {
        name: {"current_stock": 0, "max_capacity": spec["max_capacity"]}
        for name, spec in SHELF_CATALOG.items()
    },
    "active_tickets": [],
    "system_logs": [],
    "queue_breach_latched": False,
    "queue_below_since": None,
    "queue_alerts": [],
    "vision_alert_last_seen": {},
    "backup_dispatched": False,
    "playback": {"status": "idle", "scenario": None, "pid": None},
    "playback_seconds": 0.0,
    "run_id": None,
}
lock = asyncio.Lock()
calibration_lock = asyncio.Lock()
simulation_task = None
replay_process = None
zone_dwell_seconds = {zone["id"]: 0.0 for zone in registry["zones"]}
zone_visitor_ids = {zone["id"]: set() for zone in registry["zones"]}
heatmap_seconds = [0.0] * (HEATMAP_SIZE * HEATMAP_SIZE)
tracking_clock = {"last": None, "video": None}


def snapshot():
    state = deepcopy(registry)
    state["inventory_tickets"] = state["task_queue"]
    state["active_tickets"] = state["task_queue"]
    state["log_lines"] = [item if isinstance(item, str) else item["message"] for item in state["system_logs"]]
    state["system_logs"] = state["events"]
    state.pop("queue_breach_latched", None)
    state.pop("queue_below_since", None)
    state.pop("vision_alert_last_seen", None)
    state.pop("backup_dispatched", None)
    return state


def push_event(kind, message, evidence=None):
    registry["event_index"] += 1
    item = event(kind, message)
    item["index"] = registry["event_index"]
    if evidence is not None:
        item["evidence"] = evidence.model_dump() if isinstance(evidence, EvidenceIn) else evidence
    registry["events"].insert(0, item)
    del registry["events"][MAX_EVENTS:]
    registry["system_logs"].insert(0, message)
    del registry["system_logs"][MAX_EVENTS:]


def save_latest_report():
    LATEST_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=LATEST_REPORT_FILE.parent, suffix=".json", encoding="utf-8", delete=False) as output:
        temp_path = Path(output.name)
        json.dump(snapshot(), output, separators=(",", ":"))
    os.replace(temp_path, LATEST_REPORT_FILE)


def mirror_operational_state():
    registry["headcount"] = registry["live_occupancy"]
    registry["traffic"]["footfall_in"] = registry["footfall"]["in"]
    registry["traffic"]["footfall_out"] = registry["footfall"]["out"]
    for key in ("counter_01", "counter_02"):
        lane = registry["checkout"][key]
        registry["queues"][key]["status"] = "ACTIVE" if lane["status"] == "active" else "IDLE"
        registry["queues"][key]["queue_count"] = lane["queue_count"]
        registry["queues"][key]["estimated_wait_time"] = lane["wait_seconds"]
    registry["active_tickets"] = registry["task_queue"]


def project_stock_from_percent():
    for name, spec in SHELF_CATALOG.items():
        capacity = registry["shelf_health"][name]["max_capacity"]
        percent = registry["inventory_health"][spec["zone_id"]]
        registry["shelf_health"][name]["current_stock"] = min(capacity, round(capacity * percent / 100))


def project_percent_from_stock():
    for name, spec in SHELF_CATALOG.items():
        health = registry["shelf_health"][name]
        registry["inventory_health"][spec["zone_id"]] = round(health["current_stock"] / health["max_capacity"] * 100)


def apply_track_k_threshold(counter_01_count, observed_at, video_seconds=None):
    # A short missed detection must not turn one queue episode into multiple alerts.
    breached = counter_01_count >= QUEUE_THRESHOLD
    if not breached:
        if not registry["queue_breach_latched"]:
            return
        if registry["queue_below_since"] is None:
            registry["queue_below_since"] = observed_at
        if observed_at - registry["queue_below_since"] >= 8:
            push_event("QUEUE", "Counter 01 queue normalized.")
            registry["queue_breach_latched"] = False
            registry["backup_dispatched"] = False
            registry["queue_below_since"] = None
        return
    registry["queue_below_since"] = None
    if not registry["queue_breach_latched"]:
        registry["queue_breach_latched"] = True
        registry["queue_alerts"].append({"time": stamp(), "video_seconds": video_seconds, "count": counter_01_count})
        push_event("QUEUE", f"[QUEUE_ALERT] Counter 01 queue length {counter_01_count} breaks threshold.")
    # Staff dispatch is shown only when the vision payload reports it.


def apply_queue_lane(key, lane):
    target = registry["checkout"][key]
    spec_lane = registry["queues"][key]
    if lane.queue_count is not None:
        target["queue_count"] = lane.queue_count
        spec_lane["queue_count"] = lane.queue_count
    if lane.estimated_wait_time is not None:
        target["wait_seconds"] = lane.estimated_wait_time
        spec_lane["estimated_wait_time"] = lane.estimated_wait_time
    if lane.status is not None:
        spec_lane["status"] = lane.status
        target["status"] = "active" if lane.status == "ACTIVE" else "standby"


def add_ticket(task):
    existing = next((item for item in registry["task_queue"] if item["id"] == task["id"]), None)
    task["status"] = existing["status"] if existing else "Open"
    if existing:
        existing.update(task)
        return
    registry["task_queue"].insert(0, task)
    del registry["task_queue"][200:]
    push_event("STOCK", f"{task['item']}: {task['missing']} units missing at {task['location']}", task.get("evidence"))


def normalize_position(point, system):
    x, y = point.x, point.y
    if system == "world":
        x, y = x / 1300 * 100, (1200 + y) / 1200 * 100
    if not (0 <= x <= 100 and 0 <= y <= 100):
        raise HTTPException(422, "Tracking coordinates must fall within the store map")
    return {"x": round(x, 2), "y": round(y, 2)}


def nearest_zone(x, y):
    return min(registry["zones"], key=lambda zone: (
        max(zone["left"] - x, 0, x - zone["left"] - zone["width"]) ** 2
        + max(zone["top"] - y, 0, y - zone["top"] - zone["height"]) ** 2
    ))["id"]


def refresh_occupancy(delta=0):
    counts = {zone["id"]: 0 for zone in registry["zones"]}
    for shopper in registry["shoppers"]:
        if shopper["zone"] in counts:
            counts[shopper["zone"]] += 1
            zone_dwell_seconds[shopper["zone"]] += delta
            zone_visitor_ids[shopper["zone"]].add(shopper["id"])
            column = min(HEATMAP_SIZE - 1, int(shopper["x"] * HEATMAP_SIZE / 100))
            row = min(HEATMAP_SIZE - 1, int(shopper["y"] * HEATMAP_SIZE / 100))
            heatmap_seconds[row * HEATMAP_SIZE + column] += delta
    count = registry["live_occupancy"]
    registry["occupancy"] = {
        "count": count, "capacity": 24, "density_pct": round(count / 24 * 100),
        "zones": {zone_id: {
            "headcount": value, "capacity": ZONE_CAPACITY[zone_id],
            "density_pct": min(100, round(value / ZONE_CAPACITY[zone_id] * 100)),
            "dwell_hours": round(zone_dwell_seconds[zone_id] / 3600, 3),
            "visits": len(zone_visitor_ids[zone_id]),
        } for zone_id, value in counts.items()},
        "heatmap": {"size": HEATMAP_SIZE, "seconds": [round(value, 2) for value in heatmap_seconds]},
    }
    registry["footfall"]["peak_occupancy"] = max(registry["footfall"]["peak_occupancy"], count)


def append_history():
    point = {"time": stamp(), "in": registry["footfall"]["in"], "out": registry["footfall"]["out"], "occupancy": registry["live_occupancy"]}
    history = registry["footfall_history"]
    if not history or history[-1]["time"][:19] != point["time"][:19]:
        history.append(point)
        del history[:-MAX_HISTORY]


def update_hourly(in_delta, out_delta):
    hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
    hourly = registry["footfall"]["hourly"]
    if not hourly or hourly[-1]["time"] != hour:
        hourly.append({"time": hour, "in": 0, "out": 0})
        del hourly[:-12]
    hourly[-1]["in"] += in_delta
    hourly[-1]["out"] += out_delta


def simulated_position(index, elapsed):
    anchors = [(48, 64), (59, 64), (70, 64), (80, 64), (48, 35), (59, 35), (70, 35), (80, 35)]
    if elapsed < 45:
        x, y = anchors[index]
        return round(x + math.sin(elapsed * .7 + index) * 1.5, 2), round(y + math.cos(elapsed * .55 + index) * 1.2, 2), "shelf1" if index < 4 else "shelf2"
    if elapsed < 50:
        return 27 + index * 2.2, 73 + math.sin(elapsed * .7 + index) * .2, "lane1"
    lane = "lane1" if index % 2 == 0 else "lane2"
    x, y = 27 + index // 2 * 2.5, 73 if lane == "lane1" else 53
    if elapsed < 60:
        return x, y, lane
    exit_at = 60 + (index + 1) * 3.75
    if elapsed >= exit_at:
        return None
    progress = max(0, min(1, (elapsed - (exit_at - 3.75)) / 3.75))
    return round(x + (30 - x) * progress, 2), round(y + (94 - y) * progress, 2), "entry" if progress > .5 else lane


def simulated_shoppers(elapsed, cycle):
    shoppers = []
    for index in range(8):
        current = simulated_position(index, elapsed)
        if current is None:
            continue
        trail = []
        for previous in (elapsed - 1.2, elapsed - .8, elapsed - .4, elapsed):
            point = simulated_position(index, max(0, previous))
            if point is not None:
                trail.append([point[0], point[1]])
        shoppers.append({"id": f"shopper_{cycle}_{index+1:02d}", "x": current[0], "y": current[1], "zone": current[2], "trail": trail})
    return shoppers


async def advance_timeline(elapsed, cycle=0):
    elapsed = max(0, min(float(elapsed), TIMELINE_SECONDS - .001))
    async with lock:
        registry["measurements"] = {name: True for name in registry["measurements"]}
        timeline = registry["timeline"]
        previous = timeline["elapsed_seconds"]
        if timeline["cycle"] != cycle:
            timeline.update({"cycle": cycle, "elapsed_seconds": 0, "phase": "shopping", "events_seen": []})
            previous = 0
            push_event("SYSTEM", f"Simulation cycle {cycle + 1} started; eight shoppers entered.")
        if not registry["footfall"]["hourly"]:
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            for hours_ago, arrivals in enumerate((3, 5, 4, 7, 6, 9, 11, 8, 10, 12, 7), start=1):
                registry["footfall"]["hourly"].insert(0, {"time": (now - timedelta(hours=hours_ago)).isoformat(), "in": arrivals, "out": max(0, arrivals - 1)})
            timeline["base_in"] = sum(item["in"] for item in registry["footfall"]["hourly"])
            timeline["base_out"] = sum(item["out"] for item in registry["footfall"]["hourly"])
            registry["footfall"]["in"] = timeline["base_in"]
            registry["footfall"]["out"] = timeline["base_out"]
        seen = timeline["events_seen"]

        def trigger(name, when, kind, message):
            if when and name not in seen:
                seen.append(name)
                push_event(kind, message)

        if elapsed >= 25:
            add_ticket({"id": f"MILK-{cycle+1:03d}", "item": "1L Milk Carton", "category": "Dairy", "location": "Aisle_1_Tier_2", "missing": 5, "shelf": "shelf1"})
        if elapsed >= 35:
            add_ticket({"id": f"SNACK-{cycle+1:03d}", "item": "Chips Bag", "category": "Snacks", "location": "Aisle_2_Tier_1", "missing": 3, "shelf": "shelf2"})
        active_shelves = {task["shelf"] for task in registry["task_queue"] if task["status"] != "Refilled" and task["id"].endswith(f"{cycle+1:03d}")}
        registry["inventory_health"].update({"shelf1": 37 if "shelf1" in active_shelves else 100, "shelf2": 62 if "shelf2" in active_shelves else 100})
        trigger("queue", elapsed >= 45, "QUEUE", "[QUEUE_ALERT] Counter 1 queue length 4 breaks threshold. Dispatching backup.")
        trigger("dispatch", elapsed >= 50, "STAFF", "Staff_02 left backroom doorway for Counter 02.")
        trigger("active", elapsed >= 55, "STAFF", "Counter 02 active; shoppers split between both lanes.")
        trigger("clear", elapsed >= 60, "FLOW", "Checkout processing started; shoppers exiting through ENTRY.")
        exited = min(8, max(0, int((elapsed - 60) / 3.75)))
        registry["shoppers"] = simulated_shoppers(elapsed, cycle)
        registry["live_occupancy"] = len(registry["shoppers"])
        target_in = timeline["base_in"] + 8 * (cycle + 1)
        target_out = timeline["base_out"] + 8 * cycle + exited
        update_hourly(target_in - registry["footfall"]["in"], target_out - registry["footfall"]["out"])
        registry["footfall"]["in"], registry["footfall"]["out"] = target_in, target_out
        one, two = registry["checkout"]["counter_01"], registry["checkout"]["counter_02"]
        one["queue_count"] = 4 if 45 <= elapsed < 50 else 2 if 50 <= elapsed < 60 else max(0, 2 - int((elapsed - 60) / 7.5)) if elapsed >= 60 else 0
        two["queue_count"] = 2 if 50 <= elapsed < 60 else max(0, 2 - int((elapsed - 60) / 7.5)) if elapsed >= 60 else 0
        two["status"] = "standby" if elapsed < 45 else "dispatching" if elapsed < 55 else "active"
        one["customers"] = [{"id": item["id"], "wait_seconds": round(max(0, elapsed - 45), 1)} for item in registry["shoppers"][:one["queue_count"]]]
        two["customers"] = [{"id": item["id"], "wait_seconds": round(max(0, elapsed - 50), 1)} for item in registry["shoppers"][one["queue_count"]:one["queue_count"] + two["queue_count"]]]
        one["wait_seconds"] = max((item["wait_seconds"] for item in one["customers"]), default=0)
        two["wait_seconds"] = max((item["wait_seconds"] for item in two["customers"]), default=0)
        two["response_delay_seconds"] = round(max(0, min(elapsed - 45, 10)), 1)
        progress = max(0, min(1, (elapsed - 50) / 5))
        registry["staff_02"] = {"x": round(26 - 6 * progress, 2), "y": 42}
        registry["camera_feeds"] = {key: False for key in registry["camera_feeds"]}
        registry["fps"] = 10.0
        refresh_occupancy(max(0, min(.5, elapsed - previous)))
        append_history()
        timeline.update({"elapsed_seconds": round(elapsed, 1), "phase": "shopping" if elapsed < 25 else "stock alert" if elapsed < 45 else "queue alert" if elapsed < 50 else "dispatch" if elapsed < 55 else "dual checkout" if elapsed < 60 else "exit flow"})
        project_stock_from_percent()
        mirror_operational_state()
        registry["updated_at"] = stamp()
        return snapshot()


def vision_python():
    configured = os.environ.get("TRACK_K_VISION_PYTHON", "").strip()
    return configured or sys.executable


def model_runtime_issue():
    python = vision_python()
    if not Path(python).is_file():
        return "Model Python interpreter is unavailable"
    try:
        check = subprocess.run([python, "-c", "import cv2, torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "Model runtime is unavailable"
    return None if check.returncode == 0 else "CUDA inference is unavailable on this server"


def replay_issue(scenario_id):
    if scenario_id != "final_test":
        return "Unknown scenario"
    uploaded = [(VIDEO_DIR / scenario_id / f"{camera_id}.mp4").is_file() for camera_id in VIDEO_CAMERAS]
    if any(uploaded) and not all(uploaded):
        return "Replace all five camera angles before running analysis"
    if not all(((VIDEO_DIR / scenario_id / f"{camera_id}.mp4").is_file() or (FINAL_TEST_DIR / filename).is_file()) for camera_id, (_, filename) in VIDEO_CAMERAS.items()):
        return "Five synchronized camera videos are required"
    model = ROOT_DIR / "model_artifacts" / "track_k_v2_001"
    if not all((model / path).is_file() for path in ("weights/person_selected.pth", "weights/shelf_selected.pth", "calibration/calibration.json", "calibration/person_operating.json", "calibration/shelf_operating.json")):
        return "Trained model or frozen calibration is missing"
    return model_runtime_issue()


def clear_results():
    registry["measurement_source"] = "none"
    registry["measurements"] = {name: False for name in registry["measurements"]}
    registry["footfall"].update({"in": 0, "out": 0, "hourly": [], "peak_occupancy": 0})
    registry["traffic"] = {"footfall_in": 0, "footfall_out": 0}
    registry["live_occupancy"] = 0
    registry["headcount"] = 0
    registry["shoppers"] = []
    registry["footfall_history"] = []
    registry["events"] = []
    registry["system_logs"] = []
    registry["event_index"] = 0
    registry["task_queue"] = []
    registry["active_tickets"] = []
    registry["queue_breach_latched"] = False
    registry["queue_below_since"] = None
    registry["queue_alerts"] = []
    registry["vision_alert_last_seen"] = {}
    registry["backup_dispatched"] = False
    for lane in registry["checkout"].values():
        lane.update({"queue_count": 0, "status": "standby", "wait_seconds": 0, "response_delay_seconds": 0, "customers": []})
    for lane in registry["queues"].values():
        lane.update({"status": "IDLE", "queue_count": 0, "estimated_wait_time": 0})
    for name, spec in SHELF_CATALOG.items():
        registry["shelf_health"][name] = {"current_stock": 0, "max_capacity": spec["max_capacity"]}
    for zone_id in registry["inventory_health"]:
        registry["inventory_health"][zone_id] = 0
    registry["camera_feeds"] = {key: False for key in registry["camera_feeds"]}
    registry["fps"] = None
    registry["playback_seconds"] = 0.0
    registry["run_id"] = None
    registry["timeline"].update({"cycle": -1, "elapsed_seconds": 0, "phase": "idle", "events_seen": []})
    zone_dwell_seconds.update({zone_id: 0.0 for zone_id in zone_dwell_seconds})
    for visitors in zone_visitor_ids.values():
        visitors.clear()
    heatmap_seconds[:] = [0.0] * len(heatmap_seconds)
    tracking_clock["last"] = None
    tracking_clock["video"] = None
    refresh_occupancy(0)


async def watch_replay(process, scenario_id):
    log_handle = process.replay_log
    code = await process.wait()
    log_handle.close()
    shutil.rmtree(process.input_dir.parent, ignore_errors=True)
    async with lock:
        playback = registry["playback"]
        if playback.get("pid") == process.pid and playback.get("scenario") == scenario_id:
            playback["status"] = "complete" if code == 0 else "error"
            playback["exit_code"] = code
            registry["updated_at"] = stamp()
        runs = read_runs()
        for run in runs:
            if run["id"] == getattr(process, "run_id", None):
                run.update({"status": "complete" if code == 0 else "stopped" if code == -15 else "error", "ended_at": stamp()})
                write_runs(runs)
                break
        if code == 0 and registry["run_id"] == getattr(process, "run_id", None) and any(run["id"] == process.run_id and run["status"] == "complete" for run in runs):
            save_latest_report()


async def stop_replay():
    global replay_process
    process = replay_process
    replay_process = None
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 2)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def simulation_loop():
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        await advance_timeline(elapsed % TIMELINE_SECONDS, int(elapsed // TIMELINE_SECONDS))
        await asyncio.sleep(.1)


@asynccontextmanager
async def lifespan(_app):
    global simulation_task
    if not (USE_SIMULATED_DATA and AUTOPLAY) and LATEST_REPORT_FILE.is_file():
        try:
            saved = json.loads(LATEST_REPORT_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        if saved.get("run_id") in {run["id"] for run in read_runs() if run["status"] == "complete"}:
            registry.update(saved)
    simulation_task = asyncio.create_task(simulation_loop()) if USE_SIMULATED_DATA and AUTOPLAY else None
    try:
        yield
    finally:
        await stop_replay()
        if simulation_task:
            simulation_task.cancel()
            with suppress(asyncio.CancelledError):
                await simulation_task
            simulation_task = None


app = FastAPI(title="Retail Vision", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in os.environ.get(
            "TRACK_K_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173",
        ).split(",") if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def authenticate(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    ingest_token = os.environ.get("TRACK_K_INGEST_TOKEN")
    operator_user = os.environ.get("TRACK_K_OPERATOR_USER")
    operator_password = os.environ.get("TRACK_K_OPERATOR_PASSWORD")
    if not all((ingest_token, operator_user, operator_password)):
        return JSONResponse({"detail": "Server credentials are not configured"}, status_code=503)

    authorization = request.headers.get("authorization", "")
    if request.url.path == "/api/telemetry" and request.method == "POST":
        supplied = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        if not secrets.compare_digest(supplied, ingest_token):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    else:
        try:
            scheme, encoded = authorization.split(" ", 1)
            user, password = base64.b64decode(encoded, validate=True).decode().split(":", 1)
        except (ValueError, UnicodeDecodeError):
            scheme, user, password = "", "", ""
        if scheme.lower() != "basic" or not (
            secrets.compare_digest(user, operator_user)
            and secrets.compare_digest(password, operator_password)
        ):
            return JSONResponse(
                {"detail": "Unauthorized"}, status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Track K"'},
            )
    return await call_next(request)


@app.post("/api/telemetry")
async def ingest(payload: TelemetryIn):
    if simulation_task is not None and not simulation_task.done():
        raise HTTPException(409, "Simulation is active; stop autoplay before ingesting vision data")
    if payload.camera_feeds is not None:
        unknown = set(payload.camera_feeds) - set(registry["camera_feeds"])
        if unknown:
            raise HTTPException(422, f"Unknown camera feed: {', '.join(sorted(unknown))}")
    if payload.inventory_health is not None:
        if set(payload.inventory_health) - set(registry["inventory_health"]) or any(not math.isfinite(value) or not 0 <= value <= 100 for value in payload.inventory_health.values()):
            raise HTTPException(422, "inventory_health must contain known shelf IDs and percentages from 0 to 100")
    if payload.shelf_health is not None:
        unknown_shelves = set(payload.shelf_health) - set(SHELF_CATALOG)
        if unknown_shelves:
            raise HTTPException(422, f"Unknown shelf: {', '.join(sorted(unknown_shelves))}")
        for name, shelf in payload.shelf_health.items():
            if shelf.current_stock > shelf.max_capacity:
                raise HTTPException(422, f"{name} current_stock exceeds max_capacity")
    if payload.headcount is not None and payload.live_occupancy is not None and payload.headcount != payload.live_occupancy:
        raise HTTPException(422, "headcount and live_occupancy must match")
    converted = []
    if payload.shoppers is not None:
        known = {zone["id"] for zone in registry["zones"]}
        ids = [shopper.id for shopper in payload.shoppers]
        if len(ids) != len(set(ids)):
            raise HTTPException(422, "Shopper IDs must be unique")
        for shopper in payload.shoppers:
            if shopper.zone is not None and shopper.zone not in known:
                raise HTTPException(422, f"Unknown zone: {shopper.zone}")
            position = normalize_position(shopper, payload.coordinate_system)
            converted.append({"id": shopper.id, **position, "zone": shopper.zone or nearest_zone(**position)})
    staff_position = normalize_position(payload.staff_02, payload.coordinate_system) if payload.staff_02 else None
    occupancy = payload.headcount if payload.headcount is not None else payload.live_occupancy
    if payload.shoppers is not None and occupancy is not None and len(converted) > occupancy:
        raise HTTPException(422, "Visible shopper positions cannot exceed live_occupancy")
    async with lock:
        if registry["playback"]["status"] != "running":
            registry["measurement_source"] = "external_vision"
            registry["mode"] = "vision"
            registry["run_id"] = None
        if payload.headcount is not None or payload.live_occupancy is not None:
            registry["measurements"]["visible_shoppers"] = True
        if payload.traffic is not None or payload.footfall is not None:
            registry["measurements"]["footfall"] = True
        if payload.queues is not None or payload.checkout is not None or payload.queue_count is not None:
            registry["measurements"]["queues"] = True
        if payload.shoppers is not None:
            registry["measurements"]["positions"] = True
        if payload.shelf_health is not None or payload.inventory_health is not None or payload.task_queue is not None:
            registry["measurements"]["shelf_gaps"] = True
        if payload.current_scenario is not None:
            registry["current_scenario"] = payload.current_scenario
        if payload.traffic is not None:
            incoming_traffic = {"in": payload.traffic.footfall_in, "out": payload.traffic.footfall_out}
            if any(incoming_traffic[key] < registry["footfall"][key] for key in ("in", "out")):
                raise HTTPException(409, "Absolute footfall counters cannot decrease")
            update_hourly(incoming_traffic["in"] - registry["footfall"]["in"], incoming_traffic["out"] - registry["footfall"]["out"])
            registry["footfall"].update(incoming_traffic)
        if payload.footfall is not None:
            incoming = payload.footfall.model_dump(by_alias=True)
            if any(incoming[key] < registry["footfall"][key] for key in ("in", "out")):
                raise HTTPException(409, "Absolute footfall counters cannot decrease")
            update_hourly(incoming["in"] - registry["footfall"]["in"], incoming["out"] - registry["footfall"]["out"])
            registry["footfall"].update(incoming)
        if payload.shoppers is not None:
            previous = {shopper["id"]: shopper["trail"] for shopper in registry["shoppers"]}
            registry["shoppers"] = [{**shopper, "trail": (previous.get(shopper["id"], []) + [[shopper["x"], shopper["y"]]])[-5:]} for shopper in converted]
            registry["live_occupancy"] = occupancy if occupancy is not None else len(converted)
        elif occupancy is not None:
            registry["live_occupancy"] = occupancy
        if staff_position is not None:
            registry["staff_02"] = staff_position
        if payload.queues is not None:
            for key in ("counter_01", "counter_02"):
                lane = getattr(payload.queues, key)
                if lane is not None:
                    apply_queue_lane(key, lane)
        if payload.checkout is not None:
            for key in ("counter_01", "counter_02"):
                update = getattr(payload.checkout, key)
                if update is not None:
                    registry["checkout"][key].update(update.model_dump(exclude_none=True))
        if payload.queue_count is not None and not (
            (payload.queues is not None and payload.queues.counter_01 is not None and payload.queues.counter_01.queue_count is not None)
            or (payload.checkout is not None and payload.checkout.counter_01 is not None and payload.checkout.counter_01.queue_count is not None)
        ):
            registry["checkout"]["counter_01"]["queue_count"] = payload.queue_count
        if payload.shelf_health is not None:
            for name, shelf in payload.shelf_health.items():
                registry["shelf_health"][name] = {"current_stock": shelf.current_stock, "max_capacity": shelf.max_capacity}
            project_percent_from_stock()
        observed_at = payload.playback_seconds if payload.playback_seconds is not None else time.monotonic()
        apply_track_k_threshold(registry["checkout"]["counter_01"]["queue_count"], observed_at, payload.playback_seconds)
        if payload.camera_feeds is not None:
            registry["camera_feeds"].update(payload.camera_feeds)
        if payload.fps is not None:
            registry["fps"] = payload.fps
        if payload.playback_seconds is not None:
            registry["playback_seconds"] = payload.playback_seconds
        if payload.inventory_health is not None:
            registry["inventory_health"].update(payload.inventory_health)
        if payload.task_queue is not None:
            category_zone = {"Dairy": "shelf1", "Snacks": "shelf2", "Beverages": "shelf3", "CannedFoods": "shelf4", "Boxed": "wall_boxed", "CleaningSupplies": "wall_cleaning", "Biscuits": "wall_biscuits"}
            for incoming in payload.task_queue:
                add_ticket({**incoming.model_dump(), "shelf": category_zone[incoming.category]})
        if payload.alerts is not None:
            for alert in payload.alerts:
                if alert.type == "QUEUE" and alert.message.startswith("[QUEUE_ALERT]"):
                    continue
                key = (alert.type, alert.message)
                last_seen = registry["vision_alert_last_seen"].get(key)
                registry["vision_alert_last_seen"][key] = observed_at
                if last_seen is not None and 0 <= observed_at - last_seen < 8:
                    continue
                push_event(alert.type, alert.message, alert.evidence)
        now = time.monotonic()
        if payload.shoppers is not None and payload.playback_seconds is not None:
            previous_video = tracking_clock["video"]
            delta = max(0, payload.playback_seconds - previous_video) if previous_video is not None else 0
        else:
            delta = min(5, now - tracking_clock["last"]) if payload.shoppers is not None and tracking_clock["last"] is not None else 0
        if payload.shoppers is not None:
            tracking_clock["last"] = now
            tracking_clock["video"] = payload.playback_seconds
        refresh_occupancy(delta)
        append_history()
        if payload.inventory_health is not None:
            project_stock_from_percent()
        mirror_operational_state()
        registry["updated_at"] = stamp()
        return snapshot()


@app.get("/api/stream")
async def stream(request: Request):
    async def updates():
        while not await request.is_disconnected():
            async with lock:
                state = json.dumps(snapshot(), separators=(",", ":"))
            yield f"data: {state}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(
        updates(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.patch("/api/tasks/{ticket_id}")
async def update_task(ticket_id: str, patch: TaskPatch):
    async with lock:
        task = next((item for item in registry["task_queue"] if item["id"] == ticket_id), None)
        if task is None:
            raise HTTPException(404, "Ticket not found")
        next_status = {"Open": "Assigned", "Assigned": "Refilled"}.get(task["status"])
        if patch.status != next_status:
            raise HTTPException(409, f"Next status must be {next_status or 'none'}")
        task["status"] = patch.status
        push_event("TASK", f"{ticket_id} · {task['item']} marked {patch.status.lower()}")
        registry["updated_at"] = stamp()
        if registry["run_id"] and registry["playback"]["status"] == "complete":
            save_latest_report()
        return deepcopy(task)


@app.post("/api/scenarios/{scenario_id}/run")
async def run_scenario(scenario_id: str):
    global replay_process
    if scenario_id not in VIDEO_SCENARIOS:
        raise HTTPException(404, "Unknown scenario")
    issue = replay_issue(scenario_id)
    if issue:
        raise HTTPException(409, issue)
    engine_path = ROOT_DIR / "model_artifacts" / "track_k_v2_001" / "code" / "run_final_test.py"
    source = "model_inference"
    if not engine_path.is_file():
        raise HTTPException(503, "Vision engine is not installed")
    python = vision_python()
    if not Path(python).is_file():
        raise HTTPException(503, "Vision engine interpreter is not installed")
    await stop_replay()
    run_id = uuid4().hex[:12]
    async with lock:
        clear_results()
        registry["mode"] = "vision"
        registry["measurement_source"] = source
        registry["current_scenario"] = scenario_id
        registry["playback"] = {"status": "running", "scenario": scenario_id, "pid": None}
        registry["run_id"] = run_id
        registry["updated_at"] = stamp()
        runs = read_runs()
        for run in runs:
            if run["status"] == "running":
                run.update({"status": "interrupted", "ended_at": stamp()})
        write_runs([{"id": run_id, "scenario": scenario_id, "source": source, "status": "running", "started_at": stamp(), "ended_at": None}] + runs)
    REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(REPLAY_LOG, "ab")
    environment = os.environ.copy()
    environment["TRACK_K_TELEMETRY_URL"] = environment.get("TRACK_K_TELEMETRY_URL", "http://127.0.0.1:5173/api/telemetry")
    input_dir = ROOT_DIR / "backend" / "data" / "inputs" / run_id / "final_test"
    input_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, (_, filename) in VIDEO_CAMERAS.items():
        (input_dir / filename).symlink_to(video_file(scenario_id, camera_id).resolve())
    environment["TRACK_K_FINAL_TEST_DIR"] = str(input_dir)
    try:
        process = await asyncio.create_subprocess_exec(
            python,
            str(engine_path),
            scenario_id,
            cwd=str(engine_path.parent),
            env=environment,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        log_handle.close()
        shutil.rmtree(input_dir.parent, ignore_errors=True)
        async with lock:
            registry["mode"] = "idle"
            registry["playback"] = {"status": "error", "scenario": scenario_id, "pid": None}
            runs = read_runs()
            if runs and runs[0]["id"] == run_id:
                runs[0].update({"status": "error", "ended_at": stamp()})
                write_runs(runs)
        raise HTTPException(503, "Vision engine failed to start") from exc
    process.replay_log = log_handle
    process.input_dir = input_dir
    process.run_id = run_id
    replay_process = process
    async with lock:
        registry["playback"]["pid"] = process.pid
    asyncio.create_task(watch_replay(process, scenario_id))
    return {"scenario": scenario_id, "status": "running", "label": VIDEO_SCENARIOS[scenario_id], "run_id": run_id}


@app.get("/api/videos")
async def list_videos():
    scenarios = []
    for scenario_id, label in VIDEO_SCENARIOS.items():
        cameras = []
        for camera_id, (camera_label, filename) in VIDEO_CAMERAS.items():
            uploaded = VIDEO_DIR / scenario_id / f"{camera_id}.mp4"
            local_export = FINAL_TEST_DIR / filename
            if uploaded.is_file():
                source, file = "uploaded", uploaded
            elif local_export.is_file():
                source, file = "final test", local_export
            else:
                source, file = None, local_export
            cameras.append({
                "id": camera_id,
                "label": camera_label,
                "source": source,
                "bytes": file.stat().st_size if source else 0,
                "modified_at": file.stat().st_mtime_ns if source else None,
            })
        issue = replay_issue(scenario_id)
        scenarios.append({"id": scenario_id, "label": label, "cameras": cameras, "duration_seconds": TEST_SECONDS, "can_run": issue is None, "run_issue": issue})
    return {"scenarios": scenarios, "formats": ["mp4"], "max_bytes": VIDEO_LIMIT_BYTES}


def video_file(scenario_id: str, camera_id: str):
    if scenario_id not in VIDEO_SCENARIOS or camera_id not in VIDEO_CAMERAS:
        raise HTTPException(404, "Unknown scenario or camera")
    filename = VIDEO_CAMERAS[camera_id][1]
    for path in (VIDEO_DIR / scenario_id / f"{camera_id}.mp4", FINAL_TEST_DIR / filename):
        if path.is_file():
            return path
    raise HTTPException(404, "Video is not available")


@app.get("/api/videos/{scenario_id}/{camera_id}/file")
async def review_video(scenario_id: str, camera_id: str):
    return FileResponse(video_file(scenario_id, camera_id), media_type="video/mp4")


@app.get("/api/videos/{scenario_id}/{camera_id}/frame")
async def evidence_frame(scenario_id: str, camera_id: str, at: float = Query(ge=0, le=86400)):
    path = video_file(scenario_id, camera_id)
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        image, _ = await asyncio.wait_for(process.communicate(), timeout=15)
    except FileNotFoundError:
        raise HTTPException(503, "Frame extraction is unavailable") from None
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise HTTPException(504, "Frame extraction timed out") from None
    if process.returncode or not image:
        raise HTTPException(422, "The requested video frame is unavailable")
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


def calibration_key(scenario_id: str, camera_id: str):
    if scenario_id not in VIDEO_SCENARIOS or camera_id not in VIDEO_CAMERAS:
        raise HTTPException(404, "Unknown scenario or camera")
    return f"{scenario_id}/{camera_id}"


def read_calibrations():
    if not CALIBRATION_FILE.is_file():
        return {}
    return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))


def read_runs():
    if not RUN_FILE.is_file():
        return []
    return json.loads(RUN_FILE.read_text(encoding="utf-8"))


def write_runs(runs):
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=RUN_FILE.parent, suffix=".json", encoding="utf-8", delete=False) as output:
        temp_path = Path(output.name)
        json.dump(runs[:20], output, separators=(",", ":"))
    try:
        os.replace(temp_path, RUN_FILE)
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/runs")
async def list_runs():
    async with lock:
        runs = read_runs()
        for run in runs:
            if run["status"] == "running" and run["id"] != registry["run_id"]:
                run["status"] = "interrupted"
        return {"runs": runs}


@app.get("/api/calibrations/{scenario_id}/{camera_id}")
async def get_calibration(scenario_id: str, camera_id: str):
    key = calibration_key(scenario_id, camera_id)
    async with calibration_lock:
        return read_calibrations().get(key, {"regions": {}, "updated_at": None})


@app.put("/api/calibrations/{scenario_id}/{camera_id}")
async def save_calibration(scenario_id: str, camera_id: str, calibration: CalibrationIn):
    key = calibration_key(scenario_id, camera_id)
    async with calibration_lock:
        entries = read_calibrations()
        entries[key] = {"regions": calibration.model_dump()["regions"], "updated_at": stamp()}
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=CALIBRATION_FILE.parent, suffix=".json", encoding="utf-8", delete=False) as output:
            temp_path = Path(output.name)
            json.dump(entries, output, separators=(",", ":"))
        try:
            os.replace(temp_path, CALIBRATION_FILE)
        finally:
            temp_path.unlink(missing_ok=True)
        return entries[key]


@app.put("/api/videos/{scenario_id}/{camera_id}")
async def upload_video(scenario_id: str, camera_id: str, request: Request):
    if scenario_id not in VIDEO_SCENARIOS or camera_id not in VIDEO_CAMERAS:
        raise HTTPException(404, "Unknown scenario or camera")
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "video/mp4":
        raise HTTPException(415, "Only MP4 video is supported")
    length = request.headers.get("content-length")
    if length:
        if not length.isdecimal():
            raise HTTPException(400, "Invalid content length")
        if int(length) > VIDEO_LIMIT_BYTES:
            raise HTTPException(413, "Video exceeds 500 MB limit")

    destination = VIDEO_DIR / scenario_id
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".upload", delete=False) as temp:
        temp_path = Path(temp.name)
    size = 0
    try:
        with temp_path.open("wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > VIDEO_LIMIT_BYTES:
                    raise HTTPException(413, "Video exceeds 500 MB limit")
                output.write(chunk)
        with temp_path.open("rb") as staged:
            header = staged.read(12)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise HTTPException(422, "Invalid MP4 file")
        os.replace(temp_path, destination / f"{camera_id}.mp4")
    finally:
        temp_path.unlink(missing_ok=True)
    return {"scenario": scenario_id, "camera": camera_id, "source": "uploaded", "bytes": size}


@app.get("/api/electronics/scene.glb")
async def electronics_scene():
    if not ELECTRONICS_MODEL.is_file():
        raise HTTPException(404, "Electronics shop model is missing")
    return FileResponse(ELECTRONICS_MODEL, media_type="model/gltf-binary", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
async def dashboard():
    return FileResponse(DIST_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets", check_dir=False), name="assets")
mimetypes.add_type("image/webp", ".webp")
app.mount("/media", StaticFiles(directory=DIST_DIR / "media", check_dir=False), name="media")
