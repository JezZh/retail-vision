import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Camera,
  Check,
  ChevronDown,
  Clock3,
  DoorOpen,
  LayoutDashboard,
  MapPinned,
  Menu,
  PackageCheck,
  Play,
  Radio,
  ScanLine,
  ShoppingBag,
  Users,
  Upload,
  X,
} from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const nextStatus = { Open: "Assigned", Assigned: "Refilled" };
const panel = "panel bg-neutral-900/60 backdrop-blur-md border border-neutral-800/50";
const displayTime = (value) => (value ? new Date(value).toLocaleTimeString("en-GB", { hour12: false }) : "—");
const displayHour = (value) => (value ? new Date(value).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false }) : "—");
const ORBIT_FRAMES = 36;
const pages = { overview: "Overview", cameras: "Cameras", reports: "Reports", tasks: "Task Queues", logs: "System Logs" };
const currentPage = () => typeof window === "undefined" ? "Overview" : pages[window.location.hash.slice(1)] ?? "Overview";
const clipTime = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const scenarioName = (scenario) => scenario.label;
const frameUrl = (scenario, evidence) => `/api/videos/${encodeURIComponent(scenario)}/${encodeURIComponent(evidence.camera_id)}/frame?at=${encodeURIComponent(evidence.video_seconds)}`;
const formatDwell = (hours) => hours < 1 / 60 ? `${Math.round(hours * 3600)}s` : `${Math.round(hours * 60)}m`;

function CameraWorkspace({ scenario, selectedScenario, playback, playbackSeconds, reviewRequest, evidenceEvents }) {
  const [cameraId, setCameraId] = useState("cam_entrance");
  const [reviewTime, setReviewTime] = useState(0);
  const [pendingSeek, setPendingSeek] = useState(null);
  const videoRef = useRef(null);
  const camera = scenario?.cameras.find((item) => item.id === cameraId);
  const available = Boolean(camera?.source);

  useEffect(() => {
    if (!reviewRequest) return;
    if (cameraId !== reviewRequest.camera_id) setCameraId(reviewRequest.camera_id);
    setPendingSeek(reviewRequest.video_seconds);
    if (cameraId === reviewRequest.camera_id && videoRef.current?.readyState >= 1) {
      videoRef.current.currentTime = reviewRequest.video_seconds;
      videoRef.current.pause();
      setPendingSeek(null);
    }
  }, [reviewRequest]);

  const jumpToEvidence = (evidence) => {
    if (cameraId !== evidence.camera_id) { setCameraId(evidence.camera_id); setPendingSeek(evidence.video_seconds); }
    else if (videoRef.current) { videoRef.current.currentTime = evidence.video_seconds; videoRef.current.pause(); }
  };

  return <section className={`${panel} camera-workspace`} aria-label="Camera review">
        <div className="card-heading"><div><div className="section-kicker">CAMERA REVIEW</div><h2>Review video</h2></div><span className="workspace-source">{available ? camera.source === "uploaded" ? "UPLOADED CLIP" : "FINAL TEST" : "NO VIDEO"} · {clipTime(reviewTime)}</span></div>
    <div className="workspace-grid">
      <div className="workspace-player-column">
        <div className="camera-select" role="group" aria-label="Review camera angle">{scenario?.cameras.map((item) => <button type="button" key={item.id} aria-pressed={cameraId === item.id} className={cameraId === item.id ? "active" : ""} onClick={() => { setCameraId(item.id); setReviewTime(0); }}>{item.label}</button>)}</div>
        <div className="workspace-video-frame">
          {available ? <video key={`${selectedScenario}/${cameraId}/${camera.modified_at}`} ref={videoRef} src={`/api/videos/${selectedScenario}/${cameraId}/file?v=${camera.modified_at}`} controls preload="metadata" onTimeUpdate={(event) => setReviewTime(event.currentTarget.currentTime)} onLoadedMetadata={(event) => { if (pendingSeek != null) { event.currentTarget.currentTime = pendingSeek; setPendingSeek(null); } }} aria-label={`${camera.label} source video`} /> : <div className="workspace-video-empty">No clip in this camera slot</div>}
        </div>
        <div className="workspace-timeline"><span>VIDEO EVENTS · {evidenceEvents.length}</span><span>{playback.scenario === selectedScenario && playback.status === "running" ? `ANALYSIS ${clipTime(playbackSeconds)} / ${clipTime(scenario?.duration_seconds ?? 120)}` : ""}</span></div>
        {evidenceEvents.length ? <div className="cue-list">{evidenceEvents.map((item, index) => <button type="button" key={`${item.time}-${index}`} onClick={() => jumpToEvidence(item.evidence)}><strong>{clipTime(item.evidence.video_seconds)}</strong><span>{item.message}</span><ArrowRight size={13} /></button>)}</div> : <p className="workspace-empty-cues">No events available.</p>}
      </div>
    </div>
  </section>;
}

function ElectronicsOrbit() {
  return (
    <model-viewer
      src="/api/electronics/scene.glb?v=5"
      camera-controls=""
      auto-rotate=""
      camera-orbit="32deg 68deg 15m"
      camera-target="-4m 2.6m -1m"
      field-of-view="28deg"
      min-camera-orbit="auto auto 8m"
      max-camera-orbit="auto auto 32m"
      alt="Two-floor electronics shop"
    />
  );
}

function SceneOrbit() {
  const [frame, setFrame] = useState(0);
  const drag = useRef(null);
  useEffect(() => {
    const frames = Array.from({ length: ORBIT_FRAMES }, (_, index) => {
      const image = new Image();
      image.src = `/media/orbit/frame-${String(index).padStart(2, "0")}.webp`;
      return image;
    });
    return () => { frames.length = 0; };
  }, []);
  const rotate = (value) => setFrame(((value % ORBIT_FRAMES) + ORBIT_FRAMES) % ORBIT_FRAMES);
  return (
    <div
      className="scene-orbit"
      role="slider"
      tabIndex={0}
      aria-label="Store scene orbit; drag or use left and right arrow keys to rotate"
      aria-valuemin={0}
      aria-valuemax={ORBIT_FRAMES - 1}
      aria-valuenow={frame}
      onPointerDown={(event) => { drag.current = { x: event.clientX, frame }; event.currentTarget.setPointerCapture(event.pointerId); }}
      onPointerMove={(event) => { if (drag.current) rotate(drag.current.frame + Math.round((event.clientX - drag.current.x) / 14)); }}
      onPointerUp={() => { drag.current = null; }}
      onPointerCancel={() => { drag.current = null; }}
      onKeyDown={(event) => {
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
          event.preventDefault();
          rotate(frame + (event.key === "ArrowRight" ? 1 : -1));
        }
      }}
    >
      <img src={`/media/orbit/frame-${String(frame).padStart(2, "0")}.webp`} alt="" draggable="false" />
      <div className="orbit-hud" aria-hidden="true"><span>STORE 01 · SCENE ORBIT</span><span>DRAG TO ROTATE ↔</span></div>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <span>{displayTime(label)} · CUSTOMER FLOW</span>
      <strong>{payload.find((item) => item.dataKey === "in")?.value?.toLocaleString() ?? 0} entries</strong>
      <strong>{payload.find((item) => item.dataKey === "out")?.value?.toLocaleString() ?? 0} exits</strong>
    </div>
  );
}

function HeadcountGauge({ count, label = "IN STORE" }) {
  const progress = Math.min((count ?? 0) / 24, 1) * 100;
  return (
    <div className="gauge-wrap" aria-label={count == null ? "Shopper count unavailable" : `${count} ${label.toLowerCase()} on a 24 shopper display scale`} role="img">
      <svg viewBox="0 0 240 145" className="gauge-svg" aria-hidden="true">
        <path d="M 24 120 A 96 96 0 0 1 216 120" fill="none" stroke="#303039" strokeWidth="11" strokeLinecap="round" />
        <path
          d="M 24 120 A 96 96 0 0 1 216 120"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="11"
          strokeLinecap="round"
          pathLength="100"
          strokeDasharray={`${progress} 100`}
          className="gauge-progress"
        />
        <defs>
          <linearGradient id="gaugeGradient">
            <stop stopColor="#9f83ff" />
            <stop offset="1" stopColor="#e5c9ff" />
          </linearGradient>
        </defs>
      </svg>
      <div className="gauge-value">
        <strong>{count ?? "—"}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

function Ring({ radius, value, color, width = 8 }) {
  const circumference = 2 * Math.PI * radius;
  return (
    <circle
      cx="100"
      cy="100"
      r={radius}
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeLinecap="round"
      strokeDasharray={`${circumference * value} ${circumference}`}
      transform="rotate(-90 100 100)"
      className="ring-progress"
    />
  );
}

function CheckoutRings({ queue, secondStatus, available = true }) {
  const secondActive = secondStatus === "active";
  return (
    <div className="ring-wrap" role="img" aria-label={available ? `Counter 1 queue ${queue}; Counter 2 ${secondStatus ?? "standby"}` : "Queue tracking unavailable"}>
      <svg viewBox="0 0 200 200" aria-hidden="true">
        <circle cx="100" cy="100" r="78" fill="none" stroke="#27272e" strokeWidth="8" />
        <circle cx="100" cy="100" r="59" fill="none" stroke="#27272e" strokeWidth="8" />
        <Ring radius={78} value={available ? Math.min(queue / 8, 1) : 0} color={queue >= 4 ? "#ff745f" : "#f5b45d"} />
        <Ring radius={59} value={available ? (secondActive ? 0.82 : 0.27) : 0} color="#52dfad" />
      </svg>
      <div className="ring-center">
        <strong>{available ? queue : "—"}</strong>
        <span>{available ? "IN QUEUE" : "NO QUEUE DATA"}</span>
      </div>
    </div>
  );
}

const GROUND_ZONES = [
  { id: "g-stair", title: "Stair", category: "To upper floor", detail: "Stair from the ground floor up to the upper sales floor.", left: 66, top: 4, width: 22, height: 16, layout: "aisle" },
  { id: "g-tvs", title: "Floor TVs", category: "Screens", detail: "Television row on the ground floor.", left: 22, top: 28, width: 66, height: 12, layout: "aisle" },
  { id: "g-cold", title: "Refrigeration", category: "Appliances", detail: "Drinks, larder, French, and American fridges.", left: 16, top: 46, width: 62, height: 12, layout: "aisle" },
  { id: "g-laundry", title: "Laundry", category: "Appliances", detail: "Washers, dryer, dishwasher, oven, and microwave.", left: 16, top: 62, width: 66, height: 12, layout: "aisle" },
  { id: "g-phones", title: "Phones", category: "Mobile", detail: "Phone display tables near the entrance.", left: 8, top: 78, width: 52, height: 12, layout: "aisle" },
  { id: "g-till", title: "Checkout", category: "Checkout", detail: "Till and card terminal beside the doors.", left: 66, top: 76, width: 24, height: 14, layout: "lane" },
  { id: "g-entry", title: "Entrance", category: "Threshold", detail: "Three shopfront doors on the ground floor.", left: 20, top: 92, width: 48, height: 6, layout: "entry-zone" },
];
const UPPER_ZONES = [
  { id: "u-theater", title: "Home theater", category: "TVs", detail: "Back-wall televisions, speakers, and the soundbar.", left: 6, top: 3, width: 62, height: 14, layout: "rear-wall" },
  { id: "u-stair", title: "Stair", category: "From ground", detail: "Stair landing from the ground floor.", left: 72, top: 4, width: 20, height: 14, layout: "aisle" },
  { id: "u-showroom", title: "Showroom", category: "Appliances", detail: "Appliance showroom behind the upper-floor wall.", left: 14, top: 24, width: 72, height: 16, layout: "aisle" },
  { id: "u-audio", title: "Accessories", category: "Audio", detail: "Headphone shelf, fold phones, and accessory shelves.", left: 10, top: 48, width: 64, height: 12, layout: "aisle" },
  { id: "u-phones", title: "Phones", category: "Mobile", detail: "Phone tables on the upper sales floor.", left: 12, top: 64, width: 56, height: 12, layout: "aisle" },
  { id: "u-computers", title: "Computers", category: "Computing", detail: "Laptops, monitors, tablets, and the printer desk.", left: 8, top: 80, width: 80, height: 12, layout: "aisle" },
];
const ELECTRONICS_FLOORS = { ground: GROUND_ZONES, upper: UPPER_ZONES };
const ELECTRONICS_ZONE_STATS = {
  "g-stair": { visits: 9, headcount: 1, dwell_hours: 0.01, density_pct: 12 },
  "g-tvs": { visits: 14, headcount: 1, dwell_hours: 0.06, density_pct: 48 },
  "g-cold": { visits: 7, headcount: 1, dwell_hours: 0.04, density_pct: 33 },
  "g-laundry": { visits: 8, headcount: 1, dwell_hours: 0.05, density_pct: 36 },
  "g-phones": { visits: 21, headcount: 2, dwell_hours: 0.09, density_pct: 70 },
  "g-till": { visits: 16, headcount: 4, dwell_hours: 0.07, density_pct: 80 },
  "g-entry": { visits: 28, headcount: 1, dwell_hours: 0.02, density_pct: 22 },
  "u-theater": { visits: 12, headcount: 2, dwell_hours: 0.08, density_pct: 55 },
  "u-stair": { visits: 9, headcount: 0, dwell_hours: 0.01, density_pct: 10 },
  "u-showroom": { visits: 6, headcount: 1, dwell_hours: 0.04, density_pct: 28 },
  "u-audio": { visits: 11, headcount: 1, dwell_hours: 0.05, density_pct: 42 },
  "u-phones": { visits: 15, headcount: 1, dwell_hours: 0.06, density_pct: 46 },
  "u-computers": { visits: 13, headcount: 2, dwell_hours: 0.07, density_pct: 58 },
};
const ELECTRONICS_TICKETS = [
  { id: "demo-phones", item: "Flagship phone display", category: "Mobile", location: "Ground phone tables", missing: 2, status: "Open", shelf: "g-phones" },
  { id: "demo-audio", item: "Headphone facing", category: "Audio", location: "Upper accessory shelf", missing: 1, status: "Open", shelf: "u-audio" },
];
const ELECTRONICS_SHOPPERS = [
  { id: "E-04", floor: "ground", x: 32, y: 82, zone: "g-phones", trail: [[36, 96], [34, 90], [32, 82]] },
  { id: "E-07", floor: "ground", x: 48, y: 50, zone: "g-cold", trail: [[40, 94], [44, 70], [48, 50]] },
  { id: "E-11", floor: "ground", x: 76, y: 82, zone: "g-till", trail: [[40, 84], [58, 80], [76, 82]] },
  { id: "E-02", floor: "ground", x: 50, y: 32, zone: "g-tvs", trail: [[30, 90], [38, 60], [50, 32]] },
  { id: "E-15", floor: "ground", x: 36, y: 66, zone: "g-laundry", trail: [[22, 94], [28, 78], [36, 66]] },
  { id: "E-18", floor: "ground", x: 74, y: 10, zone: "g-stair", trail: [[70, 78], [72, 40], [74, 10]] },
  { id: "E-21", floor: "upper", x: 28, y: 8, zone: "u-theater", trail: [[80, 10], [50, 8], [28, 8]] },
  { id: "E-22", floor: "upper", x: 40, y: 30, zone: "u-showroom", trail: [[78, 12], [55, 22], [40, 30]] },
  { id: "E-26", floor: "upper", x: 38, y: 52, zone: "u-audio", trail: [[30, 32], [34, 42], [38, 52]] },
  { id: "E-29", floor: "upper", x: 36, y: 68, zone: "u-phones", trail: [[38, 54], [36, 62], [36, 68]] },
  { id: "E-33", floor: "upper", x: 48, y: 84, zone: "u-computers", trail: [[22, 72], [34, 78], [48, 84]] },
];
const ELECTRONICS_HEAT_WEIGHTS = {
  "g-phones": 18, "g-till": 16, "g-tvs": 8, "g-cold": 6, "g-laundry": 5, "g-entry": 10, "g-stair": 4,
  "u-theater": 14, "u-computers": 12, "u-phones": 9, "u-audio": 7, "u-showroom": 5, "u-stair": 3,
};
const floorHeatmap = (zones, weights) => {
  const size = 12;
  const seconds = Array.from({ length: size * size }, (_, index) => {
    const x = ((index % size) + 0.5) / size * 100;
    const y = (Math.floor(index / size) + 0.5) / size * 100;
    return zones.reduce((total, zone) => (
      x >= zone.left && x <= zone.left + zone.width && y >= zone.top && y <= zone.top + zone.height ? total + (weights[zone.id] ?? 0) : total
    ), 0);
  });
  return { size, seconds };
};
const ELECTRONICS_HEATMAPS = {
  ground: floorHeatmap(GROUND_ZONES, ELECTRONICS_HEAT_WEIGHTS),
  upper: floorHeatmap(UPPER_ZONES, ELECTRONICS_HEAT_WEIGHTS),
};
const electronicsDemoReading = (day) => {
  const arrivals = [6, 8, 11, 9, 14, 12, 16, 10];
  const hourly = arrivals.map((count, index) => {
    const date = new Date(`${day}T${String(index + 10).padStart(2, "0")}:00:00`);
    return { time: date.toISOString(), in: count, out: Math.max(0, count - (index === arrivals.length - 1 ? 5 : 1)) };
  });
  return {
    mode: "simulation",
    measurement_source: "demo",
    updated_at: new Date().toISOString(),
    measurements: { visible_shoppers: true, footfall: true, queues: true, positions: true, shelf_gaps: true },
    live_occupancy: ELECTRONICS_SHOPPERS.length,
    occupancy: { count: ELECTRONICS_SHOPPERS.length, density_pct: 46, zones: ELECTRONICS_ZONE_STATS, heatmap: ELECTRONICS_HEATMAPS.ground },
    footfall: {
      in: hourly.reduce((sum, point) => sum + point.in, 0),
      out: hourly.reduce((sum, point) => sum + point.out, 0),
      hourly,
    },
    checkout: {
      counter_01: { queue_count: 4, wait_seconds: 96 },
      counter_02: { queue_count: 1, status: "active", response_delay_seconds: 18 },
    },
    queue_alerts: [{ time: new Date().toISOString(), count: 4 }],
    task_queue: ELECTRONICS_TICKETS,
    system_logs: [
      { index: 2, type: "QUEUE", message: "Checkout queue breaks threshold at the ground-floor till.", time: new Date().toISOString() },
      { index: 1, type: "STOCK", message: "Shelf gap on the ground-floor phone tables.", time: new Date().toISOString() },
    ],
    shoppers: ELECTRONICS_SHOPPERS,
    camera_feeds: {},
    playback: { status: "idle", scenario: null },
  };
};

function FloorMap({ queue, tickets, zones, shoppers, staffPosition, staffStatus, selectedZone, setSelectedZone, cameraCount, processed, zoneHeat, heatmap, heatmapMode = false, store = "grocery", level = "ground" }) {
  const electronicsPlan = store === "electronics";
  const congestion = queue >= 4;
  const activeStock = new Set(tickets.filter((ticket) => ticket.status === "Open" && ticket.missing > 0).map((ticket) => ticket.shelf));
  const heatPeak = Math.max(0, ...(heatmap?.seconds ?? []));
  return (
    <div className="floor-outer">
      <div className="plan-heading"><span>STORE FLOOR PLAN</span><span>{electronicsPlan ? (level === "upper" ? "UPPER FLOOR" : "GROUND FLOOR") : "TOP VIEW · SELECT A ZONE"}</span></div>
      <div className="floor-map" aria-label={electronicsPlan ? (level === "upper" ? "Electronics shop upper floor" : "Electronics shop ground floor") : "ConvenienceStore01 floor plan"}>
        <div className="map-grid" />
        {heatmap && heatmap.seconds?.length === heatmap.size ** 2 && heatPeak > 0 && <div className="map-heatmap" style={{ gridTemplateColumns: `repeat(${heatmap.size}, 1fr)` }} aria-hidden="true">{heatmap.seconds.map((seconds, index) => <span key={index} style={{ opacity: seconds > 0 ? Math.max(0.12, seconds / heatPeak) : 0 }} />)}</div>}
        <svg className="scene-outline" viewBox="0 0 1300 1200" preserveAspectRatio="none" aria-hidden="true">
          {electronicsPlan ? <><polyline points="40,1160 1260,1160 1260,40 40,40 40,1160" />{level === "upper" && <line x1="80" y1="430" x2="980" y2="430" />}</> : <polyline points="0,1200 1300,1200 1300,0 300,0 300,400 0,400 0,1200" />}
        </svg>
        <svg className="trajectory-layer" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {shoppers.map((shopper) => <polyline key={shopper.id} points={shopper.trail?.map(([x, y]) => `${x},${y}`).join(" ")} />)}
        </svg>
        <div className="rear-wall-label" aria-hidden="true">{electronicsPlan ? "BACK WALL" : "REAR WALL SHELF"}</div>
        {!electronicsPlan && <div className="map-corridor" aria-hidden="true">CUSTOMER AISLE</div>}
        {zones.map((zone) => {
          const alert = (zone.id === "lane1" || zone.id === "g-till") && congestion ? "danger" : activeStock.has(zone.id) ? "warning" : "normal";
          const kind = zone.category === "Checkout" ? "lane" : zone.id.endsWith("entry") ? "entry-zone" : "shelf";
          const density = zoneHeat?.[zone.id]?.density_pct ?? 0;
          const heat = processed && kind === "shelf" && alert === "normal" ? density >= 60 ? "heat-high" : density >= 30 ? "heat-medium" : "" : "";
          return (
            <button
              key={zone.id}
              type="button"
              className={`map-zone ${kind} ${zone.layout ?? ""} ${alert} ${heat} ${selectedZone === zone.id ? "selected" : ""}`}
              style={{
                left: `${zone.left}%`,
                top: `${zone.top}%`,
                width: `${zone.width}%`,
                height: `${zone.height}%`,
              }}
              onClick={() => setSelectedZone(zone.id)}
              aria-label={`${zone.title}, ${alert === "danger" ? "congested" : alert === "warning" ? "stock alert" : "normal"}`}
            >
              <span className="zone-icon">{kind === "shelf" ? <span className="shelf-stripes" /> : <ShoppingBag size={14} />}</span>
              <span className="zone-name">{electronicsPlan ? zone.title : kind === "shelf" ? zone.layout && zone.layout !== "aisle" ? zone.category : `${zone.category} · ${zone.title.replace("Shelf Row ", "")}` : zone.title}</span>
              <span className="zone-detail">{zone.category}</span>
            </button>
          );
        })}
        {shoppers.map((shopper) => <span key={shopper.id} className="shopper-dot" style={{ left: `${shopper.x}%`, top: `${shopper.y}%` }} title={`${shopper.id} · ${shopper.zone}`} />)}
        {!electronicsPlan && <div className="backroom-door" style={{ left: "26%", top: "42%" }}>
          <DoorOpen size={16} />
          <span>STAFF ACCESS</span>
        </div>}
        {processed && staffPosition && <div className={`staff-marker ${staffStatus !== "standby" ? "dispatched" : ""}`} style={{ left: `${staffPosition.x}%`, top: `${staffPosition.y}%` }} title={`Staff_02 ${staffStatus}`}>
          <span className="staff-pulse" />
          <Users size={14} />
          <span className="staff-label">STAFF_02</span>
        </div>}
        {!electronicsPlan && <div className="map-coordinate">{processed ? `${cameraCount} CAMERA FEEDS` : "NO TRACKING DATA"} · STORE MAP</div>}
      </div>
      <div className="map-footer">
        {heatmapMode ? <div className="heat-legend" aria-label={heatPeak > 0 ? "Heatmap intensity from less shopper time to more shopper time" : "Heatmap awaiting tracked shopper positions"}><span>{heatPeak > 0 ? "LESS TIME" : "AWAITING TRACKS"}</span>{heatPeak > 0 && <><span className="heat-legend-scale" aria-hidden="true" /><span>MORE TIME</span></>}</div> : <div><i className="dot green" /> Shopper track <i className="dot amber" /> Stock alert <i className="dot orange" /> Queue congestion</div>}
        <span>
          Click a zone to inspect <ArrowRight size={13} />
        </span>
      </div>
    </div>
  );
}

export default function Dashboard({ initialTelemetry = null, initialSite = "grocery" }) {
  const [telemetry, setTelemetry] = useState(initialTelemetry);
  const [live, setLive] = useState(true);
  const [connection, setConnection] = useState("connecting");
  const [actionError, setActionError] = useState("");
  const [pendingTicket, setPendingTicket] = useState(null);
  const [selectedZone, setSelectedZone] = useState(() => {
    if (typeof window !== "undefined" && window.sessionStorage?.getItem("retail-site") === "electronics") return "g-phones";
    return initialSite === "electronics" ? "g-phones" : "lane1";
  });
  const [mapView, setMapView] = useState("tracks");
  const [shopFloor, setShopFloor] = useState("ground");
  const [activeNav, setActiveNav] = useState(currentPage);
  const [mobileNav, setMobileNav] = useState(false);
  const [videoScenarios, setVideoScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("final_test");
  const [videoError, setVideoError] = useState("");
  const [uploadingVideo, setUploadingVideo] = useState("");
  const [reviewRequest, setReviewRequest] = useState(null);
  const [selectedDate, setSelectedDate] = useState(() => new Date().toLocaleDateString("sv-SE"));
  const [site, setSite] = useState(() => {
    if (typeof window !== "undefined" && window.sessionStorage) {
      const stored = window.sessionStorage.getItem("retail-site");
      if (stored === "electronics" || stored === "grocery") return stored;
    }
    return initialSite === "electronics" ? "electronics" : "grocery";
  });
  const logRef = useRef(null);

  useEffect(() => {
    const onHashChange = () => { setActiveNav(currentPage()); setMobileNav(false); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (!mobileNav) return undefined;
    const closeOnEscape = (event) => { if (event.key === "Escape") setMobileNav(false); };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [mobileNav]);

  const refreshVideos = async () => {
    const response = await fetch("/api/videos");
    if (!response.ok) throw new Error(`Video inventory unavailable (${response.status})`);
    setVideoScenarios((await response.json()).scenarios);
  };

  useEffect(() => {
    refreshVideos().catch((error) => setVideoError(error.message));
  }, []);

  const runScenario = async (scenarioId) => {
    setUploadingVideo("");
    setVideoError("");
    setActionError("");
    try {
      const response = await fetch(`/api/scenarios/${scenarioId}/run`, { method: "POST" });
      if (!response.ok) throw new Error((await response.json()).detail || `Run failed (${response.status})`);
    } catch (error) {
      setVideoError(error.message);
    }
  };

  const uploadVideo = async (scenarioId, cameraId, file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".mp4") || (file.type && file.type !== "video/mp4") || file.size > 500 * 1024 * 1024) {
      setVideoError("Choose an MP4 file no larger than 500 MB.");
      return;
    }
    setUploadingVideo(`${scenarioId}/${cameraId}`);
    setVideoError("");
    try {
      const response = await fetch(`/api/videos/${scenarioId}/${cameraId}`, {
        method: "PUT",
        headers: { "Content-Type": "video/mp4" },
        body: file,
      });
      if (!response.ok) throw new Error((await response.json()).detail || `Upload failed (${response.status})`);
      await refreshVideos();
    } catch (error) {
      setVideoError(error.message);
    } finally {
      setUploadingVideo("");
    }
  };

  useEffect(() => {
    if (!live) {
      setConnection("paused");
      return undefined;
    }
    const source = new EventSource("/api/stream");
    source.onmessage = (message) => {
      try {
        setTelemetry(JSON.parse(message.data));
        setConnection("live");
      } catch {
        setConnection("error");
      }
    };
    source.onerror = () => setConnection("reconnecting");
    return () => source.close();
  }, [live]);

  const reading = site === "electronics" ? electronicsDemoReading(selectedDate) : telemetry;
  const tickets = reading?.inventory_tickets ?? reading?.task_queue ?? [];
  const logs = reading?.system_logs ?? reading?.events ?? [];
  const queueAlerts = reading?.queue_alerts ?? logs.filter((item) => item.type === "QUEUE" && item.message?.includes("breaks threshold"));
  const zones = reading?.zones ?? [];
  const traffic = reading?.footfall?.hourly ?? [];
  const datedTraffic = traffic.filter((point) => new Date(point.time).toLocaleDateString("sv-SE") === selectedDate);
  const datedEntries = datedTraffic.reduce((sum, point) => sum + (point.in ?? 0), 0);
  const datedExits = datedTraffic.reduce((sum, point) => sum + (point.out ?? 0), 0);
  const queue = reading?.checkout?.counter_01?.queue_count ?? 0;
  const counterTwo = reading?.checkout?.counter_02;
  const shoppers = reading?.shoppers ?? [];
  const cameraFeeds = reading?.camera_feeds ?? {};
  const cameraCount = Object.values(cameraFeeds).filter(Boolean).length;
  const idle = !reading || reading.mode === "idle";
  const playback = reading?.playback ?? { status: "idle", scenario: null };
  const annotationReplay = reading?.measurement_source === "annotation_replay";
  const isaacReference = reading?.measurement_source === "isaac_ground_truth";
  const modelInference = reading?.measurement_source === "model_inference";
  const measurements = reading?.measurements ?? {};
  const visibleAvailable = Boolean(measurements.visible_shoppers);
  const footfallAvailable = Boolean(measurements.footfall);
  const queueAvailable = Boolean(measurements.queues);
  const shelfAvailable = Boolean(measurements.shelf_gaps);
  const processed = Boolean(measurements.positions) && (reading?.mode === "vision" || site === "electronics");
  const operationalReportAvailable = footfallAvailable || queueAvailable || shelfAvailable || processed;
  const managerScenarios = videoScenarios;
  const selectedVideoScenario = videoScenarios.find((scenario) => scenario.id === selectedScenario);
  const shelfZones = zones.filter((zone) => !["Checkout", "Threshold"].includes(zone.category));

  useEffect(() => { logRef.current?.scrollTo({ top: 0, behavior: "smooth" }); }, [logs[0]?.index]);

  const updateTicket = async (ticket) => {
    const status = nextStatus[ticket.status];
    if (!status || pendingTicket) return;
    if (site === "electronics") {
      setActionError("These shelf tickets stay on this page. They are not sent to the grocery task service.");
      return;
    }
    setPendingTicket(ticket.id);
    setActionError("");
    try {
      const response = await fetch(`/api/tasks/${encodeURIComponent(ticket.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      if (!response.ok) throw new Error(`Task update failed (${response.status})`);
    } catch (error) {
      setActionError(error.message);
    } finally {
      setPendingTicket(null);
    }
  };

  const currentZone = zones.find((zone) => zone.id === selectedZone);
  const zoneTickets = shelfAvailable ? tickets.filter((ticket) => ticket.shelf === selectedZone && ticket.status !== "Refilled") : [];
  const zoneStats = processed ? reading?.occupancy?.zones?.[selectedZone] : null;
  const electronicsZones = ELECTRONICS_FLOORS[shopFloor];
  const electronicsZone = electronicsZones.find((zone) => zone.id === selectedZone) ?? electronicsZones[0];
  const electronicsZoneTickets = ELECTRONICS_TICKETS.filter((ticket) => ticket.shelf === electronicsZone.id);
  const electronicsFloorShoppers = ELECTRONICS_SHOPPERS.filter((shopper) => shopper.floor === shopFloor);
  const navigate = (name) => {
    setActiveNav(name);
    setMobileNav(false);
    window.location.hash = Object.keys(pages).find((key) => pages[key] === name);
    window.scrollTo(0, 0);
  };
  const chooseSite = (next) => {
    setSite(next);
    if (typeof window !== "undefined" && window.sessionStorage) window.sessionStorage.setItem("retail-site", next);
    if (next === "electronics") {
      setShopFloor("ground");
      setSelectedZone("g-phones");
      navigate("Reports");
    } else {
      setSelectedZone("lane1");
    }
  };
  const electronics = site === "electronics";
  const openEvidence = (evidence) => {
    setSelectedScenario(telemetry?.current_scenario ?? selectedScenario);
    setReviewRequest({ ...evidence, requestId: Date.now() });
    navigate("Cameras");
  };
  const downloadReport = () => {
    const report = {
      site: site === "electronics" ? "ElectronicsShop" : "ConvenienceStore01",
      scenario: site === "electronics" ? "demo" : (telemetry?.current_scenario ?? selectedScenario),
      run_id: site === "electronics" ? null : (telemetry?.run_id ?? null), generated_at: new Date().toISOString(),
      queue_alerts: queueAlerts,
      source: site === "electronics" ? "demo" : (telemetry?.measurement_source ?? "none"), measurements,
      visible_shoppers: visibleAvailable ? reading.live_occupancy : null,
      footfall: footfallAvailable ? reading.footfall : null,
      checkout: queueAvailable ? reading.checkout : null,
      zone_density: processed ? reading.occupancy.zones : null,
      heatmap: processed ? (site === "electronics" ? ELECTRONICS_HEATMAPS : reading.occupancy.heatmap) : null,
      shelf_tickets: shelfAvailable ? tickets : null,
      events: logs,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `retail-vision-${report.scenario}-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="app-shell bg-neutral-950">
      {mobileNav && <button type="button" className="nav-backdrop" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
      <aside id="primary-navigation" className={`sidebar ${mobileNav ? "mobile-open" : ""}`} aria-label="Primary navigation">
        <div className="brand-mark" aria-label="Cogno Valley Labs">
          <ScanLine size={23} strokeWidth={2.3} />
        </div>
        <div className="side-divider" />
        <nav className="nav-list">
          {[
            { name: "Overview", icon: LayoutDashboard },
            { name: "Cameras", icon: Camera },
            { name: "Reports", icon: BarChart3 },
            { name: "Task Queues", icon: PackageCheck },
            { name: "System Logs", icon: Activity },
          ].map(({ name, icon: Icon }) => (
            <button key={name} className={`nav-button ${activeNav === name ? "active" : ""}`} onClick={() => navigate(name)} aria-label={name === "Cameras" ? "Video Analysis" : name === "Reports" ? "Spatial Analysis and Reports" : name} title={name === "Cameras" ? "Video Analysis" : name === "Reports" ? "Spatial Analysis and Reports" : name} aria-current={activeNav === name ? "page" : undefined}>
              <Icon size={19} strokeWidth={1.8} />
              <span>{name === "Cameras" ? "Video Analysis" : name === "Reports" ? "Spatial Analysis & Reports" : name}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="side-divider" />
          <div className="operator-avatar">CV</div>
          <span>OPS</span>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumb">
            <button className="mobile-menu" aria-label={mobileNav ? "Close navigation" : "Open navigation"} aria-expanded={mobileNav} aria-controls="primary-navigation" onClick={() => setMobileNav(!mobileNav)}>
              {mobileNav ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span className="track-wordmark">
              COGNO <span>VALLEY</span>
            </span>
            <span className="bread-separator">/</span>
            <div className="site-switch" role="group" aria-label="Store">
              <button type="button" aria-pressed={site === "grocery"} onClick={() => chooseSite("grocery")}>Grocery</button>
              <button type="button" aria-pressed={electronics} onClick={() => chooseSite("electronics")}>Electronics</button>
            </div>
            <span>Retail Vision Analytics</span>
            <ChevronDown size={14} />
          </div>
          <div className="topbar-right">
            <span className="topbar-clock">
              <Clock3 size={14} />{" "}
              {new Date().toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </span>
            <span className="topbar-divider" />
            <button className={`live-control ${connection === "live" ? "" : "paused"}`} onClick={() => setLive((value) => !value)} aria-label={live ? "Pause telemetry stream" : "Resume telemetry stream"}>
              {live ? <Radio size={15} /> : <Play size={14} />} {connection === "live" && reading?.mode === "simulation" && !electronics ? "DEMO" : connection === "live" && isaacReference ? "REFERENCE" : connection === "live" && annotationReplay ? "REPLAY" : connection.toUpperCase()}
            </button>
            <div className="top-avatar">CV</div>
          </div>
        </header>

        <div className="content-wrap">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" /> {electronics ? "COMMAND CENTER / ELECTRONICS" : "COMMAND CENTER / SITE 01"}
              </div>
              <h1>{({ Overview: "Store overview", Reports: "Spatial analysis & reports", Cameras: "Video analysis", "Task Queues": "Shelf operations", "System Logs": "System activity" })[activeNav]}<span className="heading-dot">.</span></h1>
              <p>{({ Overview: "Shopper flow, shelf health, and checkout service at a glance.", Reports: "Store movement, zone activity, and operational findings.", Cameras: electronics ? "The shop model is beside the floor plan." : "Upload, analyze, and review the five camera angles.", "Task Queues": "Assign and close shelf replenishment tasks.", "System Logs": "Time-stamped vision events and operational alerts." })[activeNav]}</p>
              {actionError && (
                <p role="alert" className="action-error">
                  {actionError}
                </p>
              )}
            </div>
          </div>
          {telemetry?.mode === "simulation" && !electronics && activeNav !== "Cameras" && <div className="data-source-banner" role="status"><strong>DEMO DATA</strong><span>Figures are simulated, not measured from cameras.</span></div>}
          {annotationReplay && activeNav !== "Cameras" && <div className="data-source-banner" role="status"><strong>LIMITED VIDEO ANALYSIS</strong><span>Camera-visible shopper count only. Other readings are unavailable.</span></div>}
          {isaacReference && activeNav !== "Cameras" && <div className="data-source-banner" role="status"><strong>ISAAC REFERENCE</strong><span>Counts and events come from simulator annotations.</span></div>}

          {activeNav === "Overview" && <section className="analytics-grid" aria-label="Store analytics">
            <article className={`${panel} headcount-card`}>
              <div className="card-heading">
                <div>
                  <div className="section-kicker">01 / OCCUPANCY</div>
                  <h2>{idle ? "Headcount" : !electronics && reading?.mode === "simulation" ? "Simulated headcount" : isaacReference ? "Reference headcount" : !processed ? "Visible shoppers · estimate" : "Live headcount"}</h2>
                </div>
                <span className="card-icon">
                  <Users size={17} />
                </span>
              </div>
              <HeadcountGauge count={visibleAvailable ? reading?.occupancy?.count ?? reading?.live_occupancy ?? 0 : null} label={!processed ? "CAMERA VISIBLE" : "IN STORE"} />
              <div className="metric-bottom">
                <span>
                  <i className={`dot ${site === "electronics" || connection === "live" ? "green" : "amber"}`} /> {site === "electronics" ? `${reading?.occupancy?.density_pct ?? 0}% store density` : annotationReplay ? "Maximum count in one camera" : !visibleAvailable ? "Awaiting shopper detections" : !processed ? "Visible camera count · no unique tracks" : connection === "live" ? `${reading?.occupancy?.density_pct ?? 0}% store density` : "Awaiting telemetry stream"}
                </span>
                <span className="metric-cap">
                  DISPLAY SCALE <strong>24</strong>
                </span>
              </div>
            </article>
            <article className={`${panel} footfall-card`}>
              <div className="card-heading">
                <div>
                  <div className="section-kicker">02 / CUSTOMER MOVEMENT</div>
                  <h2>{!electronics && reading?.mode === "simulation" ? "Simulated store footfall" : "Store footfall"}</h2>
                </div>
                <label className="chart-period">Date <input type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} aria-label="Footfall date" /></label>
              </div>
              <div className="chart-summary">
                <strong>{footfallAvailable && datedTraffic.length ? datedEntries.toLocaleString() : "—"}<small>entries</small></strong>
                <span className="footfall-out">{footfallAvailable && datedTraffic.length ? datedExits.toLocaleString() : "—"} exits</span>
              </div>
              <div className="chart-wrap">
                {footfallAvailable && datedTraffic.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={datedTraffic} margin={{ top: 13, right: 4, left: -30, bottom: 2 }}>
                      <defs>
                        <linearGradient id="trafficFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#aa87f5" stopOpacity={0.27} />
                          <stop offset="95%" stopColor="#aa87f5" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid vertical={false} stroke="#2b2a34" strokeDasharray="3 5" />
                      <XAxis
                        dataKey="time"
                        tickFormatter={displayHour}
                        axisLine={false}
                        tickLine={false}
                        tick={{
                          fill: "#777683",
                          fontSize: 10,
                          fontFamily: "IBM Plex Mono",
                        }}
                        dy={11}
                      />
                      <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{
                          fill: "#777683",
                          fontSize: 10,
                          fontFamily: "IBM Plex Mono",
                        }}
                        allowDecimals={false}
                      />
                      <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#bda5ff", strokeDasharray: "4 4" }} />
                      <Area
                        type="monotone"
                        dataKey="in"
                        stroke="#c5a9ff"
                        strokeWidth={2.6}
                        fill="url(#trafficFill)"
                        activeDot={{
                          r: 6,
                          fill: "#dfcfff",
                          stroke: "#a479e7",
                          strokeWidth: 4,
                        }}
                      />
                      <Area type="monotone" dataKey="out" stroke="#69d9b2" strokeWidth={1.8} fill="none" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart-empty">{footfallAvailable ? "No recorded crossings for this date" : "Entry and exit counts unavailable"}</div>
                )}
              </div>
              <div className="chart-key"><span><i className="dot violet" /> Entries</span><span><i className="dot green" /> Exits</span><span>{footfallAvailable && datedTraffic.length ? `${displayHour(datedTraffic[0].time)}–${displayHour(datedTraffic[datedTraffic.length - 1].time)}` : "Selected date"}</span></div>
            </article>
            <article className={`${panel} checkout-card`}>
              <div className="card-heading">
                <div>
                  <div className="section-kicker">03 / FRONTLINE</div>
                  <h2>Checkout pulse</h2>
                </div>
                <span className="card-icon">
                  <Activity size={17} />
                </span>
              </div>
              <div className="checkout-body">
                <CheckoutRings queue={queueAvailable ? queue : 0} secondStatus={queueAvailable ? counterTwo?.status : null} available={queueAvailable} />
                <div className="counter-legend">
                  <div>
                    <i className="dot orange" />
                    <span>
                      COUNTER 01
                      <small>{!queueAvailable ? "No queue tracks" : queue >= 4 ? "Queue overflow" : modelInference ? "Wait time unavailable" : `${reading?.checkout?.counter_01?.wait_seconds ?? 0}s longest wait`}</small>
                    </span>
                    <strong>{!queueAvailable ? "—" : queue >= 4 ? "BUSY" : "OPEN"}</strong>
                  </div>
                  <div>
                    <i className="dot green" />
                    <span>
                      COUNTER 02<small>{!queueAvailable ? "No queue tracks" : modelInference ? `${counterTwo?.queue_count ?? 0} queued · response time unavailable` : `${counterTwo?.queue_count ?? 0} queued · ${counterTwo?.response_delay_seconds ?? 0}s response`}</small>
                    </span>
                    <strong>{!queueAvailable ? "—" : counterTwo?.status === "active" ? "ACTIVE" : counterTwo?.status === "dispatching" ? "EN ROUTE" : "IDLE"}</strong>
                  </div>
                </div>
              </div>
            </article>
          </section>}

          {activeNav === "Cameras" && !electronics && <section className={`${panel} video-inputs-panel`} id="video-inputs" aria-label="Video inputs">
            <div className="card-heading">
              <div>
                <div className="section-kicker">VIDEO INPUTS</div>
                <h2>Final test video <span className="title-count">{managerScenarios.reduce((total, scenario) => total + scenario.cameras.filter((camera) => camera.source).length, 0)} / 5</span></h2>
              </div>
              <button type="button" className="scenario-run" aria-label={`Run ${selectedScenario}`} disabled={playback.status === "running" || !selectedVideoScenario?.can_run} onClick={() => runScenario(selectedScenario)}>
                <Play size={13} aria-hidden="true" /> {playback.status === "running" && playback.scenario === selectedScenario ? "ANALYZING" : selectedVideoScenario?.can_run ? "RUN ANALYSIS" : "UNAVAILABLE"}
              </button>
            </div>
            {playback.status === "running" && playback.scenario === selectedScenario && <p className="video-input-explainer" role="status">
              {telemetry?.playback_seconds > 0
                ? `Predictions streaming · ${clipTime(telemetry.playback_seconds)} / ${clipTime(selectedVideoScenario?.duration_seconds ?? 120)}`
                : "Processing all five camera videos. Measured results appear when inference finishes; this can take several minutes."}
            </p>}
            {selectedVideoScenario?.run_issue && <p className="video-input-explainer" role="status">{selectedVideoScenario.run_issue}</p>}
            {playback.status === "error" && <p className="action-error" role="alert">Analysis stopped. Check the server analysis log.</p>}
            {videoError && <p className="action-error" role="alert">{videoError}</p>}
            <div className="video-scenarios">
              {videoScenarios.filter((scenario) => scenario.id === selectedScenario).map((scenario) => (
                <div className="video-scenario" key={scenario.id}>
                  <div className="video-scenario-heading"><strong>{scenarioName(scenario)}</strong><span>{scenario.cameras.filter((camera) => camera.source).length} / 5 AVAILABLE</span></div>
                  {scenario.cameras.map((camera) => {
                    const key = `${scenario.id}/${camera.id}`;
                    return <div className="video-camera-row" key={camera.id}>
                      <Camera size={14} aria-hidden="true" />
                      <span className="video-camera-name">{camera.label}</span>
                      <span className={`video-camera-status ${camera.source ? "ready" : "missing"}`}>{camera.source ? `${camera.source === "uploaded" ? "UPLOADED" : "FINAL TEST"} · ${(camera.bytes / 1048576).toFixed(1)} MB` : "NO VIDEO"}</span>
                      <label className={`video-upload ${uploadingVideo ? "disabled" : ""}`}>
                        <Upload size={13} aria-hidden="true" /> {uploadingVideo === key ? "UPLOADING" : camera.source ? "REPLACE" : "UPLOAD"}
                        <input type="file" accept="video/mp4,.mp4" disabled={Boolean(uploadingVideo)} aria-label={`Upload ${scenario.label} ${camera.label} MP4`} onChange={(event) => { uploadVideo(scenario.id, camera.id, event.target.files?.[0]); event.target.value = ""; }} />
                      </label>
                    </div>;
                  })}
                </div>
              ))}
            </div>
          </section>}

          {activeNav === "Cameras" && !electronics && <CameraWorkspace scenario={selectedVideoScenario} selectedScenario={selectedScenario} playback={playback} playbackSeconds={telemetry?.playback_seconds ?? 0} reviewRequest={reviewRequest} evidenceEvents={selectedScenario === telemetry?.current_scenario ? logs.filter((item) => item.evidence?.camera_id && Number.isFinite(item.evidence?.video_seconds) && selectedVideoScenario?.cameras.some((camera) => camera.id === item.evidence.camera_id)) : []} />}

          {activeNav === "Cameras" && !electronics && <section className={`${panel} orbit-review`} aria-label="Interactive store scene preview">
            <div className="card-heading"><div><div className="section-kicker">STORE REFERENCE</div><h2>3D scene orbit</h2></div><span className="workspace-source">DRAG · KEYBOARD ARROWS</span></div>
            <div className="scene-video-frame"><SceneOrbit /><div className="video-corner top-left" aria-hidden="true" /><div className="video-corner top-right" aria-hidden="true" /><div className="video-corner bottom-left" aria-hidden="true" /><div className="video-corner bottom-right" aria-hidden="true" /></div>
            <div className="scene-video-caption"><ScanLine size={14} /> Store scene preview <span>Reference view</span></div>
          </section>}

          {activeNav === "Cameras" && electronics && <section className={`${panel} orbit-review`} aria-label="Electronics scene location">
            <div className="card-heading"><div><div className="section-kicker">STORE REFERENCE</div><h2>Scene on the floor plan</h2></div></div>
            <p className="video-input-explainer">The electronics shop model is on the same page as the floor plan.</p>
            <button type="button" className="scenario-run" onClick={() => navigate("Reports")}>Open floor plan</button>
          </section>}

          {activeNav === "Reports" && electronics && <section className="middle-grid spatial-analysis electronics-spatial" id="floorplan" aria-label="Electronics floor plan and scene">
            <article className={`${panel} orbit-review electronics-orbit`}>
              <div className="card-heading"><div><div className="section-kicker">STORE REFERENCE</div><h2>3D scene orbit</h2></div><span className="workspace-source">DRAG TO ROTATE · TWO FLOORS</span></div>
              <div className="scene-video-frame electronics-model"><ElectronicsOrbit /><div className="video-corner top-left" aria-hidden="true" /><div className="video-corner top-right" aria-hidden="true" /><div className="video-corner bottom-left" aria-hidden="true" /><div className="video-corner bottom-right" aria-hidden="true" /></div>
              <div className="scene-video-caption"><ScanLine size={14} /> Electronics shop <span>Live model</span></div>
            </article>
            <article className={`${panel} map-panel`}>
              <div className="card-heading map-title">
                <div>
                  <div className="section-kicker">STORE GEOMETRY</div>
                  <h2>{mapView === "heatmap" ? "Shopper dwell heatmap" : shopFloor === "upper" ? "Upper floor" : "Ground floor"}</h2>
                </div>
                <div className="map-controls">
                  <div className="map-view-switch" role="group" aria-label="Shop floor">
                    <button type="button" aria-pressed={shopFloor === "ground"} onClick={() => { setShopFloor("ground"); setSelectedZone("g-phones"); }}>Ground</button>
                    <button type="button" aria-pressed={shopFloor === "upper"} onClick={() => { setShopFloor("upper"); setSelectedZone("u-theater"); }}>Upper</button>
                  </div>
                  <div className="map-view-switch" role="group" aria-label="Floor plan display">
                    <button type="button" aria-pressed={mapView === "tracks"} onClick={() => setMapView("tracks")}>TRACKS</button>
                    <button type="button" aria-pressed={mapView === "heatmap"} onClick={() => setMapView("heatmap")}>HEATMAP</button>
                  </div>
                  <div className="map-status"><span className="status-spark" /> TRACKS · {electronicsFloorShoppers.length}</div>
                </div>
              </div>
              <div className="spatial-layout">
                <FloorMap queue={queue} tickets={ELECTRONICS_TICKETS} zones={electronicsZones} shoppers={mapView === "tracks" ? electronicsFloorShoppers : []} staffPosition={null} staffStatus="standby" selectedZone={selectedZone} setSelectedZone={setSelectedZone} cameraCount={0} processed zoneHeat={ELECTRONICS_ZONE_STATS} heatmap={mapView === "heatmap" ? ELECTRONICS_HEATMAPS[shopFloor] : null} heatmapMode={mapView === "heatmap"} store="electronics" level={shopFloor} />
              </div>
            </article>
            <aside className="side-stack">
              <article className={`${panel} response-panel`}>
                <div className="card-heading">
                  <div>
                    <div className="section-kicker">ZONE DETAIL</div>
                    <h2>{electronicsZone.title}</h2>
                  </div>
                  <MapPinned size={18} className="muted-icon" />
                </div>
                <div className="response-hero">
                  <span className={electronicsZone.id === "g-till" && queue >= 4 ? "risk" : electronicsZoneTickets.length ? "attention" : "clear"}>
                    {electronicsZone.id === "g-till" && queue >= 4 ? "HIGH PRIORITY" : electronicsZoneTickets.length ? "ATTENTION NEEDED" : "ALL CLEAR"}
                  </span>
                  <h3>
                    {electronicsZone.id === "g-till" && queue >= 4
                      ? "Queue threshold exceeded"
                      : electronicsZoneTickets.length
                        ? `${electronicsZoneTickets.length} active inventory ${electronicsZoneTickets.length === 1 ? "ticket" : "tickets"}`
                        : electronicsZone.category}
                  </h3>
                  <p>
                    {electronicsZone.id === "g-till" && queue >= 4
                      ? "Queue of 4 at the ground-floor till. Backup counter is active."
                      : electronicsZoneTickets.length
                        ? `${electronicsZoneTickets.map((ticket) => ticket.item).join(", ")} requires attention.`
                        : electronicsZone.detail}
                  </p>
                </div>
                <div className="zone-metrics">
                  <div><span>HEADCOUNT</span><strong>{ELECTRONICS_ZONE_STATS[electronicsZone.id]?.headcount ?? "—"}</strong></div>
                  <div><span>DENSITY</span><strong>{ELECTRONICS_ZONE_STATS[electronicsZone.id] ? `${ELECTRONICS_ZONE_STATS[electronicsZone.id].density_pct}%` : "—"}</strong></div>
                  <div><span>DWELL</span><strong>{ELECTRONICS_ZONE_STATS[electronicsZone.id] ? `${ELECTRONICS_ZONE_STATS[electronicsZone.id].dwell_hours.toFixed(3)} h` : "—"}</strong></div>
                </div>
                <div className="zone-alerts">
                  {electronicsZoneTickets.length ? electronicsZoneTickets.map((ticket) => `${ticket.item}: ${ticket.missing} missing`).join(" · ") : "No active shelf health alerts"}
                </div>
                <div className="response-divider" />
                <div className="response-meta">
                  <span>SOURCE</span>
                  <strong>SHOP MODEL</strong>
                </div>
              </article>
            </aside>
          </section>}

          {activeNav === "Reports" && electronics && <section className="analysis-report" aria-label="Electronics activity report">
            <div className="analysis-report-heading"><span>CURRENT SESSION · ELECTRONICS</span><div><button type="button" className="export-report" onClick={downloadReport}>EXPORT REPORT</button></div></div>
            <article className={`${panel} report-card`}>
              <div className="section-kicker">01 / ENTRY TRACKING</div>
              <h2>Entry crossings</h2>
              <div className="report-pair"><div><span>IN</span><strong>{reading.footfall.in}</strong></div><div><span>OUT</span><strong>{reading.footfall.out}</strong></div></div>
              <p>{reading.live_occupancy} shoppers currently inside, across both floors.</p>
            </article>
            <article className={`${panel} report-card`}>
              <div className="section-kicker">02 / CHECKOUT TRACKING</div>
              <h2>Checkout service</h2>
              <div className="report-pair"><div><span>COUNTER 01</span><strong>{queue}</strong><small>{reading.checkout.counter_01.wait_seconds}s longest wait</small></div><div><span>COUNTER 02</span><strong>{counterTwo?.queue_count ?? 0}</strong><small>{counterTwo?.status ?? "standby"} · {counterTwo?.response_delay_seconds ?? 0}s response</small></div></div>
              <p>{queue >= 4 ? "Queue threshold reached · backup needed" : "Queues within monitored limit"}</p>
            </article>
            <article className={`${panel} report-card zone-engagement-card`}>
              <div className="section-kicker">03 / SHOPPER MOVEMENT</div>
              <h2>Zone engagement</h2>
              <div className="zone-engagement-head"><span>ZONE</span><span>VISITS</span><span>DWELL</span><span>NOW</span></div>
              {[...GROUND_ZONES, ...UPPER_ZONES].filter((zone) => !["Checkout", "Threshold"].includes(zone.category)).map((zone) => {
                const stats = ELECTRONICS_ZONE_STATS[zone.id];
                return <div className="zone-engagement-row" key={zone.id}><span>{zone.id.startsWith("u-") ? "Upper" : "Ground"} · {zone.title}</span><strong>{stats?.visits ?? "—"}</strong><strong>{stats ? formatDwell(stats.dwell_hours) : "—"}</strong><strong>{stats?.headcount ?? "—"}</strong></div>;
              })}
            </article>
            <article className={`${panel} report-card shelf-findings-card`}>
              <div className="section-kicker">04 / REPLENISHMENT</div>
              <h2>Shelf findings</h2>
              {ELECTRONICS_TICKETS.map((ticket) => <div className="shelf-finding" key={ticket.id}><span><strong>{ticket.item}</strong><small>{ticket.category} · {ticket.location}</small></span><span className={`status-badge ${ticket.status.toLowerCase()}`}>{ticket.missing} missing · {ticket.status}</span></div>)}
              <button type="button" className="report-link" onClick={() => navigate("Task Queues")}>Open shelf tasks <ArrowRight size={14} /></button>
            </article>
          </section>}

          {activeNav === "Reports" && !electronics && <section className="middle-grid spatial-analysis" id="floorplan" aria-label="Spatial analysis and zone detail">
            <article className={`${panel} map-panel`}>
              <div className="card-heading map-title">
                <div>
                  <div className="section-kicker">STORE GEOMETRY</div>
                  <h2>{mapView === "heatmap" ? "Shopper dwell heatmap" : "Store floor plan"}</h2>
                </div>
                <div className="map-controls">
                  <div className="map-view-switch" role="group" aria-label="Floor plan display">
                    <button type="button" aria-pressed={mapView === "tracks"} onClick={() => setMapView("tracks")}>TRACKS</button>
                    <button type="button" aria-pressed={mapView === "heatmap"} onClick={() => setMapView("heatmap")}>HEATMAP</button>
                  </div>
                  <div className={`map-status ${!processed ? "simulated" : ""}`}><span className="status-spark" /> {processed ? playback.status === "complete" ? "ANALYSIS COMPLETE" : "TRACKING ACTIVE" : "NO TRACKING DATA"} · {cameraCount} / {Object.keys(cameraFeeds).length}</div>
                </div>
              </div>
              <div className="spatial-layout">
                <FloorMap queue={queueAvailable ? queue : 0} tickets={shelfAvailable ? tickets : []} zones={zones} shoppers={processed && mapView === "tracks" ? shoppers : []} staffPosition={modelInference ? null : telemetry?.staff_02} staffStatus={counterTwo?.status ?? "standby"} selectedZone={selectedZone} setSelectedZone={setSelectedZone} cameraCount={cameraCount} processed={processed} zoneHeat={telemetry?.occupancy?.zones} heatmap={processed && mapView === "heatmap" ? telemetry?.occupancy?.heatmap : null} heatmapMode={mapView === "heatmap"} />
              </div>
            </article>
            <aside className="side-stack">
              <article className={`${panel} response-panel`}>
                <div className="card-heading">
                  <div>
                    <div className="section-kicker">ZONE DETAIL</div>
                    <h2>{currentZone?.title ?? "Awaiting scene"}</h2>
                  </div>
                  <MapPinned size={18} className="muted-icon" />
                </div>
                <div className="response-hero">
                  <span className={queueAvailable && selectedZone === "lane1" && queue >= 4 ? "risk" : zoneTickets.length ? "attention" : processed ? "clear" : "attention"}>
                    {queueAvailable && selectedZone === "lane1" && queue >= 4 ? "HIGH PRIORITY" : zoneTickets.length ? "ATTENTION NEEDED" : !processed ? "AWAITING ANALYSIS" : "ALL CLEAR"}
                  </span>
                  <h3>
                    {queueAvailable && selectedZone === "lane1" && queue >= 4
                      ? "Queue threshold exceeded"
                      : zoneTickets.length
                        ? `${zoneTickets.length} active inventory ${zoneTickets.length === 1 ? "ticket" : "tickets"}`
                        : !processed ? "No zone activity yet" : "Flow operating normally"}
                  </h3>
                  <p>
                    {queueAvailable && selectedZone === "lane1" && queue >= 4
                      ? `Staff_02 ${counterTwo?.status === "active" ? "is operating Counter 02" : "is moving from the staff access area to Counter 02"}.`
                      : zoneTickets.length
                        ? `${zoneTickets.map((ticket) => ticket.item).join(", ")} requires attention.`
                        : !processed ? "No zone measurements available." : `${currentZone?.category ?? "Scene"} is within monitored limits.`}
                  </p>
                </div>
                <div className="zone-metrics">
                  <div><span>HEADCOUNT</span><strong>{zoneStats?.headcount ?? "—"}</strong></div>
                  <div><span>DENSITY</span><strong>{zoneStats ? `${zoneStats.density_pct}%` : "—"}</strong></div>
                  <div><span>DWELL</span><strong>{zoneStats ? `${zoneStats.dwell_hours.toFixed(3)} h` : "—"}</strong></div>
                </div>
                <div className="zone-alerts">
                  {!modelInference && shelfAvailable && telemetry?.inventory_health?.[selectedZone] != null && <strong>Stock density {telemetry.inventory_health[selectedZone]}% · </strong>}
                  {!shelfAvailable ? "Shelf health unavailable" : zoneTickets.length ? zoneTickets.map((ticket) => `${ticket.item}: ${modelInference ? "at least " : ""}${ticket.missing} missing`).join(" · ") : "No active shelf health alerts"}
                </div>
                <div className="response-divider" />
                <div className="response-meta">
                  <span>LAST DETECTED</span>
                  <strong>{processed ? displayTime(telemetry?.updated_at) : "—"}</strong>
                </div>
              </article>
            </aside>
          </section>}

          {activeNav === "Reports" && !electronics && <section className="analysis-report" aria-label="Store activity report">
            <div className="analysis-report-heading"><span>CURRENT SESSION · SITE 01</span><div><span className="report-freshness">{operationalReportAvailable && telemetry?.updated_at ? `LAST SIGNAL ${displayTime(telemetry.updated_at)}` : "AWAITING MEASUREMENTS"}</span><button type="button" className="export-report" onClick={downloadReport}>EXPORT REPORT</button></div></div>
            {!operationalReportAvailable ? <article className={`${panel} report-empty`}>
              <span className="report-empty-icon"><ScanLine size={22} /></span>
              <div><div className="section-kicker">REPORT STATUS</div><h2>No measured activity for this session</h2><p>{visibleAvailable ? `Up to ${telemetry.live_occupancy} shoppers were annotated in one camera. Entry, queue and shelf results are unavailable.` : "Entry, queue and shelf results will appear when camera analysis provides them."}</p><button type="button" className="report-link" onClick={() => navigate("Cameras")}>Review camera inputs <ArrowRight size={14} /></button></div>
            </article> : <>
            {footfallAvailable && <article className={`${panel} report-card`}>
              <div className="section-kicker">01 / ENTRY TRACKING</div>
              <h2>Entry crossings</h2>
              <div className="report-pair"><div><span>IN</span><strong>{telemetry.footfall.in}</strong></div><div><span>OUT</span><strong>{telemetry.footfall.out}</strong></div></div>
              <p>{processed ? modelInference && playback.status === "complete" ? `${telemetry.live_occupancy} shoppers estimated inside at clip end` : `${telemetry.live_occupancy} shoppers currently tracked` : "Occupancy tracking unavailable"}</p>
            </article>}
            {queueAvailable && <article className={`${panel} report-card`}>
              <div className="section-kicker">02 / CHECKOUT TRACKING</div>
              <h2>Checkout service</h2>
              <div className="report-pair"><div><span>COUNTER 01</span><strong>{queue}</strong><small>{modelInference ? "Wait time unavailable" : `${telemetry.checkout.counter_01.wait_seconds}s longest wait`}</small></div><div><span>COUNTER 02</span><strong>{counterTwo?.queue_count ?? 0}</strong><small>{modelInference ? counterTwo?.status === "dispatching" ? "Backup movement detected" : counterTwo?.status ?? "standby" : `${counterTwo?.status ?? "standby"} · ${counterTwo?.response_delay_seconds ?? 0}s response`}</small></div></div>
              <p>{queueAlerts.length ? `Queue length 4: ${queueAlerts.length} ${queueAlerts.length === 1 ? "alert" : "alerts"} at ${queueAlerts.map((alert) => alert.video_seconds == null ? displayTime(alert.time) : clipTime(alert.video_seconds)).join(", ")} · ${queue >= 4 ? "backup needed" : "now clear"}` : queue >= 4 ? "Queue threshold reached · backup needed" : "Queues within monitored limit"}</p>
            </article>}
            {processed && <article className={`${panel} report-card zone-engagement-card`}>
              <div className="section-kicker">03 / SHOPPER MOVEMENT</div>
              <h2>Zone engagement</h2>
              <div className="zone-engagement-head"><span>ZONE</span><span>VISITS</span><span>DWELL</span><span>NOW</span></div>
              {shelfZones.map((zone) => { const stats = telemetry.occupancy.zones?.[zone.id]; return <div className="zone-engagement-row" key={zone.id}><span>{zone.category}</span><strong>{modelInference ? "—" : stats?.visits ?? "—"}</strong><strong>{stats ? formatDwell(stats.dwell_hours) : "—"}</strong><strong>{stats?.headcount ?? "—"}</strong></div>; })}
            </article>}
            {shelfAvailable && <article className={`${panel} report-card shelf-findings-card`}>
              <div className="section-kicker">04 / REPLENISHMENT</div>
              <h2>Shelf findings</h2>
              {tickets.filter((ticket) => ticket.status !== "Refilled").length ? tickets.filter((ticket) => ticket.status !== "Refilled").map((ticket) => <div className="shelf-finding" key={ticket.id}>{ticket.evidence && <img className="evidence-thumb" src={frameUrl(telemetry?.current_scenario ?? selectedScenario, ticket.evidence)} alt={`${ticket.item} shelf gap`} loading="lazy" onError={(event) => { event.currentTarget.hidden = true; }} />}<span><strong>{ticket.item}</strong><small>{ticket.category} · {ticket.location}</small></span><span className={`status-badge ${ticket.status.toLowerCase()}`}>{modelInference ? "≥" : ""}{ticket.missing} missing · {ticket.status}</span>{ticket.evidence && <button type="button" className="evidence-action" onClick={() => openEvidence(ticket.evidence)} aria-label={`Review ${ticket.item} source frame at ${clipTime(ticket.evidence.video_seconds)}`}>VIEW FRAME · {clipTime(ticket.evidence.video_seconds)} <ArrowRight size={11} /></button>}</div>) : <p>No shelf gaps detected.</p>}
              <button type="button" className="report-link" onClick={() => navigate("Task Queues")}>Open shelf tasks <ArrowRight size={14} /></button>
            </article>}
            </>}
          </section>}

          {(activeNav === "Task Queues" || activeNav === "System Logs") && <section className="bottom-grid single-page" aria-label="Operations and alerts">
            {activeNav === "Task Queues" && <article className={`${panel} tickets-panel`} id="tasks">
              <div className="card-heading">
                <div>
                  <div className="section-kicker">05 / SHELF OPERATIONS</div>
                  <h2>
                    Inventory task queue <span className="title-count">{tickets.filter((ticket) => ticket.status !== "Refilled").length}</span>
                  </h2>
                </div>
                <div className="table-note">
                  <i className="dot violet" /> SYNCED WITH FLOOR MAP
                </div>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>ITEM / TICKET</th>
                      <th>CATEGORY</th>
                      <th>LOCATION</th>
                      <th>{modelInference ? "CONFIRMED EMPTY" : "EMPTY COUNT"}</th>
                      <th>STATUS</th>
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tickets.map((ticket) => (
                      <tr key={ticket.id}>
                        <td>
                          {ticket.evidence && <img className="evidence-thumb" src={frameUrl(telemetry?.current_scenario ?? selectedScenario, ticket.evidence)} alt={`${ticket.item} shelf gap`} loading="lazy" onError={(event) => { event.currentTarget.hidden = true; }} />}
                          <strong>{ticket.item}</strong>
                          <small>{ticket.id}</small>
                          {ticket.evidence && <button type="button" className="evidence-action" onClick={() => openEvidence(ticket.evidence)}>VIEW SOURCE FRAME · {clipTime(ticket.evidence.video_seconds)}{ticket.evidence.confidence != null ? ` · ${Math.round(ticket.evidence.confidence * 100)}%` : ""} <ArrowRight size={11} /></button>}
                        </td>
                        <td>{ticket.category}</td>
                        <td className="mono">{ticket.location}</td>
                        <td>
                          <span className="missing-count">{modelInference ? "≥" : ""}{ticket.missing}</span> Units Missing
                        </td>
                        <td>
                          <span className={`status-badge ${ticket.status.toLowerCase()}`}>
                            <i className="dot" />
                            {ticket.status}
                          </span>
                        </td>
                        <td>
                          {nextStatus[ticket.status] && (
                            <button
                              className="row-action"
                              disabled={pendingTicket === ticket.id}
                              onClick={() => updateTicket(ticket)}
                              aria-label={`Mark ${ticket.item} ${nextStatus[ticket.status]}`}
                              title={`Mark ${nextStatus[ticket.status]}`}
                            >
                              {ticket.status === "Assigned" ? <Check size={14} /> : <ArrowRight size={14} />}
                              {ticket.status === "Assigned" ? "REFILLED" : "ASSIGN"}
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {tickets.length === 0 && <div className="table-empty">{!shelfAvailable ? "Awaiting shelf-specific video analysis." : telemetry?.mode === "simulation" ? "No shelf gaps in this demo phase." : "No shelf gaps reported by the vision pipeline."}</div>}
              </div>
              <div className="table-footer">
                <span>{tickets.length} tracked items · Select the action arrow to advance status</span>
                <span>
                  OPEN <ArrowRight size={12} /> ASSIGNED <ArrowRight size={12} /> REFILLED
                </span>
              </div>
            </article>}
            {activeNav === "System Logs" && <article className={`${panel} logs-panel`} id="logs">
              <div className="card-heading">
                <div>
                  <div className="section-kicker">06 / EVENT STREAM</div>
                  <h2>System activity</h2>
                </div>
                <span className="log-live">
                  <i className={`dot ${connection === "live" ? "green" : "amber"}`} /> {connection.toUpperCase()}
                </span>
              </div>
              <div className="log-list" ref={logRef} role="log" aria-live="polite" aria-label="Telemetry activity">
                {logs.map((log, index) => (
                  <div className="log-item" key={log.index ?? `${log.time}-${index}`}>
                    <div className={`log-icon ${log.type === "QUEUE" ? "red" : log.type === "STOCK" ? "amber" : "green"}`}>
                      {log.type === "STOCK" ? <ShoppingBag size={14} /> : log.type === "QUEUE" ? <Users size={14} /> : <PackageCheck size={14} />}
                    </div>
                    <div>
                      <div className="log-meta">
                        <span>[{log.video_seconds == null ? displayTime(log.time) : `${clipTime(log.video_seconds)} VIDEO`}]</span>
                        <span>#{String(log.index ?? logs.length - index).padStart(4, "0")}</span>
                        <strong className={log.type === "QUEUE" ? "red" : log.type === "STOCK" ? "amber" : "green"}>[{log.type}]</strong>
                      </div>
                      <p>{log.message}</p>
                      {log.evidence && <button type="button" className="evidence-action" onClick={() => openEvidence(log.evidence)}>VIEW {log.evidence.camera_id.replace("cam_", "").replaceAll("_", " ")} · {clipTime(log.evidence.video_seconds)}{log.evidence.confidence != null ? ` · ${Math.round(log.evidence.confidence * 100)}%` : ""} <ArrowRight size={11} /></button>}
                    </div>
                  </div>
                ))}
                {logs.length === 0 && <div className="table-empty">No vision events received yet.</div>}
              </div>
              <div className="log-footer">
                <span>
                  <span className="terminal-cursor" /> {connection.toUpperCase()}
                </span>
                <span>{logs.length} EVENTS</span>
              </div>
            </article>}
          </section>}

          <footer className="page-footer">
            <span>COGNO VALLEY LABS / RETAIL VISION ANALYTICS</span>
            <span>Site 01 · Store operations</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
