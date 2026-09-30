"""
M-Smart Flow — Vehicle Detection & Zone Occupancy (Layer 2: AI & Processing)
==============================================================================
Runs YOLO26 vehicle detection on a video/RTSP source, maps detections to
named parking zones, and emits per-frame occupancy events — the edge-side
output that feeds the cloud Density Analytics Service (Layer 3) in the
M-Smart Flow architecture.

Install:
    pip install ultralytics opencv-python numpy

Usage:
    python vehicle_detection.py --source video.mp4 --zones zones_example.json
    python vehicle_detection.py --source 0                     # webcam
    python vehicle_detection.py --source rtsp://camera-ip/stream

Output:
    - Annotated video window (bounding boxes + zone overlays + occupancy %)
    - JSONL event log (one line per frame) written to --events-out,
      matching the zone density-snapshot shape used downstream by the
      Decision Engine and ESG Engine (see system architecture doc, §5).

Note on zones:
    Zone polygons are pixel coordinates in the camera's frame, so each
    camera/zone needs its own calibrated zones.json. The included
    zones_example.json is a placeholder for testing — replace the
    polygon points with coordinates matched to your actual CCTV footage.
"""

import argparse
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs relevant to vehicle detection (YOLO26 default weights are COCO-trained)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class ParkingZone:
    zone_id: str
    polygon: list  # [[x, y], ...] pixel coordinates in the camera frame
    capacity: int   # approx. number of vehicles the zone holds when full

    def as_np(self):
        return np.array(self.polygon, dtype=np.int32)

    def contains_point(self, x, y):
        return cv2.pointPolygonTest(self.as_np(), (float(x), float(y)), False) >= 0


def load_zones(zones_path: str):
    with open(zones_path, "r") as f:
        raw = json.load(f)
    return [ParkingZone(z["zone_id"], z["polygon"], z["capacity"]) for z in raw["zones"]]


def detect_vehicles(model: YOLO, frame, conf_threshold: float = 0.4):
    """Run YOLO26 inference and return a list of vehicle detections."""
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detections.append({
            "class": VEHICLE_CLASSES[cls_id],
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
        })
    return detections


def assign_to_zones(detections, zones):
    """Bucket detections into zones and compute per-zone occupancy %."""
    zone_counts = {z.zone_id: 0 for z in zones}
    for det in detections:
        cx, cy = det["center"]
        for zone in zones:
            if zone.contains_point(cx, cy):
                zone_counts[zone.zone_id] += 1
                break  # a vehicle belongs to at most one zone

    snapshots = []
    for zone in zones:
        count = zone_counts[zone.zone_id]
        density_pct = round(min(count / zone.capacity, 1.0) * 100, 1) if zone.capacity else 0.0
        snapshots.append({
            "zone_id": zone.zone_id,
            "vehicle_count": count,
            "capacity": zone.capacity,
            "density_pct": density_pct,
        })
    return snapshots


def draw_overlay(frame, detections, zones, snapshots):
    for zone, snap in zip(zones, snapshots):
        if snap["density_pct"] < 60:
            color = (0, 200, 0)
        elif snap["density_pct"] < 80:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.polylines(frame, [zone.as_np()], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, f"{zone.zone_id}: {snap['density_pct']}%", tuple(zone.polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="M-Smart Flow vehicle detection & zone occupancy")
    parser.add_argument("--source", default="0", help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--zones", default="zones_example.json", help="Path to zone config JSON")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO26 weights (nano recommended for edge/CPU)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--events-out", default="zone_events.jsonl", help="Path to write JSONL occupancy events")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no video window)")
    parser.add_argument("--start-seconds", type=float, default=0,
                         help="Skip ahead this many seconds before starting playback/detection")
    args = parser.parse_args()

    model = YOLO(args.model)  # auto-downloads weights on first run
    zones = load_zones(args.zones)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_sec = total_frames / fps if fps else 0
    print(f"[info] Video duration: ~{duration_sec:.1f} seconds ({int(total_frames)} frames at {fps:.1f} fps)")

    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
        print(f"[info] Skipped ahead to {args.start_seconds} seconds")

    events_file = open(args.events_out, "a")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detect_vehicles(model, frame, args.conf)
            snapshots = assign_to_zones(detections, zones)

            event = {
                "timestamp": time.time(),
                "vehicle_total": len(detections),
                "zones": snapshots,
            }
            events_file.write(json.dumps(event) + "\n")
            events_file.flush()  # simulates the MQTT publish step in the architecture doc

            if not args.no_display:
                frame = draw_overlay(frame, detections, zones, snapshots)

                max_display_width = 700
                if frame.shape[1] > max_display_width:
                    scale = max_display_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                cv2.namedWindow("M-Smart Flow — Vehicle Detection", cv2.WINDOW_NORMAL)
                cv2.imshow("M-Smart Flow — Vehicle Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
