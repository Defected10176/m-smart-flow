"""
M-Smart Flow — Vehicle Detection & Zone Occupancy (Hosted API version)
==========================================================================
Same as vehicle_detection.py, but calls your trained model through
Roboflow's Serverless Cloud API instead of loading local .pt weights.
Use this if your plan doesn't allow downloading weights directly.

Install:
    pip install inference-sdk opencv-python numpy

You'll need:
    - Your Roboflow API key (Settings > API Keys)
    - Your model ID, in the form "project-slug/version", e.g. "parking-lot-jqsj2/1"
      (find this on your model's page, or in the "Code Samples" section)

Usage:
    export ROBOFLOW_API_KEY="UHKIluNzHGyjWsAMxoqk"
    python vehicle_detection_api.py --source video.mp4 --model-id parking-lot-jqsj2/1 --zones zones_example.json
"""
ROBOFLOW_API_KEY="UHKIluNzHGyjWsAMxoqk"

import argparse
import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

ROBOFLOW_API_URL = "https://serverless.roboflow.com"


@dataclass
class ParkingZone:
    zone_id: str
    polygon: list
    capacity: int

    def as_np(self):
        return np.array(self.polygon, dtype=np.int32)

    def contains_point(self, x, y):
        return cv2.pointPolygonTest(self.as_np(), (float(x), float(y)), False) >= 0


def load_zones(zones_path: str):
    with open(zones_path, "r") as f:
        raw = json.load(f)
    return [ParkingZone(z["zone_id"], z["polygon"], z["capacity"]) for z in raw["zones"]]


def detect_vehicles(client: InferenceHTTPClient, model_id: str, frame, conf_threshold: float = 0.4):
    """Send a frame to the hosted model and return normalized detections."""
    result = client.infer(frame, model_id=model_id)
    detections = []
    for pred in result.get("predictions", []):
        if pred.get("confidence", 0) < conf_threshold:
            continue
        cx, cy = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        detections.append({
            "class": pred.get("class", "unknown"),
            "confidence": round(pred["confidence"], 3),
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
        })
    return detections


def assign_to_zones(detections, zones):
    zone_counts = {z.zone_id: 0 for z in zones}
    for det in detections:
        cx, cy = det["center"]
        for zone in zones:
            if zone.contains_point(cx, cy):
                zone_counts[zone.zone_id] += 1
                break
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
    parser = argparse.ArgumentParser(description="M-Smart Flow detection via Roboflow hosted API")
    parser.add_argument("--source", default="0", help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--zones", default="zones_example.json", help="Path to zone config JSON")
    parser.add_argument("--model-id", required=True, help="e.g. parking-lot-jqsj2/1")
    parser.add_argument("--api-key", default=None, help="Falls back to ROBOFLOW_API_KEY env var if omitted")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--events-out", default="zone_events.jsonl")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=5,
                         help="Only run inference every N frames — hosted API calls are rate/credit limited")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit('No API key found. Set it with: export ROBOFLOW_API_KEY="your_key_here"')

    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key)
    zones = load_zones(args.zones)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    events_file = open(args.events_out, "a")
    frame_count = 0
    last_detections, last_snapshots = [], []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            # Only call the hosted API every N frames to stay within free-tier limits
            if frame_count % args.frame_skip == 0:
                last_detections = detect_vehicles(client, args.model_id, frame, args.conf)
                last_snapshots = assign_to_zones(last_detections, zones)

                event = {
                    "timestamp": time.time(),
                    "vehicle_total": len(last_detections),
                    "zones": last_snapshots,
                }
                events_file.write(json.dumps(event) + "\n")
                events_file.flush()

            if not args.no_display:
                frame = draw_overlay(frame, last_detections, zones, last_snapshots)
                cv2.imshow("M-Smart Flow — Vehicle Detection (Hosted API)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
