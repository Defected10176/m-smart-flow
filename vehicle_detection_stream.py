"""
M-Smart Flow — Vehicle Detection & Zone Occupancy (Hosted Video Streaming)
==========================================================================
Uses Roboflow's official Serverless Video Streaming API (WebRTC) to run
your trained model on a video, instead of loading local .pt weights or
manually sending individual frame requests. This is Roboflow's recommended
approach for video/live-camera inference and handles frame delivery more
efficiently than calling .infer() per frame.

Install:
    pip install "inference-sdk[webrtc]" supervision opencv-python numpy

You'll need:
    - Your Roboflow API key (Settings > API Keys)
    - Your model ID, in the form "project-slug/version", e.g. "parking-lot-jqsj2/1"

Usage:
    export ROBOFLOW_API_KEY="UHKIluNzHGyjWsAMxoqk"
    python vehicle_detection_stream.py --source video.mp4 --model-id parking-lot-jqsj2/1 --zones zones_example.json
"""
ROBOFLOW_API_KEY="UHKIluNzHGyjWsAMxoqk"

import argparse
import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource

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


def assign_to_zones(detections: sv.Detections, zones):
    """Map supervision Detections (xyxy boxes) to zones and compute occupancy %."""
    zone_counts = {z.zone_id: 0 for z in zones}
    for box in detections.xyxy:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
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


def draw_zones(frame, zones, snapshots):
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
    return frame


def main():
    parser = argparse.ArgumentParser(description="M-Smart Flow detection via Roboflow video streaming API")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--zones", default="zones_example.json", help="Path to zone config JSON")
    parser.add_argument("--model-id", required=True, help="e.g. parking-lot-jqsj2/1")
    parser.add_argument("--api-key", default=None, help="Falls back to ROBOFLOW_API_KEY env var if omitted")
    parser.add_argument("--events-out", default="zone_events.jsonl")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("UHKIluNzHGyjWsAMxoqkY")
    if not api_key:
        raise SystemExit('No API key found. Set it with: export ROBOFLOW_API_KEY="your_key_here"')

    zones = load_zones(args.zones)
    events_file = open(args.events_out, "a")

    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key)
    session = client.webrtc.stream(
        source=VideoFileSource(args.source),
        model_id=args.model_id,
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    @session.on_frame
    def show(frame, data):
        if data is None:
            return

        detections = sv.Detections.from_inference(data)
        snapshots = assign_to_zones(detections, zones)

        event = {
            "timestamp": time.time(),
            "vehicle_total": len(detections),
            "zones": snapshots,
        }
        events_file.write(json.dumps(event) + "\n")
        events_file.flush()

        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections)
        annotated = draw_zones(annotated, zones, snapshots)

        cv2.imshow("M-Smart Flow — Vehicle Detection (Streaming)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()

    try:
        session.run()
    finally:
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
