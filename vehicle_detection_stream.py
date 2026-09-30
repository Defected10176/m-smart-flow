"""
M-Smart Flow — Vehicle Detection & Zone Occupancy (Workflow-based)
==========================================================================
Uses the exact connection method Roboflow generated for this specific
deployed model: a Workflow (not a raw model_id), addressed by
workflow= + workspace=. This is the pattern that actually works for
this account — confirmed via Roboflow's own "Deploy My API" code sample.

Install:
    pip install "inference-sdk[webrtc]" supervision opencv-python numpy

Usage:
    set ROBOFLOW_API_KEY=your_key_here
    python vehicle_detection_stream.py --source IMG_8838.MOV --zones zones_example.json

Runs the WebRTC session in a background thread (collecting detections),
while the main thread plays the video and overlays the latest known
zone occupancy — giving a live-feeling preview even though detections
arrive asynchronously.
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
WORKFLOW_ID = "parking-lot-vparking-lot-jqsj2-nujb1-1-yolo26n-t2-logic"
WORKSPACE = "pisitjirawoottiwat101-gmail-com"


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
    parser = argparse.ArgumentParser(description="M-Smart Flow detection via Roboflow Workflow")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--zones", default="zones_example.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--events-out", default="zone_events.jsonl")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set it with: set ROBOFLOW_API_KEY=your_key_here")

    zones = load_zones(args.zones)
    events_file = open(args.events_out, "a")

    # Shared state updated by the background WebRTC thread, read by the display loop
    state = {"detections": sv.Detections.empty(), "snapshots": [], "frame_count": 0}
    state_lock = threading.Lock()

    client = InferenceHTTPClient.init(api_url=ROBOFLOW_API_URL, api_key=api_key)
    source = VideoFileSource(args.source, realtime_processing=True)
    config = StreamConfig(
        stream_output=[],
        data_output=["predictions"],
        requested_plan="webrtc-gpu-medium",
        requested_region="us",
    )

    session = client.webrtc.stream(
        source=source,
        workflow=WORKFLOW_ID,
        workspace=WORKSPACE,
        image_input="image",
        config=config,
    )

    debug_state = {"printed": False}

    @session.on_data()
    def on_data(data: dict, metadata):
        if not debug_state["printed"]:
            print("\n[DEBUG] data keys received:", list(data.keys()))
            print("[DEBUG] raw 'predictions' value:")
            print(json.dumps(data.get("predictions"), indent=2, default=str)[:3000])
            print("[DEBUG] --- end of raw data ---\n")
            debug_state["printed"] = True

        result = data.get("predictions")
        if result is None:
            return

        try:
            detections = sv.Detections.from_inference(result)
        except Exception as e:
            print(f"[DEBUG] sv.Detections.from_inference failed: {e}")
            return

        snapshots = assign_to_zones(detections, zones)

        with state_lock:
            state["detections"] = detections
            state["snapshots"] = snapshots
            state["frame_count"] += 1

        event = {
            "timestamp": time.time(),
            "vehicle_total": len(detections),
            "zones": snapshots,
        }
        events_file.write(json.dumps(event) + "\n")
        events_file.flush()
        print(f"[frame {state['frame_count']}] {len(detections)} vehicles detected")

    # Run the WebRTC session in the background
    session_thread = threading.Thread(target=session.run, daemon=True)
    session_thread.start()

    # Display loop: read the same video file ourselves and overlay the latest known detections
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    cap = cv2.VideoCapture(args.source)
    cv2.namedWindow("M-Smart Flow — Vehicle Detection", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            with state_lock:
                detections = state["detections"]
                snapshots = state["snapshots"]

            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections)
            annotated = draw_zones(annotated, zones, snapshots)

            max_width = 1000
            if annotated.shape[1] > max_width:
                scale = max_width / annotated.shape[1]
                annotated = cv2.resize(annotated, None, fx=scale, fy=scale)

            cv2.imshow("M-Smart Flow — Vehicle Detection", annotated)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        events_file.close()
        cv2.destroyAllWindows()
        session.close()


if __name__ == "__main__":
    main()