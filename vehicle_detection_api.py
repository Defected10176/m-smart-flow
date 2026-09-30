"""
M-Smart Flow — Vehicle Detection & Zone Occupancy (run_workflow version)
==========================================================================
Uses client.run_workflow() — a simple, synchronous HTTP call per frame,
using the exact workspace_name + workflow_id Roboflow generated for this
deployed model. No WebRTC/streaming complexity, so no STUN/firewall
issues — just a plain request/response per frame.

Install:
    pip install inference-sdk opencv-python numpy

Usage:
    set ROBOFLOW_API_KEY=your_key_here
    python vehicle_detection_api.py --source IMG_8838.MOV --zones zones_example.json
"""

import argparse
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
MODEL_ID = "pisitjirawoottiwat101-gmail-com/parking-lot-jqsj2-nujb1-1-yolo26n-t2"


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


def extract_predictions(result):
    """The workflow's result shape can vary — find the list of prediction dicts wherever it is."""
    if isinstance(result, list):
        result = result[0] if result else {}
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, dict) and "predictions" in value:
                return value["predictions"]
            if isinstance(value, list) and value and isinstance(value[0], dict) and "class" in value[0]:
                return value
        if "predictions" in result:
            return result["predictions"]
    return []


def detect_vehicles(client, frame, conf_threshold: float = 0.05):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, frame)

    try:
        result = client.infer(tmp_path, model_id=MODEL_ID)
    finally:
        os.remove(tmp_path)

    if not detect_vehicles._debug_printed:
        print("\n[DEBUG] Raw result (first call only):")
        print(json.dumps(result, indent=2, default=str)[:2000])
        print("[DEBUG] --- end ---\n")
        detect_vehicles._debug_printed = True

    predictions = result.get("predictions", []) if isinstance(result, dict) else []

    detections = []
    for pred in predictions:
        if pred.get("confidence", 0) < conf_threshold:
            continue
        cx, cy = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        detections.append({
            "class": pred.get("class", "unknown"),
            "confidence": round(pred.get("confidence", 0), 3),
            "box": [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
            "center": [cx, cy],
        })
    return detections


detect_vehicles._debug_printed = False


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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="M-Smart Flow detection via run_workflow")
    parser.add_argument("--source", default="0", help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--zones", default="zones_example.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold, 0-1 (this model needs a low value on real footage, e.g. 0.05)")
    parser.add_argument("--events-out", default="zone_events.jsonl")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=5,
                         help="How often (in frames) to grab a frame for background detection")
    parser.add_argument("--start-seconds", type=float, default=0.0,
                         help="Skip ahead this many seconds before starting (use to skip past intro/ceiling footage)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set it with: set ROBOFLOW_API_KEY=your_key_here")

    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    ).configure(InferenceConfiguration(api_key_transport="header"))

    zones = load_zones(args.zones)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    print(f"[INFO] Opened video source: {args.source}")

    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
        print(f"[INFO] Skipped ahead to {args.start_seconds}s")

    events_file = open(args.events_out, "a")

    # Shared state: background thread writes detections, display loop reads them.
    # Video plays at full speed; detections update whenever the network call finishes,
    # instead of freezing playback while waiting on each request.
    state = {"detections": [], "snapshots": [], "latest_frame": None, "frame_count": 0}
    state_lock = threading.Lock()
    stop_event = threading.Event()

    def detection_worker():
        while not stop_event.is_set():
            with state_lock:
                frame = state["latest_frame"]
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                detections = detect_vehicles(client, frame, args.conf)
            except Exception as e:
                print(f"[ERROR] detection call failed: {type(e).__name__}: {e}")
                # clear the frame so we don't just hammer the API with the same
                # failing frame in a tight loop, and back off briefly
                with state_lock:
                    state["latest_frame"] = None
                time.sleep(1.0)
                continue

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
            print(f"[detection {state['frame_count']}] {len(detections)} vehicles found")

    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()
    print("[INFO] Detection worker thread started, waiting for first frame...")

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            if frame_index % args.frame_skip == 0:
                with state_lock:
                    state["latest_frame"] = frame.copy()

            if not args.no_display:
                with state_lock:
                    detections = state["detections"]
                    snapshots = state["snapshots"]

                annotated = draw_overlay(frame.copy(), detections, zones, snapshots)

                max_display_width = 1000
                if annotated.shape[1] > max_display_width:
                    scale = max_display_width / annotated.shape[1]
                    annotated = cv2.resize(annotated, None, fx=scale, fy=scale)

                cv2.namedWindow("M-Smart Flow — Vehicle Detection", cv2.WINDOW_NORMAL)
                cv2.imshow("M-Smart Flow — Vehicle Detection", annotated)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
    finally:
        stop_event.set()
        cap.release()
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()