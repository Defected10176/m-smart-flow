"""
M-Smart Flow - Vehicle Detection & Zone Occupancy (run_workflow version)
==========================================================================
Uses client.run_workflow() — a simple, synchronous HTTP call per frame,
using the exact workspace_name + workflow_id Roboflow generated for this
deployed model. No WebRTC/streaming complexity, so no STUN/firewall
issues — just a plain request/response per frame.

Install:
    pip install inference-sdk opencv-python numpy

Usage:
    set ROBOFLOW_API_KEY=UHKIluNzHGyjWsAMxoqk
    python vehicle_detection_api.py --source IMG_8838.MOV --zones zones_example.json
"""

"python vehicle_detection_api.py --source IMG_8844.MOV --zones zones_8844.json --conf 0.05 --start-seconds 8 --infer-free"
"python vehicle_detection_api.py --source IMG_8836.MOV --zones zones_8836.json --conf 0.05 --start-seconds 8 --spots --freeze"
"python vehicle_detection_api.py --source IMG_8838.MOV --zones zones_8838.json --conf 0.05 --start-seconds 8 --spots --freeze"
"python vehicle_detection_api.py --source IMG_8843.MOV --zones zones_8843.json --conf 0.05 --start-seconds 8 --spots --freeze"
"python vehicle_detection_api.py --source IMG_8845.MOV --zones zones_8845.json --conf 0.05 --start-seconds 8 --spots --freeze"
"python vehicle_detection_api.py --source IMG_8847.MOV --zones zones_8847.json --conf 0.05 --start-seconds 8 --spots --freeze"


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


def detect_vehicles(client, frame, conf_threshold: float = 0.05, model_id: str = MODEL_ID):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, frame)

    try:
        result = client.infer(tmp_path, model_id=model_id)
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


def compute_zone_crop_box(zone, frame_shape,
                           pad_top_ratio=0.25, pad_bottom_ratio=0.15,
                           pad_side_ratio=0.1, min_pad_px=60):
    """
    The model performs much better on a tight crop around a single row of
    cars than on the full wide-angle frame (confirmed via direct testing:
    full-frame confidences topped out ~0.56 with almost nothing else
    detected, while cropped-to-zone confidences reliably found multiple
    cars at once). Zones are expected to be drawn around the WHOLE cars in a
    row (floor to roof), so we only add a small margin on each side. Large
    padding on a tall zone would make the crop nearly the whole frame and
    lose the zoom benefit. Result is clamped to the frame bounds.
    """
    xs = [p[0] for p in zone.polygon]
    ys = [p[1] for p in zone.polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1

    pad_top = max(int(h * pad_top_ratio), min_pad_px)
    pad_bottom = max(int(h * pad_bottom_ratio), min_pad_px // 2)
    pad_side = max(int(w * pad_side_ratio), min_pad_px)

    frame_h, frame_w = frame_shape[:2]
    cx1 = max(0, x1 - pad_side)
    cy1 = max(0, y1 - pad_top)
    cx2 = min(frame_w, x2 + pad_side)
    cy2 = min(frame_h, y2 + pad_bottom)
    return cx1, cy1, cx2, cy2


COCO_VEHICLE_CLASSES = {"car", "truck", "bus"}
# Mutable so the worker thread can switch the helper model off after a failure.
car_model_state = {"enabled": False, "model_id": None, "warned": False}


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def point_in_box(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def detect_vehicles_in_zone(client, frame, zone, args):
    """Crop to this zone's region (with padding), run detection on just the
    crop, then offset the returned boxes/centers back into full-frame
    coordinates so overlay drawing and zone point-in-polygon tests keep
    working unchanged.

    Two improvements over a single-model call:
      1. Separate confidence cutoffs: 'free' needs a higher confidence than
         'car', because on our footage the custom model often mislabels cars
         as 'free' at low confidence.
      2. Optional helper model (a standard COCO detector, also on Roboflow's
         cloud) that is much better at spotting cars from any angle. A car it
         finds is added as 'car', and any 'free' box whose center lies on a
         detected car is dropped (a car there means the spot is occupied).
    """
    cx1, cy1, cx2, cy2 = compute_zone_crop_box(zone, frame.shape)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []

    # Query the custom model at the lower of the two cutoffs, then filter per class.
    raw = detect_vehicles(client, crop, min(args.conf, args.free_conf))
    detections = [
        d for d in raw
        if (d["class"] == "free" and d["confidence"] >= args.free_conf)
        or (d["class"] != "free" and d["confidence"] >= args.conf)
    ]

    if car_model_state["enabled"]:
        try:
            coco = detect_vehicles(client, crop, args.car_conf, model_id=car_model_state["model_id"])
            coco_cars = [d for d in coco if d["class"] in COCO_VEHICLE_CLASSES]
            for d in coco_cars:
                d["class"] = "car"
            # drop 'free' boxes sitting on a detected car
            detections = [
                d for d in detections
                if not (d["class"] == "free" and any(point_in_box(d["center"], c["box"]) for c in coco_cars))
            ]
            # merge: skip a helper car if the custom model already has the same car
            for c in coco_cars:
                dup = next((d for d in detections
                            if d["class"] != "free" and box_iou(d["box"], c["box"]) > 0.4), None)
                if dup is None:
                    detections.append(c)
                elif c["confidence"] > dup["confidence"]:
                    dup["confidence"] = c["confidence"]
        except Exception as e:
            if not car_model_state["warned"]:
                print(f"[WARN] helper car model '{car_model_state['model_id']}' failed "
                      f"({type(e).__name__}: {e}). Continuing with the custom model only.")
                car_model_state["warned"] = True
            car_model_state["enabled"] = False

    for det in detections:
        det["box"][0] += cx1
        det["box"][1] += cy1
        det["box"][2] += cx1
        det["box"][3] += cy1
        det["center"][0] += cx1
        det["center"][1] += cy1
    return detections


def assign_to_zones(detections, zones):
    """
    The model classifies each detection as either "car" (an occupied space)
    or "free" (an empty space) — these must be counted separately. Lumping
    them together would count an empty spot as a parked vehicle and inflate
    occupancy.
    """
    zone_cars = {z.zone_id: 0 for z in zones}
    zone_free = {z.zone_id: 0 for z in zones}

    for det in detections:
        cx, cy = det["center"]
        cls = det["class"]
        for zone in zones:
            if zone.contains_point(cx, cy):
                if cls == "free":
                    zone_free[zone.zone_id] += 1
                else:
                    # anything that isn't "free" is treated as an occupying vehicle
                    # (covers "car" plus any other vehicle class the model might emit)
                    zone_cars[zone.zone_id] += 1
                break

    snapshots = []
    for zone in zones:
        cars = zone_cars[zone.zone_id]
        free = zone_free[zone.zone_id]
        detected_spots = cars + free

        # Prefer the model's own detected-spot count for density when it saw any
        # spots at all; fall back to the configured capacity otherwise (e.g. the
        # model missed some spots this frame, or zone is empty of detections).
        denominator = detected_spots if detected_spots > 0 else zone.capacity
        density_pct = round(min(cars / denominator, 1.0) * 100, 1) if denominator else 0.0

        snapshots.append({
            "zone_id": zone.zone_id,
            "vehicle_count": cars,
            "free_count": free,
            "capacity": zone.capacity,
            "density_pct": density_pct,
        })
    return snapshots


CAR_BOX_COLOR = (0, 0, 255)     # red — occupied
FREE_BOX_COLOR = (0, 200, 0)    # green — empty spot


def draw_overlay(frame, detections, zones, snapshots):
    for zone, snap in zip(zones, snapshots):
        if snap["density_pct"] < 60:
            color = (0, 200, 0)
        elif snap["density_pct"] < 80:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.polylines(frame, [zone.as_np()], isClosed=True, color=color, thickness=2)
        label = f"{zone.zone_id}: {snap['vehicle_count']} car / {snap['free_count']} free ({snap['density_pct']}%)"
        cv2.putText(frame, label, tuple(zone.polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        is_free = det["class"] == "free"
        color = FREE_BOX_COLOR if is_free else CAR_BOX_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="M-Smart Flow detection via run_workflow")
    parser.add_argument("--source", default="0", help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--zones", default="zones_example.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold, 0-1 (this model needs a low value on real footage, e.g. 0.05)")
    parser.add_argument("--free-conf", type=float, default=0.35,
                         help="Higher confidence cutoff for 'free' detections (the model often mislabels cars as free at low confidence)")
    parser.add_argument("--car-model", default="yolov8n-640",
                         help="Helper COCO car-detector model ID on Roboflow (better at spotting cars from any angle). Use '' to disable.")
    parser.add_argument("--car-conf", type=float, default=0.25, help="Confidence cutoff for the helper car model")
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

    if args.car_model:
        car_model_state["enabled"] = True
        car_model_state["model_id"] = args.car_model
        print(f"[INFO] Helper car model enabled: {args.car_model} (conf {args.car_conf}); "
              f"'free' cutoff {args.free_conf}, 'car' cutoff {args.conf}")

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
                if zones:
                    detections = []
                    for zone in zones:
                        detections.extend(detect_vehicles_in_zone(client, frame, zone, args))
                else:
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

            # Raw totals: every detection the model returned, regardless of
            # whether it falls inside a configured zone polygon. zones_example.json
            # is still a placeholder (not calibrated to real camera footage), so
            # the zone-scoped counts below can be 0 even when the model is finding
            # plenty of cars elsewhere in frame — these raw totals are what
            # actually reflect what the model sees.
            raw_car_total = sum(1 for d in detections if d["class"] != "free")
            raw_free_total = sum(1 for d in detections if d["class"] == "free")

            # Zone-scoped totals: only detections that fall inside a zone polygon.
            # Only meaningful once zones_example.json is calibrated to this camera.
            zone_car_total = sum(s["vehicle_count"] for s in snapshots)
            zone_free_total = sum(s["free_count"] for s in snapshots)

            event = {
                "timestamp": time.time(),
                "vehicle_total": raw_car_total,
                "free_total": raw_free_total,
                "zone_vehicle_total": zone_car_total,
                "zone_free_total": zone_free_total,
                "zones": snapshots,
            }
            events_file.write(json.dumps(event) + "\n")
            events_file.flush()
            print(f"[detection {state['frame_count']}] {raw_car_total} cars, {raw_free_total} free spots total "
                  f"(in zones: {zone_car_total} cars, {zone_free_total} free)")

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

                cv2.namedWindow("M-Smart Flow - Vehicle Detection", cv2.WINDOW_NORMAL)
                cv2.imshow("M-Smart Flow - Vehicle Detection", annotated)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
    finally:
        stop_event.set()
        cap.release()
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()