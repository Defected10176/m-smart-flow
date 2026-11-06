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
    set ROBOFLOW_API_KEY=your_key_here
    python vehicle_detection_api.py --source IMG_8838.MOV --zones zones_example.json
"""

import argparse
import atexit
import json
import os
import sys
import tempfile
import threading
import time
from collections import deque
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
    # Optional: hand-drawn polygons, one per parking spot. If absent, --spots
    # splits the zone into `capacity` equal slots left-to-right.
    spots: list = None

    def as_np(self):
        return np.array(self.polygon, dtype=np.int32)

    def contains_point(self, x, y):
        return cv2.pointPolygonTest(self.as_np(), (float(x), float(y)), False) >= 0


def load_zones(zones_path: str):
    with open(zones_path, "r") as f:
        raw = json.load(f)
    return [ParkingZone(z["zone_id"], z["polygon"], z["capacity"], z.get("spots")) for z in raw["zones"]]


def clip_polygon_x(poly, xmin, xmax):
    """Clip a polygon to the vertical strip xmin <= x <= xmax (Sutherland-Hodgman)."""
    def clip(points, inside, intersect):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
        return out

    def at_x(x):
        def f(p, q):
            t = (x - p[0]) / float(q[0] - p[0])
            return (x, p[1] + t * (q[1] - p[1]))
        return f

    pts = [(float(p[0]), float(p[1])) for p in poly]
    pts = clip(pts, lambda p: p[0] >= xmin, at_x(xmin)) if pts else pts
    pts = clip(pts, lambda p: p[0] <= xmax, at_x(xmax)) if pts else pts
    return [[int(round(x)), int(round(y))] for x, y in pts]


def zone_slots(zone):
    """Parking-spot polygons for a zone: the hand-drawn `spots` if given,
    otherwise the zone cut into `capacity` equal strips left-to-right."""
    if zone.spots:
        return [s for s in zone.spots if len(s) >= 3]
    n = max(int(zone.capacity), 1)
    xs = [p[0] for p in zone.polygon]
    x1, x2 = min(xs), max(xs)
    step = (x2 - x1) / float(n)
    slots = []
    for i in range(n):
        piece = clip_polygon_x(zone.polygon, x1 + i * step, x1 + (i + 1) * step)
        if len(piece) >= 3:
            slots.append(piece)
    return slots


def dedupe_cars(detections, iou_thr=0.4, contain_thr=0.7):
    """Keep one box per car. The custom model and the helper model can both box
    the same car; keep the higher-confidence one. 'free' boxes are untouched."""
    cars = sorted([d for d in detections if d["class"] != "free"],
                  key=lambda d: d["confidence"], reverse=True)
    others = [d for d in detections if d["class"] == "free"]
    kept = []
    for d in cars:
        dup = False
        for k in kept:
            a, b = d["box"], k["box"]
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1)
            area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1)
            iou = inter / float(area_a + area_b - inter)
            if iou > iou_thr or inter / float(min(area_a, area_b)) > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept + others


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


# --- Local model mode (--weights) -------------------------------------------------
# When set, our own trained YOLO weights run on this computer instead of the Roboflow cloud.
local_model = {"model": None, "imgsz": 640, "times": []}


def detect_vehicles_local(frame, conf_threshold):
    """Run our local weights and return detections in the same format the cloud path uses."""
    t0 = time.perf_counter()
    r = local_model["model"].predict(frame, conf=conf_threshold, imgsz=local_model["imgsz"], verbose=False)[0]
    local_model["times"].append((time.perf_counter() - t0) * 1000)
    detections = []
    for b in r.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        detections.append({
            "class": r.names[int(b.cls)],
            "confidence": round(float(b.conf), 3),
            "box": [x1, y1, x2, y2],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
        })
    return detections


def print_local_speed():
    t = local_model["times"]
    if t:
        avg = sum(t) / len(t)
        print(f"[LOCAL] {len(t)} inferences, average {avg:.1f} ms each (~{1000 / avg:.1f} per second) on this computer")


def detect_vehicles(client, frame, conf_threshold: float = 0.05, model_id: str = MODEL_ID):
    if local_model["model"] is not None and model_id == MODEL_ID:
        return detect_vehicles_local(frame, conf_threshold)
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


def assign_to_zones(detections, zones, infer_free=False, use_spots=False):
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
        free_inferred = False
        free_slots = []
        if use_spots and zone.spots:
            # Per-spot check: a spot is occupied if a detected car's center or
            # bottom-center falls inside it; otherwise it is free. The free
            # spots are drawn as green boxes. These come from the spot shapes
            # (zone geometry), not from a model detection, hence "inferred".
            slots = zone_slots(zone)
            zone_dets = [d for d in detections
                         if d["class"] != "free" and zone.contains_point(*d["center"])]
            for slot in slots:
                slot_np = np.array(slot, dtype=np.int32)
                occupied = False
                for d in zone_dets:
                    x1, y1, x2, y2 = d["box"]
                    probes = [d["center"], ((x1 + x2) // 2, y2)]
                    if any(cv2.pointPolygonTest(slot_np, (float(px), float(py)), False) >= 0
                           for px, py in probes):
                        occupied = True
                        break
                if not occupied:
                    free_slots.append(slot)
            free = len(free_slots)
            free_inferred = True
        elif use_spots:
            # No hand-drawn bays for this zone: do NOT invent free boxes from equal strips (they land on cars).
            # Estimate a free COUNT from the zone's capacity instead.
            free = max(zone.capacity - cars, 0)
            free_inferred = True
        elif infer_free:
            # Fallback when the model cannot see empty spots (e.g. unusual floor
            # paint): free = configured capacity minus detected cars. This is an
            # estimate from the zone's capacity, not a model detection.
            estimated = max(zone.capacity - cars, 0)
            if estimated > free:
                free = estimated
                free_inferred = True
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
            "free_inferred": free_inferred,
            "free_slots": free_slots,
            "capacity": zone.capacity,
            "density_pct": density_pct,
        })
    return snapshots


# ---------------------------------------------------------------------------
# Moving-car detection (--moving)
# ---------------------------------------------------------------------------
MOVE_GRAY_WIDTH = 640     # motion is analysed on a small grayscale copy (fast)
MOVING_BOX_COLOR = (255, 0, 255)   # magenta - car that is driving, not parked


def to_motion_gray(frame):
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if w > MOVE_GRAY_WIDTH:
        s = MOVE_GRAY_WIDTH / float(w)
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return g


def residual_motion_map(prev_gray, cur_gray):
    """Per-pixel motion that is NOT explained by the camera moving.

    The camera is hand-held, so the whole picture shifts between frames. We
    estimate that global shift from background feature points (affine fit with
    RANSAC), subtract it from the dense optical flow, and what is left is the
    motion of things that really move (a driving car). Units: pixels on the
    small (640 px wide) image between the two compared frames. Returns None if
    the frames are missing or identical (e.g. --freeze)."""
    if prev_gray is None or cur_gray is None or prev_gray.shape != cur_gray.shape:
        return None
    if prev_gray is cur_gray or np.array_equal(prev_gray, cur_gray):
        return None
    h, w = cur_gray.shape

    A = None
    pts = cv2.goodFeaturesToTrack(prev_gray, 300, 0.01, 8)
    if pts is not None and len(pts) >= 10:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
        ok = st.ravel() == 1
        if ok.sum() >= 10:
            A, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                               ransacReprojThreshold=1.5)
    if A is None:
        A = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pred_dx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2] - xs
    pred_dy = A[1, 0] * xs + A[1, 1] * ys + A[1, 2] - ys
    res = np.hypot(flow[..., 0] - pred_dx, flow[..., 1] - pred_dy).astype(np.float32)

    # Flat areas (white ceiling, plain floor) have no texture to track, so the flow
    # there is just a guess and would look like "motion" whenever the camera moves.
    # Ignore them: only textured pixels can vote that something is moving.
    gx = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.GaussianBlur(np.hypot(gx, gy), (7, 7), 0)
    res[texture < 25.0] = 0.0
    return res


def find_motion_regions(res, frame_w, thr=1.5, min_area=120, max_regions=4, max_cover=0.35):
    """Boxes (full-frame pixel coordinates) around every blob that moves by
    itself, ANYWHERE in the picture - this is what lets moving cars be found in
    every part of the camera view, not only inside the drawn zones."""
    if res is None:
        return []
    h, w = res.shape
    s = frame_w / float(w)
    mask = (cv2.GaussianBlur(res, (9, 9), 0) > thr).astype(np.uint8) * 255
    mx, my = int(w * 0.03), int(h * 0.03)   # frame edges give false motion when the camera shifts
    mask[:my, :] = 0; mask[-my:, :] = 0; mask[:, :mx] = 0; mask[:, -mx:] = 0
    if mask.mean() / 255.0 > max_cover:
        return []   # most of the picture "moves": camera shake/parallax, not cars
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            boxes.append((bw * bh, [int(x * s), int(y * s), int((x + bw) * s), int((y + bh) * s)]))
    boxes.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in boxes[:max_regions]]


def flag_moving(res, detections, frame_w, thr=1.5):
    """Mark each non-'free' detection with det['moving'] (True/False) and
    det['motion'] using the residual motion map: a parked car leaves ~0 motion
    inside its box, a driving car leaves a clear amount."""
    for d in detections:
        d["moving"] = False
        d["motion"] = 0.0
    if res is None:
        return
    h, w = res.shape
    s = w / float(frame_w)
    for d in detections:
        if d["class"] == "free":
            continue
        x1, y1, x2, y2 = d["box"]
        x1, x2 = int(max(0, x1 * s)), int(min(w, x2 * s))
        y1, y2 = int(max(0, y1 * s)), int(min(h, y2 * s))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        sub = res[y1:y2, x1:x2]
        # Car bodies are often plain (white/black paint = no texture to track), so a
        # median would read ~0 even for a driving car. Instead: what share of the box
        # shows real leftover motion? A parked car has almost none.
        frac = float((sub > thr).mean())
        d["motion"] = round(float(np.percentile(sub, 90)), 2)
        d["moving"] = frac > 0.06


def detect_cars_in_regions(client, frame, regions, args, pad=40, max_width=960):
    """Confirm with the cloud car detector what each moving blob is: crop around
    it, detect cars in the crop, and map the boxes back to full-frame coordinates."""
    out = []
    fh, fw = frame.shape[:2]
    for (x1, y1, x2, y2) in regions:
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(fw, x2 + pad), min(fh, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        sc = 1.0
        if crop.shape[1] > max_width:
            sc = max_width / float(crop.shape[1])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if car_model_state["enabled"]:
            raw = detect_vehicles(client, crop, args.car_conf, model_id=car_model_state["model_id"])
            raw = [d for d in raw if d["class"] in COCO_VEHICLE_CLASSES]
        else:
            raw = detect_vehicles(client, crop, args.conf)
            raw = [d for d in raw if d["class"] != "free"]
        for d in raw:
            d["class"] = "car"
            d["box"] = [int(d["box"][0] / sc) + cx1, int(d["box"][1] / sc) + cy1,
                        int(d["box"][2] / sc) + cx1, int(d["box"][3] / sc) + cy1]
            d["center"] = [int(d["center"][0] / sc) + cx1, int(d["center"][1] / sc) + cy1]
            out.append(d)
    return out


def detect_cars_full_frame(client, frame, args, max_width=1280):
    """One extra call on the whole (downscaled) frame, so cars driving through
    the aisle - outside every parking zone - are found too."""
    h, w = frame.shape[:2]
    s = 1.0
    small = frame
    if w > max_width:
        s = max_width / float(w)
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if car_model_state["enabled"]:
        raw = detect_vehicles(client, small, args.car_conf, model_id=car_model_state["model_id"])
        raw = [d for d in raw if d["class"] in COCO_VEHICLE_CLASSES]
    else:
        raw = detect_vehicles(client, small, args.conf)
        raw = [d for d in raw if d["class"] != "free"]
    for d in raw:
        d["class"] = "car"
        d["box"] = [int(v / s) for v in d["box"]]
        d["center"] = [int(v / s) for v in d["center"]]
    return raw


CAR_BOX_COLOR = (0, 0, 255)     # red — occupied
FREE_BOX_COLOR = (0, 200, 0)    # green — empty spot


overlay_options = {"show_conf": False}   # --show-conf turns the per-car confidence tags on


def _label(frame, text, x, y, color, scale, thick, bg=(25, 25, 25)):
    """Text on a dark box so it stays readable on any floor colour. (x, y) = top-left of the box."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(3, int(scale * 6))
    h, w = frame.shape[:2]
    x = int(min(max(x, 0), max(0, w - tw - 2 * pad)))
    y = int(min(max(y, 0), max(0, h - th - base - 2 * pad)))
    cv2.rectangle(frame, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), bg, -1)
    cv2.putText(frame, text, (x + pad, y + th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, detections, zones, snapshots):
    # Everything is sized from the frame width, so it stays readable after the video is
    # scaled down (our clips are 4K, the saved demo is ~1280 px wide).
    s = frame.shape[1] / 1280.0
    line = max(2, int(round(2 * s)))
    fs = 0.55 * s
    ft = max(1, int(round(1.2 * s)))
    GREEN, ORANGE, RED, WHITE = (60, 200, 60), (0, 165, 255), (60, 60, 255), (255, 255, 255)

    # 1) zone outlines (thin) with a readable label
    for zi, (zone, snap) in enumerate(zip(zones, snapshots)):
        pct = snap["density_pct"]
        color = GREEN if pct < 60 else ORANGE if pct < 80 else RED
        cv2.polylines(frame, [zone.as_np()], isClosed=True, color=color, thickness=max(1, int(round(s))))
        star = "*" if snap.get("free_inferred") else ""
        x0, y0 = zone.polygon[0]
        _label(frame, f"{zone.zone_id}: {snap['vehicle_count']} car / {snap['free_count']} free{star}",
               x0, y0 - int((30 + 34 * (zi % 2)) * s), color, fs * 0.9, ft)   # alternate rows so neighbouring labels do not overlap

    # 2) free bays: translucent fill in the bay's real (slanted) shape
    overlay = frame.copy()
    for snap in snapshots:
        for poly in snap.get("free_slots") or []:
            cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], GREEN)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for snap in snapshots:
        for poly in snap.get("free_slots") or []:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(frame, [pts], True, GREEN, line)
            cx, cy = pts.mean(axis=0).astype(int)
            _label(frame, "free", cx - int(18 * s), cy - int(8 * s), GREEN, fs * 0.8, ft)

    # 3) detections
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        is_free = det["class"] == "free"
        is_moving = bool(det.get("moving"))
        color = MOVING_BOX_COLOR if is_moving else (FREE_BOX_COLOR if is_free else CAR_BOX_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)
        if overlay_options["show_conf"] or is_moving:
            name = "MOVING car" if is_moving else det["class"]
            _label(frame, f"{name} {det['confidence']:.2f}", x1, y1 - int(22 * s), color, fs * 0.75, max(1, ft - 1))

    # 4) summary panel (top-left)
    cars = sum(sn["vehicle_count"] for sn in snapshots)
    free = sum(sn["free_count"] for sn in snapshots)
    est = any(sn.get("free_inferred") for sn in snapshots)
    lines = [("M-Smart Flow", WHITE, 1.15), (f"Parked cars: {cars}", (120, 120, 255), 1.0),
             (f"Free bays: {free}{'*' if est else ''}", (120, 230, 120), 1.0)]
    if est:
        lines.append(("* from bay layout, not model", (190, 190, 190), 0.7))
    pad, gap = int(14 * s), int(10 * s)
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4 * k, ft + 1)[0] for t, _, k in lines]
    w = max(sz[0] for sz in sizes) + 2 * pad
    h = sum(sz[1] for sz in sizes) + gap * (len(lines) - 1) + 2 * pad
    x0, y0 = int(16 * s), int(16 * s)
    panel = frame.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)
    y = y0 + pad
    for (t, c, k), sz in zip(lines, sizes):
        y += sz[1]
        cv2.putText(frame, t, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4 * k, c, ft + 1, cv2.LINE_AA)
        y += gap
    return frame


def run_detection(client, frame, zones, args, motion_pair=None):
    """One full synchronous detection pass on a frame: zones + (optional) moving cars.
    Returns (detections, snapshots)."""
    detections = []
    if zones:
        for zone in zones:
            detections.extend(detect_vehicles_in_zone(client, frame, zone, args))
    else:
        detections = detect_vehicles(client, frame, args.conf)

    res = None
    if args.moving:
        if motion_pair is not None:
            res = residual_motion_map(motion_pair[0], motion_pair[1])
        try:
            detections.extend(detect_cars_full_frame(client, frame, args))
            regions = find_motion_regions(res, frame.shape[1], args.move_thr)
            if regions:
                detections.extend(detect_cars_in_regions(client, frame, regions, args))
        except Exception as e:
            print(f"[WARN] moving-car detection failed: {type(e).__name__}: {e}")
    detections = dedupe_cars(detections)
    if args.moving:
        flag_moving(res, detections, frame.shape[1], args.move_thr)
    parked = [d for d in detections if not d.get("moving")]
    snapshots = assign_to_zones(parked, zones, infer_free=args.infer_free, use_spots=args.spots)
    return detections, snapshots


def render_video(args, client, zones):
    """Offline mode (--render OUT.mp4): go through the video frame by frame, run a
    full detection every --every-seconds, and write an annotated video in the
    ORIGINAL speed. Because detection finishes before the next frame is written,
    the boxes always belong to the frame they are drawn on (no live lag), and
    moving-car detection has proper before/after frames to compare."""
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    start_idx = int(args.start_seconds * fps)
    end_idx = int(args.end_seconds * fps) if args.end_seconds > 0 else (total or 10**9)
    step = max(1, int(round(args.every_seconds * fps)))
    print(f"[RENDER] {args.source}: {fps:.1f} fps, detection every {step} frames "
          f"({args.every_seconds}s), writing {args.render}")

    hist = deque(maxlen=max(args.move_lag, 1) + 1)
    writer = None
    dets, snaps = [], []
    n_det = 0
    last_sample = None
    idx = start_idx
    try:
        while idx < end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if args.moving:
                hist.append(to_motion_gray(frame))

            ready = (not args.moving) or len(hist) == hist.maxlen
            if ready and (last_sample is None or idx - last_sample >= step):
                pair = (hist[0], hist[-1]) if args.moving else None
                try:
                    # Run the (slow) cloud detection in a helper thread and keep the
                    # window alive meanwhile, otherwise Windows shows "Not responding".
                    box = {}

                    def _job(f=frame, p_=pair):
                        try:
                            box["r"] = run_detection(client, f, zones, args, p_)
                        except Exception as ex:   # noqa
                            box["e"] = ex

                    th = threading.Thread(target=_job, daemon=True)
                    th.start()
                    wait_img = draw_overlay(frame.copy(), dets, zones, snaps)
                    cv2.putText(wait_img, f"detecting... {idx / fps:.1f}s", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                    if wait_img.shape[1] > 1000:
                        wait_img = cv2.resize(wait_img, None, fx=1000 / wait_img.shape[1],
                                              fy=1000 / wait_img.shape[1])
                    stop_now = False
                    while th.is_alive():
                        if not args.no_display:
                            cv2.imshow("M-Smart Flow - rendering (q to stop)", wait_img)
                            if cv2.waitKey(50) & 0xFF == ord("q"):
                                stop_now = True
                                break
                        else:
                            th.join(0.1)
                    if stop_now:
                        print("[RENDER] stopped by user - saving what was rendered so far")
                        break
                    if "e" in box:
                        raise box["e"]
                    dets, snaps = box["r"]
                    n_det += 1
                    cars = sum(1 for d in dets if d["class"] != "free")
                    mv = sum(1 for d in dets if d.get("moving"))
                    print(f"[RENDER] t={idx / fps:5.1f}s  detection #{n_det}: {cars} cars"
                          + (f", {mv} moving" if args.moving else "")
                          + (f"  ({idx - start_idx}/{max(end_idx - start_idx, 1)} frames)" if total else ""))
                except Exception as e:
                    print(f"[ERROR] detection failed at t={idx / fps:.1f}s: {type(e).__name__}: {e}")
                last_sample = idx

            out = draw_overlay(frame.copy(), dets, zones, snaps)
            if not snaps:
                for z in zones:
                    cv2.polylines(out, [z.as_np()], True, (255, 255, 0), 2)
            if out.shape[1] > 1280:
                sc = 1280 / float(out.shape[1])
                out = cv2.resize(out, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(args.render, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)

            if not args.no_display:
                cv2.imshow("M-Smart Flow - rendering (q to stop)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[RENDER] stopped by user - saving what was rendered so far")
                    break
            idx += 1
    finally:
        if writer is not None:
            writer.release()
            print(f"[RENDER] Saved: {args.render}")
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


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
    parser.add_argument("--infer-free", action="store_true",
                         help="Estimate free spots as zone capacity minus detected cars when the model does not detect them (shown with a * in the overlay)")
    parser.add_argument("--spots", action="store_true",
                         help="Per-spot mode: split each zone into 'capacity' slots (or use hand-drawn 'spots' in the zones JSON) and draw a green box on each empty slot (marked *, inferred). Overrides --infer-free")
    parser.add_argument("--moving", action="store_true",
                         help="Also detect cars that are driving (anywhere in the frame, even outside zones). They are drawn in magenta and NOT counted as parked. Needs a playing video, so do not combine with --freeze")
    parser.add_argument("--move-thr", type=float, default=1.5,
                         help="Motion threshold for --moving (pixels of leftover motion on a 640 px-wide image). Lower = more sensitive")
    parser.add_argument("--move-lag", type=int, default=6,
                         help="For --moving: compare the frame being detected with the one this many frames earlier")
    parser.add_argument("--freeze", action="store_true",
                         help="Freeze on the first frame (after --start-seconds) and keep detecting it. Best for checking zones/boxes, since zones only fit the frame they were drawn on")
    parser.add_argument("--save-video", default=None, metavar="OUT.mp4",
                         help="Save the annotated output to this video file (real-time length). With --freeze it stops by itself a few seconds after the first result")
    parser.add_argument("--hold-seconds", type=float, default=5.0,
                         help="With --save-video: seconds to keep recording after the detection result is on screen (default 5)")
    parser.add_argument("--render", default=None, metavar="OUT.mp4",
                         help="OFFLINE mode: process the video step by step and write an annotated video at the original speed. Boxes line up exactly with each frame (no live lag). Best option for demo videos; takes a few minutes")
    parser.add_argument("--every-seconds", type=float, default=1.5,
                         help="With --render: run a full detection every this many seconds of video (smaller = smoother boxes, more API calls)")
    parser.add_argument("--end-seconds", type=float, default=0.0,
                         help="With --render: stop at this time in the video (0 = until the end)")
    parser.add_argument("--events-out", default="zone_events.jsonl")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=5,
                         help="How often (in frames) to grab a frame for background detection")
    parser.add_argument("--start-seconds", type=float, default=0.0,
                         help="Skip ahead this many seconds before starting (use to skip past intro/ceiling footage)")
    parser.add_argument("--weights", default=None, metavar="MODEL.pt",
                         help="Run OUR trained weights locally (e.g. parking_v3_last.pt) instead of the Roboflow cloud. No API key needed. Uses --conf 0.15 unless you set --conf")
    parser.add_argument("--show-conf", action="store_true", help="Draw the confidence score on every car box (hidden by default: cleaner video)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for local weights (our models were trained at 640)")
    args = parser.parse_args()

    overlay_options["show_conf"] = args.show_conf
    if args.weights:
        from ultralytics import YOLO
        local_model["model"] = YOLO(args.weights)
        local_model["imgsz"] = args.imgsz
        if "--conf" not in sys.argv:
            args.conf = 0.15            # best threshold on our holdout clips
        args.car_model = ""             # the cloud helper detector is not used in local mode
        atexit.register(print_local_speed)
        client = None
        print(f"[INFO] LOCAL mode: {args.weights} (conf {args.conf}, imgsz {args.imgsz}); no cloud calls")
    else:
        api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("No API key found. Set it with: set ROBOFLOW_API_KEY=your_key_here  (or use --weights MODEL.pt to run locally)")

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

    if args.render:
        render_video(args, client, zones)
        return

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
    state = {"detections": [], "snapshots": [], "latest_frame": None, "frame_count": 0,
             "motion_pair": None}
    state_lock = threading.Lock()
    stop_event = threading.Event()

    def detection_worker():
        while not stop_event.is_set():
            with state_lock:
                frame = state["latest_frame"]
                state["latest_frame"] = None
                motion_pair = state["motion_pair"]
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                if zones:
                    detections = []
                    for zone in zones:
                        zone_dets = detect_vehicles_in_zone(client, frame, zone, args)
                        print(f"[zone] {zone.zone_id}: {len(zone_dets)} raw detections")
                        detections.extend(zone_dets)
                else:
                    detections = detect_vehicles(client, frame, args.conf)
            except Exception as e:
                print(f"[ERROR] detection call failed: {type(e).__name__}: {e}")
                # clear the frame so we don't just hammer the API with the same
                # failing frame in a tight loop, and back off briefly
                with state_lock:
                    state["latest_frame"] = None
                    state["submitted"] = False
                time.sleep(1.0)
                continue

            try:
                res = None
                if args.moving:
                    if motion_pair is not None:
                        res = residual_motion_map(motion_pair[0], motion_pair[1])
                    try:
                        # 1) whole-frame look for cars anywhere in the picture
                        detections.extend(detect_cars_full_frame(client, frame, args))
                        # 2) zoom in on everything that moves by itself, anywhere in the
                        #    frame, and let the car detector confirm what it is
                        regions = find_motion_regions(res, frame.shape[1], args.move_thr)
                        if regions:
                            detections.extend(detect_cars_in_regions(client, frame, regions, args))
                    except Exception as e:
                        print(f"[WARN] moving-car detection failed: {type(e).__name__}: {e}")
                detections = dedupe_cars(detections)
                if args.moving:
                    flag_moving(res, detections, frame.shape[1], args.move_thr)
                # moving cars are driving through, so they do not count as parked
                parked = [d for d in detections if not d.get("moving")]
                snapshots = assign_to_zones(parked, zones, infer_free=args.infer_free,
                                            use_spots=args.spots)
            except Exception:
                import traceback
                traceback.print_exc()
                with state_lock:
                    state["latest_frame"] = None
                    state["submitted"] = False
                time.sleep(1.0)
                continue

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
            moving_total = sum(1 for d in detections if d.get("moving"))
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
                "moving_total": moving_total,
                "zones": [{k: v for k, v in s_.items() if k != "free_slots"} for s_ in snapshots],
            }
            events_file.write(json.dumps(event) + "\n")
            events_file.flush()
            print(f"[detection {state['frame_count']}] {raw_car_total} cars, {raw_free_total} free spots total "
                  f"(in zones: {zone_car_total} cars, {zone_free_total} free)"
                  + (f", {moving_total} moving" if args.moving else ""))

    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()
    print("[INFO] Detection worker thread started, waiting for first frame...")

    gray_hist = deque(maxlen=max(args.move_lag, 1) + 1)
    writer = None
    rec_start = time.time()
    rec_written = 0
    REC_FPS = 15
    done_at = None   # time the first detection result appeared (for --save-video auto-stop)
    try:
        frame_index = 0
        frame = None
        while True:
            if frame is None or not args.freeze:
                ok, new_frame = cap.read()
                if not ok:
                    if frame is None:
                        break
                    # Video ended: keep showing the last frame (and keep detecting it)
                    # until 'q' is pressed, so slow cloud results are not missed.
                    print("[INFO] Video ended. Holding last frame - press q in the video window to quit.")
                    args.freeze = True
                else:
                    frame = new_frame
            frame_index += 1

            cur_gray = None
            if args.moving:
                cur_gray = to_motion_gray(frame)
                gray_hist.append(cur_gray)

            if frame_index % args.frame_skip == 0 or frame_index == 1:
                with state_lock:
                    if args.freeze and state.get("submitted"):
                        pass  # frozen: one result is enough, avoid re-calling the API
                    elif state["latest_frame"] is None:
                        state["latest_frame"] = frame.copy()
                        state["submitted"] = True
                        if args.moving:
                            state["motion_pair"] = (gray_hist[0], cur_gray)

            if not args.no_display:
                with state_lock:
                    detections = state["detections"]
                    snapshots = state["snapshots"]

                annotated = draw_overlay(frame.copy(), detections, zones, snapshots)
                if not snapshots:
                    for z in zones:
                        cv2.polylines(annotated, [z.as_np()], True, (255, 255, 0), 2)
                    cv2.putText(annotated, "detecting...", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                max_display_width = 1000
                if annotated.shape[1] > max_display_width:
                    scale = max_display_width / annotated.shape[1]
                    annotated = cv2.resize(annotated, None, fx=scale, fy=scale)

                cv2.namedWindow("M-Smart Flow - Vehicle Detection", cv2.WINDOW_NORMAL)
                cv2.imshow("M-Smart Flow - Vehicle Detection", annotated)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            else:
                annotated = draw_overlay(frame.copy(), state["detections"], zones, state["snapshots"])
                time.sleep(0.03)

            if args.save_video:
                if writer is None:
                    h_, w_ = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             REC_FPS, (w_, h_))
                    print(f"[INFO] Recording to {args.save_video}")
                due = int((time.time() - rec_start) * REC_FPS) - rec_written
                for _ in range(max(due, 0)):
                    writer.write(annotated)
                    rec_written += 1
                if state["snapshots"] and done_at is None:
                    done_at = time.time()
                if done_at is not None and args.freeze and time.time() - done_at >= args.hold_seconds:
                    print("[INFO] Recording finished.")
                    break
    finally:
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved video: {args.save_video}")
        stop_event.set()
        cap.release()
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()