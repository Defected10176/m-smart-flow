"""
Zone Calibrator — click to define parking zone polygons on a real camera frame.

Grabs a single frame from your video, shows it in a window, and lets you
click points to trace the outline of each parking zone. Saves the result in
the same format vehicle_detection_api.py / vehicle_detection.py expect.

Usage:
    python zone_calibrator.py --source IMG_8838.MOV --seconds 8 --out zones_calibrated.json

Controls:
    Left click   - add a point to the zone outline you're currently tracing
    u            - undo last point
    c            - clear current (unfinished) zone outline
    n            - finish current zone (prompts for a name + capacity in the terminal)
    s            - save all finished zones to --out and exit
    q / ESC      - quit without saving

Tips:
    - Pick a --seconds timestamp where the camera view is stable and zones
      are clearly visible (not ceiling/panning footage).
    - Trace loosely around where cars actually park in that area — it
      doesn't need to be pixel-perfect, just roughly matching the real rows.
    - You can define multiple zones in one run — repeat click-trace-'n' for each.
"""
import argparse
import json

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Click to define parking zone polygons on a real camera frame")
    parser.add_argument("--source", required=True, help="Video file to grab a reference frame from")
    parser.add_argument("--seconds", type=float, default=5.0,
                         help="Timestamp to grab the frame from (pick a moment with zones clearly visible)")
    parser.add_argument("--out", default="zones_calibrated.json")
    parser.add_argument("--max-display-width", type=int, default=1400)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    print(f"[INFO] {args.source}: {fps:.2f} fps, {total_frames} frames, ~{duration:.1f}s duration")
    if args.seconds > duration:
        raise SystemExit(f"[ERROR] Requested {args.seconds}s but video is only ~{duration:.1f}s long. "
                          f"Pick a smaller --seconds value.")

    # Sequential read (not seek) — reliable for HEVC/iPhone .MOV files.
    target_frame = int(args.seconds * fps)
    frame = None
    idx = 0
    while idx <= target_frame:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} (target was {target_frame})")
        idx += 1
    cap.release()

    orig_h, orig_w = frame.shape[:2]
    scale = min(1.0, args.max_display_width / orig_w)
    base_display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
    print(f"[INFO] Frame is {orig_w}x{orig_h}, displaying at scale {scale:.3f}")

    zones = []
    current_points = []  # in DISPLAY coords; converted to original-frame coords on save

    window = "Zone Calibrator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    instructions = [
        "click: add point   u: undo   c: clear zone   n: finish zone (name in terminal)",
        "s: save all & exit   q/ESC: quit without saving",
    ]

    def redraw():
        canvas = base_display.copy()
        for i, line in enumerate(instructions):
            cv2.putText(canvas, line, (10, 25 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        for z in zones:
            pts = np.array([[int(px * scale), int(py * scale)] for px, py in z["polygon"]], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
            cx, cy = pts[0]
            cv2.putText(canvas, z["zone_id"], (cx, max(cy - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if current_points:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 165, 255), thickness=2)
            for p in current_points:
                cv2.circle(canvas, p, 4, (0, 165, 255), -1)

        cv2.imshow(window, canvas)

    print("\n[INFO] Click points to trace a parking zone's outline, then press 'n' to name and save it.")
    print("[INFO] Repeat for each zone. Press 's' when done with all zones.\n")

    while True:
        redraw()
        key = cv2.waitKey(30) & 0xFF

        if key == ord('u'):
            if current_points:
                current_points.pop()

        elif key == ord('c'):
            current_points = []
            print("[INFO] Cleared current (unfinished) zone.")

        elif key == ord('n'):
            if len(current_points) < 3:
                print("[WARN] Need at least 3 points to define a zone polygon — keep clicking.")
                continue
            zone_id = input("Zone ID (e.g. floor2_zoneA): ").strip() or f"zone_{len(zones) + 1}"
            capacity_raw = input(f"Capacity for {zone_id} (number of parking spots in this zone): ").strip()
            try:
                capacity = int(capacity_raw) if capacity_raw else 1
            except ValueError:
                capacity = 1
            # Convert display coords back to the original frame's pixel coords —
            # this is what the detection scripts actually run on.
            orig_points = [[int(px / scale), int(py / scale)] for px, py in current_points]
            zones.append({"zone_id": zone_id, "polygon": orig_points, "capacity": capacity})
            print(f"[OK] Saved zone '{zone_id}' with {len(orig_points)} points, capacity {capacity}.\n")
            current_points = []

        elif key == ord('s'):
            if not zones:
                print("[WARN] No finished zones yet — nothing to save.")
                continue
            with open(args.out, "w") as f:
                json.dump({"zones": zones}, f, indent=2)
            print(f"[OK] Saved {len(zones)} zone(s) to {args.out}")
            break

        elif key in (27, ord('q')):
            print("[INFO] Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
