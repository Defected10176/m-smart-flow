"""
Spot Calibrator - draw each parking BAY by hand inside zones you already made.

Why: with --spots the detector marks a bay free when no car sits in it. If a zone has no hand-drawn
bays, the script cuts the zone into equal vertical strips, which ignores perspective and looks wrong.
Bays drawn here are saved as "spots" in the zones JSON and used instead of the strips.

Usage:
    python spot_calibrator.py --source IMG_8844.MOV --zones zones_8844.json --seconds 8
    -> writes zones_8844_spots.json (your original file is not changed)

Then run detection with  --zones zones_8844_spots.json --spots

Controls (click on the video frame):
    left click   add a corner. After 4 corners the bay is finished automatically
    b            finish a bay that has 3 corners (a triangle) - rarely needed
    u            undo last corner (or, if none, delete the last finished bay of this zone)
    c            clear the corners of the bay you are drawing
    r            delete ALL bays of the current zone (back to equal strips)
    z / x        next / previous zone (only changes which outline is highlighted: on save,
                 every bay is filed under the zone it sits in, so it does not matter which one is active)
    s            save and exit
    q / ESC      quit without saving

How to draw a bay: trace the area a PARKED CAR would cover - roughly the car's body, with the
bottom edge on the floor line. Draw it a little taller than the floor marking, because a bay counts
as occupied when the detected car's centre or bottom-centre falls inside it.
Draw the bays of one zone left to right (the order does not matter, but it is tidier).
"""
import argparse
import json
import os

import cv2
import numpy as np


def grab_frame(path, seconds):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target = int(seconds * fps)
    frame, idx = None, 0
    while idx <= target:          # sequential read: reliable for iPhone HEVC .MOV
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Could not read frame {idx}; pick a smaller --seconds")
        idx += 1
    cap.release()
    return frame


def main():
    ap = argparse.ArgumentParser(description="Draw parking bays inside existing zones")
    ap.add_argument("--source", required=True)
    ap.add_argument("--zones", required=True, help="existing zones JSON for this video")
    ap.add_argument("--out", default=None, help="default: <zones file>_spots.json")
    ap.add_argument("--seconds", type=float, default=8.0, help="use the same time you used for the zones")
    ap.add_argument("--max-display-width", type=int, default=1400)
    args = ap.parse_args()
    out_path = args.out or os.path.splitext(args.zones)[0] + "_spots.json"

    data = json.load(open(args.zones))
    zones = data["zones"]
    if not zones:
        raise SystemExit("No zones in that file.")
    for z in zones:
        z["spots"] = [list(map(list, s)) for s in (z.get("spots") or [])]

    frame = grab_frame(args.source, args.seconds)
    H, W = frame.shape[:2]
    scale = min(1.0, args.max_display_width / W)
    base = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
    print(f"[INFO] frame {W}x{H}, shown at scale {scale:.3f}")

    state = {"zi": 0, "pts": []}
    window = "Spot Calibrator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def finish_bay():
        z = zones[state["zi"]]
        z["spots"].append([[int(px / scale), int(py / scale)] for px, py in state["pts"]])
        print(f"[OK] bay {len(z['spots'])} added to {z['zone_id']}")
        state["pts"] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((x, y))
            if len(state["pts"]) == 4:
                finish_bay()

    cv2.setMouseCallback(window, on_mouse)

    def to_disp(poly):
        return np.array([[int(px * scale), int(py * scale)] for px, py in poly], dtype=np.int32)

    def redraw():
        canvas = base.copy()
        for i, z in enumerate(zones):
            active = i == state["zi"]
            cv2.polylines(canvas, [to_disp(z["polygon"])], True, (0, 255, 255) if active else (0, 140, 0), 3 if active else 1)
            for n, s in enumerate(z["spots"], 1):
                pts = to_disp(s)
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (255, 200, 0) if active else (120, 120, 0))
                canvas[:] = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
                cv2.polylines(canvas, [pts], True, (255, 200, 0) if active else (120, 120, 0), 2)
                c = pts.mean(axis=0).astype(int)
                cv2.putText(canvas, str(n), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if state["pts"]:
            p = np.array(state["pts"], dtype=np.int32)
            cv2.polylines(canvas, [p], False, (0, 165, 255), 2)
            for q in state["pts"]:
                cv2.circle(canvas, q, 5, (0, 165, 255), -1)
        z = zones[state["zi"]]
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(canvas, f"Zone {state['zi']+1}/{len(zones)}: {z['zone_id']} | bays drawn: {len(z['spots'])} | original capacity: {z.get('capacity')}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(canvas, "click 4 corners per bay | u undo | c clear | r reset zone | z/x next/prev zone | s save | q quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, canvas)

    print("[INFO] Click 4 corners for each bay of the yellow-outlined zone. z = next zone, s = save.")
    while True:
        redraw()
        key = cv2.waitKey(30) & 0xFF
        z = zones[state["zi"]]
        if key == ord("u"):
            if state["pts"]:
                state["pts"].pop()
            elif z["spots"]:
                z["spots"].pop(); print(f"[INFO] removed last bay of {z['zone_id']}")
        elif key == ord("c"):
            state["pts"] = []
        elif key == ord("b"):
            if len(state["pts"]) >= 3:
                finish_bay()
        elif key == ord("r"):
            z["spots"] = []; state["pts"] = []; print(f"[INFO] cleared all bays of {z['zone_id']}")
        elif key == ord("z"):
            state["zi"] = (state["zi"] + 1) % len(zones); state["pts"] = []
        elif key == ord("x"):
            state["zi"] = (state["zi"] - 1) % len(zones); state["pts"] = []
        elif key == ord("s"):
            # Re-file every bay by WHERE it is drawn, so it does not matter which zone was active.
            all_bays = [b for zz in zones for b in zz["spots"]]
            for zz in zones:
                zz["spots"] = []
            for b in sorted(all_bays, key=lambda p: np.mean([q[0] for q in p])):
                cx, cy = float(np.mean([q[0] for q in b])), float(np.mean([q[1] for q in b]))
                best, bd = None, -1e18
                for zz in zones:      # inside a zone outline wins; otherwise the nearest outline
                    d = cv2.pointPolygonTest(np.array(zz["polygon"], dtype=np.int32), (cx, cy), True)
                    if d > bd:
                        best, bd = zz, d
                if bd < 0:
                    print(f"[WARN] bay at ({cx:.0f},{cy:.0f}) is outside every zone; put in nearest zone '{best['zone_id']}' ({-bd:.0f}px away)")
                best["spots"].append(b)
            for zz in zones:
                if zz["spots"]:
                    zz["capacity"] = len(zz["spots"])     # capacity now = number of bays drawn
                else:
                    zz.pop("spots", None)                 # untouched zone: keeps the equal-strip fallback
                    zz["capacity"] = zz.get("capacity", 1)
            json.dump({"zones": zones}, open(out_path, "w"), indent=2)
            print(f"[OK] saved {out_path}")
            for zz in zones:
                print(f"   {zz['zone_id']:14} {len(zz.get('spots', []))} bays")
            print(f"Run detection with:  --zones {out_path} --spots")
            break
        elif key in (27, ord("q")):
            print("[INFO] quit without saving"); break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
