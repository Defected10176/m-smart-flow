"""
Turn bays drawn in X-AnyLabeling into "spots" inside a zones file.

1) Make a frame:        python grab_frame.py --source IMG_8844.MOV --seconds 8 --out spotwork/frame_8844.jpg
2) Open the spotwork folder in X-AnyLabeling and draw one POLYGON per parking bay (any label, e.g. "bay").
   X-AnyLabeling saves spotwork/frame_8844.json next to the picture.
3) Run:                 python spots_from_xanylabeling.py --labels spotwork/frame_8844.json --zones zones_8844.json
   -> writes zones_8844_spots.json (your original zones file is not changed)

Each bay goes to the zone whose outline contains the bay's centre, so the label name does not matter.
"""
import argparse, json, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True, help="X-AnyLabeling .json for the frame")
ap.add_argument("--zones", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.splitext(a.zones)[0] + "_spots.json"

zones = json.load(open(a.zones))["zones"]
shapes = [s for s in json.load(open(a.labels))["shapes"] if s.get("shape_type") in ("polygon", "rectangle") and len(s["points"]) >= 3]
for z in zones:
    z["spots"] = []

def centre(pts): return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
unassigned = 0
for s in sorted(shapes, key=lambda s: centre(s["points"])[0]):          # left to right
    cx, cy = centre(s["points"])
    pts = [[int(round(x)), int(round(y))] for x, y in s["points"]]
    for z in zones:
        if cv2.pointPolygonTest(np.array(z["polygon"], dtype=np.int32), (cx, cy), False) >= 0:
            z["spots"].append(pts); break
    else:
        unassigned += 1
        print(f"[WARN] a bay centred at ({cx:.0f},{cy:.0f}) is outside every zone outline - skipped")

for z in zones:
    if z["spots"]:
        z["capacity"] = len(z["spots"])
    else:
        z.pop("spots")
json.dump({"zones": zones}, open(out, "w"), indent=2)
print(f"{len(shapes) - unassigned} bays assigned -> {out}")
for z in zones:
    print(f"  {z['zone_id']:14} {len(z.get('spots', []))} bays" + ("" if z.get("spots") else "  (none drawn: keeps equal strips)"))
print(f"\nRun detection with:  --zones {out} --spots")
