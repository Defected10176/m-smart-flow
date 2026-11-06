"""
Quick check of the Colab-trained parking model (parking_yolo26n.pt) on our own frames.

Usage (from the project folder, venv active):
    python test_local_weights.py                      # all jpgs in jpg_files/frame_*.jpg
    python test_local_weights.py --conf 0.25 --imgsz 1280

Output: annotated images in local_test_out/ and a table of car/free counts + speed per frame.
Look at the images yourself: count how many cars the model missed or invented.
"""
import argparse
import glob
import json
import os
import time

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="parking_yolo26n.pt")
p.add_argument("--images", default="jpg_files/frame_*.jpg")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--out", default="local_test_out")
p.add_argument("--zones", default=None,
               help="zones JSON for ONE video (e.g. zones_8844.json). Crops each zone and detects inside the crop. Use with images of that same video only")
p.add_argument("--pad", type=int, default=80, help="pixels of padding around each zone crop")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
model = YOLO(args.weights)
files = sorted(glob.glob(args.images))
if not files:
    raise SystemExit(f"No images match {args.images}")

model.predict(cv2.imread(files[0]), conf=args.conf, imgsz=args.imgsz, verbose=False)  # warm-up
print(f"{'image':34} {'car':>4} {'free':>5} {'ms':>7}")
times = []
zones = json.load(open(args.zones))["zones"] if args.zones else None

def run(img):
    """Returns (list of (cls, conf, x1, y1, x2, y2) in full-frame coords, ms)."""
    out = []
    t = time.perf_counter()
    if zones is None:
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        for b in r.boxes:
            out.append((r.names[int(b.cls)], float(b.conf), *map(int, b.xyxy[0].tolist())))
    else:
        H, W = img.shape[:2]
        for z in zones:
            xs = [q[0] for q in z["polygon"]]; ys = [q[1] for q in z["polygon"]]
            x1, y1 = max(0, min(xs) - args.pad), max(0, min(ys) - args.pad)
            x2, y2 = min(W, max(xs) + args.pad), min(H, max(ys) + args.pad)
            r = model.predict(img[y1:y2, x1:x2], conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            for b in r.boxes:
                a = b.xyxy[0].tolist()
                out.append((r.names[int(b.cls)], float(b.conf), int(a[0]) + x1, int(a[1]) + y1, int(a[2]) + x1, int(a[3]) + y1))
    return out, (time.perf_counter() - t) * 1000

for f in files:
    img = cv2.imread(f)
    dets, ms = run(img)
    times.append(ms)
    names = [d[0] for d in dets]
    print(f"{os.path.basename(f):34} {names.count('car'):>4} {names.count('free'):>5} {ms:>7.1f}")
    for c, cf, x1, y1, x2, y2 in dets:
        col = (0, 0, 255) if c == "car" else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 4)
        cv2.putText(img, f"{c} {cf:.2f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    cv2.imwrite(os.path.join(args.out, os.path.basename(f)), cv2.resize(img, None, fx=0.5, fy=0.5))

avg = sum(times) / len(times)
print(f"\n{len(files)} images | avg {avg:.1f} ms/img = {1000/avg:.1f} FPS (this machine, imgsz {args.imgsz})")
