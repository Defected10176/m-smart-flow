"""
Quick experiment: run the plate-character model on our own images and print what it "reads".

Usage (project folder, venv active):
    python test_plate.py                                   # all frame_*.jpg and dataset_additions crops
    python test_plate.py --images "jpg_files/frame_IMG_8844_*.jpg" --imgsz 1280 --conf 0.2

Characters are sorted left-to-right (single-line plates only; Thai plates often have two
lines, so this is only a rough read). Annotated images go to plate_test_out/.
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="plate_chars_yolo26n.pt")
p.add_argument("--images", default="dataset_additions/images/*.jpg")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--out", default="plate_test_out")
p.add_argument("--limit", type=int, default=60)
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
model = YOLO(a.weights)
files = sorted(glob.glob(a.images))[: a.limit]
if not files:
    raise SystemExit(f"No images match {a.images}")
print("classes:", len(model.names), "e.g.", list(model.names.values())[:12])
hits = 0
for f in files:
    r = model.predict(f, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
    chars = sorted(((float(b.xyxy[0][0]), r.names[int(b.cls)], float(b.conf)) for b in r.boxes))
    text = "".join(c for _, c, _ in chars)
    if chars:
        hits += 1
    print(f"{os.path.basename(f):44} {len(chars):>3} chars  {text}")
    cv2.imwrite(os.path.join(a.out, os.path.basename(f)), r.plot())
print(f"\n{hits}/{len(files)} images had any detection. Look at plate_test_out/ to judge by eye.")
