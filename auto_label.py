"""
Auto-draft car boxes with a general-purpose COCO detector (much better at cars from any angle
than our parking model), keep the 'free' boxes from our own draft labels, and write
X-AnyLabeling files into a NEW folder so your manual work in labeling_work/ is untouched.

Usage (project folder, venv active, NOT the labelenv one):
    python auto_label.py                      # uses yolo26x.pt (downloads once), falls back to yolo26n.pt
    python auto_label.py --model yolo26l.pt --conf 0.2

Output: labeling_auto/  (images + .json) -> open this folder in X-AnyLabeling and REVIEW every image.
Auto labels are a draft: expect some missed or extra boxes. Fix them, then export to YOLO.
"""
import argparse
import glob
import json
import os
import shutil

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolo26x.pt")
p.add_argument("--src", default="labeling_work")
p.add_argument("--dst", default="labeling_auto")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=1280)
a = p.parse_args()

try:
    model = YOLO(a.model)
except Exception as e:
    print(f"[WARN] could not load {a.model} ({e}); falling back to yolo26n.pt")
    model = YOLO("yolo26n.pt")

CAR_CLASSES = {"car", "truck", "bus"}     # COCO names -> all become our single 'car' class
os.makedirs(a.dst, exist_ok=True)

# Images you already corrected by hand (their .json changed after it was first generated) are kept as-is.
_js = sorted(glob.glob(f"{a.src}/*.json"), key=os.path.getmtime)
_t0 = os.path.getmtime(_js[0]) if _js else 0
edited = {os.path.splitext(os.path.basename(f))[0] for f in _js if os.path.getmtime(f) - _t0 > 120}
print(f"keeping {len(edited)} images you already edited by hand")
n = cars = free = kept = 0
for img_path in sorted(glob.glob(f"{a.src}/*.jpg")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    shutil.copy2(img_path, f"{a.dst}/{base}.jpg")
    if base in edited:
        shutil.copy2(f"{a.src}/{base}.json", f"{a.dst}/{base}.json")
        kept += 1
        continue
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    shapes = []

    def add(label, x1, y1, x2, y2, score=None):
        shapes.append({"label": label, "score": score, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                       "group_id": None, "description": "", "difficult": False, "shape_type": "rectangle",
                       "flags": {}, "attributes": {}, "kie_linking": []})

    r = model.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
    for b in r.boxes:
        if r.names[int(b.cls)] in CAR_CLASSES:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            add("car", x1, y1, x2, y2, float(b.conf)); cars += 1

    old = f"{a.src}/{base}.json"                      # keep 'free' boxes from the earlier draft
    if os.path.exists(old):
        for s in json.load(open(old)).get("shapes", []):
            if s["label"] == "free":
                shapes.append(s); free += 1

    json.dump({"version": "3.0.0", "flags": {}, "shapes": shapes, "imagePath": f"{base}.jpg",
               "imageData": None, "imageHeight": h, "imageWidth": w},
              open(f"{a.dst}/{base}.json", "w"), indent=2)
    n += 1
print(f"{n} images auto-labeled + {kept} hand-edited kept -> {a.dst}/ | {cars} auto car boxes, {free} free boxes kept from the earlier draft")
