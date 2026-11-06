"""
Build our own training images + draft labels from the parking clips.

Usage (project folder, venv active):
    python make_dataset.py                       # every IMG_*.MOV, a frame every 2 s
    python make_dataset.py --every 3             # fewer frames
    python make_dataset.py --holdout none        # put ALL clips in the training set

For each clip it saves:
  - one close-up crop per parking zone (from zones_<number>.json), or
  - overlapping 1280x720 tiles when that clip has no zones file
Draft labels (YOLO .txt) are made with parking_yolo26n.pt. They are only a starting
point: open them in a labeling tool and fix every image.

Output:
  dataset_additions/images/*.jpg      -> copy into  train/images  (and some into valid/images)
  dataset_additions/labels/*.txt      -> copy into  train/labels   (same names)
  holdout_test/images, holdout_test/labels   -> NEVER train on these; honest test set
  classes.txt                         -> class order the labels use (must match data.yaml)
"""
import argparse
import glob
import json
import os
import re

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="parking_yolo26n.pt")
p.add_argument("--every", type=float, default=2.0, help="seconds between frames")
p.add_argument("--pad", type=int, default=80, help="pixels around each zone crop")
p.add_argument("--conf", type=float, default=0.1, help="draft-label confidence")
p.add_argument("--holdout", nargs="*", default=["8845", "8847"],
               help="clip numbers kept for testing, or 'none'")
p.add_argument("--no-prelabel", action="store_true", help="only save images, no draft labels")
args = p.parse_args()
holdout = [] if args.holdout == ["none"] else args.holdout

model = None if args.no_prelabel else YOLO(args.weights)
if model is not None:
    with open("classes.txt", "w") as f:
        f.write("\n".join(model.names[i] for i in sorted(model.names)))
    print("classes:", model.names)


def crops_for(img, num):
    """Yield (suffix, x_offset, y_offset, crop)."""
    H, W = img.shape[:2]
    zf = f"zones_{num}.json"
    if os.path.exists(zf):
        for z in json.load(open(zf))["zones"]:
            xs = [q[0] for q in z["polygon"]]; ys = [q[1] for q in z["polygon"]]
            x1, y1 = max(0, min(xs) - args.pad), max(0, min(ys) - args.pad)
            x2, y2 = min(W, max(xs) + args.pad), min(H, max(ys) + args.pad)
            if x2 - x1 > 50 and y2 - y1 > 50:
                yield z["zone_id"], x1, y1, img[y1:y2, x1:x2]
    else:
        tw, th, sx, sy = 1280, 720, 960, 540
        for yy in range(0, max(1, H - th + sy), sy):
            for xx in range(0, max(1, W - tw + sx), sx):
                x1, y1 = min(xx, max(0, W - tw)), min(yy, max(0, H - th))
                yield f"tile{y1}_{x1}", x1, y1, img[y1:y1 + th, x1:x1 + tw]


def save(crop, base, out_dir):
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    cv2.imwrite(f"{out_dir}/images/{base}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    if model is not None:
        r = model.predict(crop, conf=args.conf, imgsz=960, verbose=False)[0]
        h, w = crop.shape[:2]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            lines.append(f"{int(b.cls)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    open(f"{out_dir}/labels/{base}.txt", "w").write("\n".join(lines))


videos = sorted({os.path.normcase(v): v for v in glob.glob("IMG_*.MOV") + glob.glob("IMG_*.mov")}.values())
total = {"train": 0, "holdout": 0}
for v in videos:
    num = re.search(r"IMG_(\d+)", v).group(1)
    out_dir = "holdout_test" if num in holdout else "dataset_additions"
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        print("[SKIP]", v); continue
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(args.every * fps))
    idx, n = 0, 0
    while True:
        if idx % step != 0:
            if not cap.grab():      # skip frames without converting them (faster)
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if True:
            t = int(idx / fps)
            for suffix, _, _, crop in crops_for(frame, num):
                save(crop, f"my_{num}_t{t:03d}_{suffix}", out_dir)
                n += 1
        idx += 1
    cap.release()
    total["holdout" if out_dir == "holdout_test" else "train"] += n
    print(f"{v}: {n} crops -> {out_dir}")

print(f"\nDone. training additions: {total['train']} | holdout test: {total['holdout']}")
print("Next: fix the draft labels in a labeling tool (labels/ must match images/ names).")
