"""
Where does the model miss? Compares predictions with our hand-checked holdout labels.

Usage (project venv):
    python diagnose_holdout.py parking_v3_last.pt            # conf 0.25
    python diagnose_holdout.py parking_v3_last.pt --conf 0.1

Prints car recall by box size (small / medium / large) and false-positive counts, and saves pictures to diagnose_out/:
  green = labeled car found, YELLOW = labeled car MISSED, red = prediction with no matching label.
"""
import argparse, glob, os
import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("--data", default="holdout_final")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()
os.makedirs("diagnose_out", exist_ok=True)
model = YOLO(a.weights)

def iou(x, y):
    w = min(x[2], y[2]) - max(x[0], y[0]); h = min(x[3], y[3]) - max(x[1], y[1])
    i = max(0, w) * max(0, h); u = (x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - i
    return i / u if u else 0

buckets = {"small (<1% of image)": [0, 0], "medium (1-4%)": [0, 0], "large (>4%)": [0, 0]}
def bucket(frac): return "small (<1% of image)" if frac < 0.01 else "medium (1-4%)" if frac < 0.04 else "large (>4%)"
tp = fn = fp = 0
per_image = []
for f in sorted(glob.glob(f"{a.data}/images/*.jpg")):
    img = cv2.imread(f); H, W = img.shape[:2]
    base = os.path.splitext(os.path.basename(f))[0]
    gts = []
    for line in open(f"{a.data}/labels/{base}.txt"):
        q = line.split()
        if len(q) == 5 and int(q[0]) == 0:
            x, y, w, h = map(float, q[1:])
            gts.append(((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H, w*h))
    r = model.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
    preds = [tuple(b.xyxy[0].tolist()) for b in r.boxes if int(b.cls) == 0]
    used = set(); miss_here = 0
    for g in gts:
        best, bj = 0, None
        for j, pr in enumerate(preds):
            if j in used: continue
            v = iou(g[:4], pr)
            if v > best: best, bj = v, j
        b = bucket(g[4]); buckets[b][1] += 1
        if best >= 0.5:
            used.add(bj); buckets[b][0] += 1; tp += 1
            cv2.rectangle(img, tuple(map(int, g[:2])), tuple(map(int, g[2:4])), (0, 200, 0), 3)
        else:
            fn += 1; miss_here += 1
            cv2.rectangle(img, tuple(map(int, g[:2])), tuple(map(int, g[2:4])), (0, 255, 255), 4)
    for j, pr in enumerate(preds):
        if j not in used:
            fp += 1; cv2.rectangle(img, (int(pr[0]), int(pr[1])), (int(pr[2]), int(pr[3])), (0, 0, 255), 3)
    per_image.append((miss_here, os.path.basename(f)))
    cv2.imwrite(f"diagnose_out/{base}.jpg", img)
print(f"\n{a.weights} @ conf {a.conf}: labeled cars found {tp}, missed {fn}, extra boxes (no label) {fp}")
print(f"recall {tp/max(1,tp+fn):.2f} | precision {tp/max(1,tp+fp):.2f}")
for k, (hit, tot) in buckets.items():
    print(f"  {k:24} {hit}/{tot} found" + (f" ({hit/tot:.0%})" if tot else ""))
print("images with most misses:", sorted(per_image, reverse=True)[:6])
print("pictures saved in diagnose_out/ (yellow = missed)")
