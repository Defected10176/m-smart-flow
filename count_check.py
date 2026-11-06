"""
Zone-level check: does the model get the NUMBER of cars right in each holdout crop?

Usage (project venv):
    python count_check.py parking_v3_last.pt
    python count_check.py parking_yolo26n.pt parking_v3_last.pt --conf 0.25

Compares predicted car count with the hand-labeled car count for each crop in holdout_final/.
Prints: exact-match %, within-1 %, mean absolute error, and a per-clip breakdown.
"""
import argparse, glob, os, re
from collections import defaultdict
import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("weights", nargs="+")
p.add_argument("--data", default="holdout_final")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()

files = sorted(glob.glob(f"{a.data}/images/*.jpg"))
truth = {}
for f in files:
    base = os.path.splitext(os.path.basename(f))[0]
    truth[f] = sum(1 for l in open(f"{a.data}/labels/{base}.txt") if l.split() and l.split()[0] == "0")

print(f"{len(files)} crops, {sum(truth.values())} labeled cars in total (avg {sum(truth.values())/len(files):.1f} per crop)\n")
for w in a.weights:
    m = YOLO(w)
    rows = []
    for f in files:
        r = m.predict(cv2.imread(f), conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        pred = sum(1 for b in r.boxes if int(b.cls) == 0)
        rows.append((os.path.basename(f), truth[f], pred))
    n = len(rows)
    exact = sum(t == q for _, t, q in rows); w1 = sum(abs(t - q) <= 1 for _, t, q in rows)
    mae = sum(abs(t - q) for _, t, q in rows) / n
    under = sum(q < t for _, t, q in rows); over = sum(q > t for _, t, q in rows)
    print(f"== {os.path.basename(w)} @ conf {a.conf}")
    print(f"   exact count: {exact}/{n} ({exact/n:.0%}) | within 1 car: {w1}/{n} ({w1/n:.0%}) | mean error: {mae:.2f} cars/crop")
    print(f"   too few cars in {under} crops, too many in {over} crops")
    clips = defaultdict(list)
    for name, t, q in rows: clips[re.match(r"my_(\d+)_", name).group(1)].append((t, q))
    for c, v in sorted(clips.items()):
        print(f"   clip {c}: labeled {sum(t for t,_ in v)}, predicted {sum(q for _,q in v)} (mean error {sum(abs(t-q) for t,q in v)/len(v):.2f})")
    print()
