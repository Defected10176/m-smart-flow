"""
Turn the reviewed X-AnyLabeling files into a clean YOLO dataset folder, with sanity checks.

Usage:  python finalize_dataset.py [source_folder] [output_folder]
        default: labeling_auto -> final_dataset.   For the test set: python finalize_dataset.py holdout_auto holdout_final
Checks / fixes:
  - drops 'free' boxes that overlap a car box (IoU > 0.2 or mostly inside a car)
  - drops tiny boxes (< 8 px) and exact/near-duplicate boxes of the same class
  - reports images with no boxes at all
Writes final_dataset/images/*.jpg and final_dataset/labels/*.txt (class order from classes.txt).
"""
import glob, json, os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "labeling_auto"
DST = sys.argv[2] if len(sys.argv) > 2 else "final_dataset"
names = [l.strip() for l in open("classes.txt") if l.strip()]
cid = {n: i for i, n in enumerate(names)}

def box(s):
    xs = [p[0] for p in s["points"]]; ys = [p[1] for p in s["points"]]
    return min(xs), min(ys), max(xs), max(ys)

def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)

def area(a): return max(0, a[2]-a[0]) * max(0, a[3]-a[1])
def iou(a, b):
    i = inter(a, b); u = area(a) + area(b) - i
    return i / u if u else 0

os.makedirs(f"{DST}/images", exist_ok=True); os.makedirs(f"{DST}/labels", exist_ok=True)
stats = dict(images=0, cars=0, free=0, free_dropped=0, tiny=0, dup=0, empty=[], unknown=set())
for jf in sorted(glob.glob(f"{SRC}/*.json")):
    base = os.path.basename(jf)[:-5]
    d = json.load(open(jf)); W, H = d["imageWidth"], d["imageHeight"]
    items = []
    for s in d["shapes"]:
        if s["label"] not in cid:
            stats["unknown"].add(s["label"]); continue
        b = box(s)
        if b[2]-b[0] < 8 or b[3]-b[1] < 8:
            stats["tiny"] += 1; continue
        items.append((s["label"], b))
    cars = [b for l, b in items if l == "car"]
    keep = []
    for l, b in items:
        if l == "free" and any(iou(b, c) > 0.2 or inter(b, c) > 0.5 * area(b) for c in cars):
            stats["free_dropped"] += 1; continue
        if any(l == l2 and iou(b, b2) > 0.9 for l2, b2 in keep):
            stats["dup"] += 1; continue
        keep.append((l, b))
    lines = []
    for l, (x1, y1, x2, y2) in keep:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        lines.append(f"{cid[l]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        stats["cars" if l == "car" else "free"] += 1
    if not lines: stats["empty"].append(base)
    shutil.copy2(f"{SRC}/{base}.jpg", f"{DST}/images/{base}.jpg")
    open(f"{DST}/labels/{base}.txt", "w").write("\n".join(lines))
    stats["images"] += 1
print(f"images: {stats['images']} | car boxes: {stats['cars']} | free boxes: {stats['free']}")
print(f"removed: {stats['free_dropped']} free boxes overlapping a car, {stats['tiny']} tiny boxes, {stats['dup']} duplicate boxes")
print(f"images with NO boxes: {len(stats['empty'])}")
if stats["unknown"]: print("unknown labels skipped:", stats["unknown"])
