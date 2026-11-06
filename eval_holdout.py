"""
Compare models on the hand-checked holdout clips (the honest test).

Usage (project folder, project venv):
    python eval_holdout.py parking_yolo26n.pt parking_v2_best.pt parking_v2_last.pt
    python eval_holdout.py parking_v2_best.pt --imgsz 960 --conf 0.25

Needs holdout_final/images and holdout_final/labels (made by:  python finalize_dataset.py holdout_auto holdout_final).
Prints precision, recall, mAP50, mAP50-95 per model (overall and per class) and speed.
"""
import argparse, os
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("weights", nargs="+")
p.add_argument("--data", default="holdout_final")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--conf", type=float, default=0.001, help="keep low for mAP; this is the standard for validation")
a = p.parse_args()

root = os.path.abspath(a.data)
assert os.path.isdir(f"{root}/images") and os.path.isdir(f"{root}/labels"), f"{root}/images and /labels not found"
names = [l.strip() for l in open("classes.txt") if l.strip()]
yaml_path = os.path.join(root, "holdout.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {root.replace(os.sep, '/')}\ntrain: images\nval: images\nnames:\n" +
            "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))

print(f"{'model':28} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}   per-class mAP50 (" + ", ".join(names) + ")   ms/img")
for w in a.weights:
    m = YOLO(w)
    r = m.val(data=yaml_path, imgsz=a.imgsz, conf=a.conf, verbose=False, plots=False)
    per = ", ".join(f"{x:.3f}" for x in r.box.ap50)
    print(f"{os.path.basename(w):28} {r.box.mp:6.3f} {r.box.mr:6.3f} {r.box.map50:7.3f} {r.box.map:9.3f}   {per}   {r.speed['inference']:.1f}")
