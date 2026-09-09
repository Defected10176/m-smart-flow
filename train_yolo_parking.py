"""
M-Smart Flow — Fine-tune YOLO26 on a custom Roboflow parking-lot dataset
==========================================================================
Downloads a Roboflow dataset in YOLO format and fine-tunes YOLO26 on it,
so the model learns your dataset's actual classes (e.g. car/empty/occupied)
instead of relying only on generic COCO vehicle classes.

Install:
    pip install roboflow ultralytics

You'll need, from your Roboflow project page:
    - workspace slug   e.g. "workspace-wswxh"
    - project slug     e.g. "parking-lot-jqsj2"
    - version number   (check the "Versions" tab — export a version first if none exists)
    - API key          from Roboflow > Account Settings > API Keys
      (keep this private — don't commit it or paste it into shared files)

Usage:
    python train_yolo_parking.py \
        --workspace workspace-wswxh \
        --project parking-lot-jqsj2 \
        --version 1 \
        --api-key YOUR_KEY
"""

import argparse
import os
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO


def download_dataset(api_key, workspace, project, version, out_dir="datasets"):
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    # YOLO26 reads the same label format as YOLOv8/v11 exports
    dataset = proj.version(version).download("yolov8", location=out_dir)
    return Path(dataset.location) / "data.yaml"


def train(data_yaml, base_model="yolo26n.pt", epochs=100, imgsz=640, batch=16):
    model = YOLO(base_model)  # starts from COCO-pretrained weights, then fine-tunes
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,           # early stop if validation stops improving
        project="msmartflow_runs",
        name="parking_yolo26",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 on a Roboflow parking-lot dataset")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", type=int, required=True, help="Dataset version number")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Your Roboflow API key. If omitted, reads from the ROBOFLOW_API_KEY environment variable.",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="Base YOLO26 weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Either pass --api-key, or set it first with:\n"
            '  export ROBOFLOW_API_KEY="your_key_here"   (Mac/Linux)\n'
            "  set ROBOFLOW_API_KEY=your_key_here        (Windows)"
        )

    data_yaml = download_dataset(api_key, args.workspace, args.project, args.version)
    print(f"Dataset ready: {data_yaml}")

    train(data_yaml, base_model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    print("Training complete. Best weights saved to msmartflow_runs/parking_yolo26/weights/best.pt")


if __name__ == "__main__":
    main()
