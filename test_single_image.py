"""
Quick diagnostic: run your deployed workflow on cars_frame.jpg directly,
bypassing the Roboflow website's Test page entirely.

Usage:
    set ROBOFLOW_API_KEY=your_key_here
    python test_single_image.py
"""

import base64
import json
import os

from inference_sdk import InferenceHTTPClient, InferenceConfiguration

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set ROBOFLOW_API_KEY first: set ROBOFLOW_API_KEY=your_key_here")

image_path = "cars_frame.jpg"
if not os.path.exists(image_path):
    raise SystemExit(f"File not found: {os.path.abspath(image_path)}")

file_size = os.path.getsize(image_path)
print(f"Found {image_path} ({file_size} bytes)")

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
print(f"Encoded to base64 ({len(image_b64)} chars)")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key,
).configure(InferenceConfiguration(api_key_transport="header"))

print("Running workflow on cars_frame.jpg (base64 string method) ...")
try:
    result = client.run_workflow(
        workspace_name="pisitjirawoottiwat101-gmail-com",
        workflow_id="parking-lot-vparking-lot-jqsj2-nujb1-1-yolo26n-t2-logic",
        images={"image": image_b64},
        use_cache=False,
    )
    print("\n=== RESULT (base64 string) ===")
    print(json.dumps(result, indent=2, default=str))
    print("=== END ===\n")
except Exception as e:
    print(f"\n[base64 string method failed]: {e}\n")

print("Running workflow on cars_frame.jpg (numpy array method) ...")
try:
    import cv2
    img_array = cv2.imread(image_path)
    print(f"Loaded array shape: {img_array.shape if img_array is not None else 'FAILED TO LOAD'}")
    result2 = client.run_workflow(
        workspace_name="pisitjirawoottiwat101-gmail-com",
        workflow_id="parking-lot-vparking-lot-jqsj2-nujb1-1-yolo26n-t2-logic",
        images={"image": img_array},
        use_cache=False,
    )
    print("\n=== RESULT (numpy array) ===")
    print(json.dumps(result2, indent=2, default=str))
    print("=== END ===\n")
except Exception as e:
    print(f"\n[numpy array method failed]: {e}\n")
