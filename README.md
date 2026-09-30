# M-Smart Flow & Eco-Rewards

AI-powered smart parking system for the MUICT & The Mall Hackathon 2026 — built by team **Cboog**.

Helps drivers find parking faster via CCTV-based vehicle/zone detection, and rewards
faster turnover through an Exit Delay Promotion engine and Eco-Rewards (ESG / CO2
savings) gamification layer.

## How it works

1. **Detection** — a YOLO26 model (fine-tuned on a custom Roboflow parking-lot dataset)
   detects vehicles and free spaces from CCTV/video footage, either locally or via
   Roboflow's hosted Serverless Cloud API.
2. **Zone mapping** — detections are mapped into named parking zones (polygons) to
   compute per-zone occupancy/density.
3. **Eco-Rewards / ESG layer** — reduced search time is converted into estimated CO2
   savings (idling-emissions based), feeding the gamification/rewards side of the app.

## Files

| File | Purpose |
|---|---|
| `vehicle_detection.py` | Local-only detection using generic pretrained YOLO26 (COCO weights) via `ultralytics`. Detects car/motorcycle/bus/truck classes directly, no cloud dependency. Good fallback / offline demo. |
| `vehicle_detection_api.py` | **Primary cloud deployment.** Calls our custom-trained model on Roboflow's Serverless Cloud API (`client.infer()`), with a background thread so video playback isn't blocked by network round-trips. This is what satisfies the hackathon's cloud-deployment requirement. |
| `vehicle_detection_stream.py` | WebRTC streaming variant (`client.webrtc.stream()`). Kept for reference — unreliable on networks that block STUN/TURN traffic; prefer `vehicle_detection_api.py`. |
| `grab_frame.py` | Utility — grabs a single frame from a video at a given timestamp (sequential read, safe for HEVC/iPhone `.MOV` files where seeking is unreliable). Useful for quick model testing via curl without running the full pipeline. |
| `train_yolo_parking.py` | Downloads the Roboflow dataset and fine-tunes YOLO26 locally. Not used in the final workflow (training was done via Roboflow's web UI instead) — kept for reference. |
| `zones_example.json` | Placeholder zone polygon coordinates. **Not calibrated** to real camera footage yet — replace with coordinates matched to your actual CCTV angle before demoing zone occupancy. |
| `requirements.txt` | Python dependencies. |

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Set your Roboflow API key (get it from Roboflow → Settings → API Keys):

```
set ROBOFLOW_API_KEY=your_key_here      # Windows cmd
```

## Usage

**Cloud API detection (primary demo path):**
```
python vehicle_detection_api.py --source your_video.mov --zones zones_example.json --conf 0.05
```

**Local-only detection (no cloud, fallback):**
```
python vehicle_detection.py --source your_video.mov --zones zones_example.json
```

Useful flags on `vehicle_detection_api.py`:
- `--conf` — confidence threshold (0–1). The current model needs a low value (~0.05) on real footage.
- `--start-seconds` — skip past dead/intro footage before detection starts.
- `--frame-skip` — how often (in frames) to sample for detection vs. just displaying video.

## Model

- Base: YOLO26n, fine-tuned on a custom Roboflow parking-lot dataset (forked from
  `parking-lot-jqsj2`, extended with our own labels).
- Best run so far: ~96.8% mAP@50 on the validation set.
- Hosted at: `pisitjirawoottiwat101-gmail-com/parking-lot-jqsj2-nujb1-1-yolo26n-t2`
  via Roboflow Serverless Cloud API.
- Classifies `car` vs `free` (empty spot) directly.

**Known limitation:** the model generalizes well to footage similar to the training
images, but shows a domain gap on some of our own mall-camera footage (different
angle/distance/lighting/fisheye distortion than the Roboflow training set) — some
videos return few/no detections even at very low confidence thresholds. Verified via
direct API testing across multiple frames/videos. Workarounds being explored: cropping
frames closer to vehicle clusters before sending to the model, and/or adding real
camera footage to the training set for a future retrain.

## Not included in this repo

Video footage (`.mov`/`.mp4`), model weights (`.pt`), and generated logs (`.jsonl`) are
gitignored — too large for git and easy to regenerate/re-download. Share these
separately (Drive, Roboflow project link) if teammates need them.
