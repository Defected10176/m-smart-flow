# M-Smart Flow & Eco-Rewards

AI-powered smart parking system for the MUICT & The Mall Hackathon 2026 — built by team **Cboog**.

Helps drivers find parking faster via CCTV-based vehicle/zone detection and rewards
faster turnover through an Exit Delay Promotion engine and Eco-Rewards (ESG / CO2
savings) gamification layer.

## How it works

1. **Detection** — a YOLO26 model (fine-tuned on a custom Roboflow parking-lot dataset)
   detects vehicles (`car` = occupied) and `free` (empty spot) from CCTV/video footage via
   Roboflow's hosted Serverless Cloud API.
2. **Zone mapping** — each parking row is a named polygon ("zone"). The script crops the
   frame around each zone, runs detection on the crop, and counts cars vs free spots per zone.
3. **Eco-Rewards / ESG layer** — reduced search time is converted into estimated CO2
   savings (idling-emissions based), feeding the gamification/rewards side of the app.

## Files

| File | Purpose |
|---|---|
| `vehicle_detection_api.py` | **Main script (cloud).** Calls our custom model on Roboflow's Serverless Cloud API. Crops each zone before detection, counts cars/free per zone, draws boxes (red = car, green = free), logs to `zone_events.jsonl`. |
| `zone_calibrator.py` | Click-to-draw tool for defining parking zones on a real video frame. Outputs a zones JSON file. |
| `zones_calibrated.json` | Zones already drawn for `IMG_8838.MOV` (2 zones). Only valid for that video/camera angle. |
| `zones_example.json` | Placeholder zones, not calibrated to any real footage. Format reference only. |
| `grab_frame.py` | Saves one frame from a video at a given second (safe for iPhone `.MOV`). Handy for testing the model with curl. |
| `vehicle_detection.py` | Local-only detection with generic pretrained YOLO (no cloud). Offline fallback. |
| `vehicle_detection_stream.py` | WebRTC streaming variant. Reference only; unreliable on networks that block STUN/TURN. |
| `train_yolo_parking.py` | Local training script. Not used in the final workflow (we trained via Roboflow web UI). |
| `requirements.txt` | Python dependencies. |

## Setup 

1. Clone the repo (or `git pull` if you already have it) and enter the folder.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```
3. **Get your own Roboflow API key** (free account): Roboflow → Settings → API Keys.
   Do not share keys or commit them to the repo. Ask the repo owner to add you to the
   Roboflow workspace so you can use the hosted model.
4. Set the key in the same terminal you will run the script in (Windows cmd; it only lasts
   for that terminal window, so repeat in every new terminal):
   ```
   set ROBOFLOW_API_KEY=your_key_here
   ```
   PowerShell: `$env: ROBOFLOW_API_KEY="your_key_here"`. Or pass `--api-key your_key_here` per run.
5. **Get the video files from the repo owner** (Drive/USB). Videos are not in git because
   they are too large. Put `IMG_8838.MOV` in the repo folder.

## Run the detection

```
python vehicle_detection_api.py --source IMG_8838.MOV --zones zones_calibrated.json --conf 0.05 --start-seconds 8
```

A window opens with the video. Zone outlines show `N cars / M free (X%)`. Boxes are red for
cars and green for free spots. Press `q` to quit. Each processed frame also prints a line:

```
[detection 12] 5 cars, 2 free spots total (in zones: 4 cars, 1 free)
```

- "total" = everything the model found anywhere in the frame.
- "in zones" = only detections whose center falls inside a drawn zone.

Detection runs in a background thread, so video playback stays smooth. Each update makes
one API call per zone, so new results arrive every few seconds.

Flags:
- `--conf` confidence threshold (0-1). Use low values (~0.05) on real footage.
- `--start-seconds` skip ahead (use to skip ceiling/intro footage).
- `--frame-skip` how often a frame is sampled for detection (default 5).
- `--zones` path to a zones JSON file.
- `--no-display` run without a window.

## Calibrate zones for a new video or camera angle

Zones are pixel coordinates, so they must be redrawn for each video/camera angle.

```
python zone_calibrator.py --source YOUR_VIDEO.MOV --seconds 8 --out my_zones.json
```

1. A frame from the video opens. Pick `--seconds` where the cars are clearly visible.
2. **One zone = one row of cars.** Click 4 points tracing the outline of that row, around
   where the cars sit. Do not cross over to other rows or the aisle.
3. Press `n`, then type a zone name (e.g. `zone_1`) and the number of parking spots in that row.
4. Repeat for each row.
5. Press `s` to save and exit.

Other keys: `u` undo last point, `c` clear current zone, `q` / ESC quit without saving.

Then run detection with `--zones my_zones.json`.

## Model

- Base: YOLO26n, fine-tuned on a custom Roboflow parking-lot dataset (forked from
  `parking-lot-jqsj2`, extended with our own labels).
- Best run so far: ~96.8% mAP@50 on the validation set.
- Hosted at: `pisitjirawoottiwat101-gmail-com/parking-lot-jqsj2-nujb1-1-yolo26n-t2`
  via Roboflow Serverless Cloud API.
- Classes: `car` (occupied) and `free` (empty spot).

**Known limitation:** the model performs best on footage similar to its training images.
On our wide-angle mall-camera footage, confidence is low and detections are inconsistent
when the whole frame is sent. Cropping tightly around each parking row (done automatically
per zone) improves both detection count and confidence noticeably. Adding our own labeled
frames to the training set and retraining would close the gap further.

## Troubleshooting

- **"No API key found"**: the key is not set in this terminal. Run the `set` command again.
- **Few or no detections**: lower `--conf`, check `--start-seconds` lands on cars (not
  ceiling), and make sure the zones match the video you are running.
- **Zone count is 0 but "total" is not**: zones do not line up with the cars. Re-run the calibrator.
- **Video will not open**: check the file name and that it is in the folder you run from.
- **VS Code reverts edits after a push**: turn off File → Auto Save, then use Revert File.

## Not included in this repo

Video footage (`.mov`/`.mp4`), model weights (`.pt`), and generated logs (`.jsonl`) are
gitignored: too large for git and easy to regenerate. Share them separately (Drive,
Roboflow project link). Never commit API keys.
