"""
Grab a single frame from a video at a given timestamp and save it as a JPG.

Usage:
    python grab_frame.py --source IMG_8836.MOV --seconds 10 --out frame_8836.jpg

Play the video (or scrub through it in VLC/Windows Photos) first to find a
timestamp where cars are CLEARLY visible, then use that timestamp here.

Note: iPhone .MOV files (HEVC) often don't support random seeking reliably
in OpenCV, so this reads frames sequentially up to the target time instead
of jumping directly to it. Slower, but accurate.
"""
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", default="frame.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    print(f"[INFO] {args.source}: {fps:.2f} fps, {total_frames} frames, ~{duration:.1f}s duration")

    if args.seconds > duration:
        raise SystemExit(
            f"[ERROR] Requested {args.seconds}s but video is only ~{duration:.1f}s long. "
            f"Pick a smaller --seconds value."
        )

    target_frame = int(args.seconds * fps)

    frame = None
    idx = 0
    while idx <= target_frame:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Could not read frame {idx} (target was {target_frame}). "
                f"Video may be shorter or corrupted past that point."
            )
        idx += 1

    cv2.imwrite(args.out, frame)
    h, w = frame.shape[:2]
    print(f"[OK] Saved {args.out} ({w}x{h}) from {args.source} at ~{args.seconds}s (frame {target_frame})")


if __name__ == "__main__":
    main()
