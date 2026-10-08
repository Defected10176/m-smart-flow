"""
Export a few still frames from every video in this folder.

Usage:
    python export_frames.py                 # all *.MOV / *.mp4 in the current folder
    python export_frames.py --times 2 8 15  # choose the timestamps (seconds)

Output: frame_<video name>_t<seconds>.jpg next to the videos
(these match the 'frame_*.jpg' rule in .gitignore, so they won't be committed).

Uses sequential reading (not seeking), which is reliable for iPhone HEVC .MOV files.
"""
import argparse
import glob
import os

import cv2


def export(video_path, times):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] could not open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {video_path}: {w}x{h}, {fps:.1f} fps, ~{duration:.1f}s")

    wanted = {int(t * fps): t for t in times if t <= duration}
    if not wanted:
        print(f"[SKIP] {video_path} is shorter than the requested times")
        cap.release()
        return

    last = max(wanted)
    name = os.path.splitext(os.path.basename(video_path))[0]
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            out = f"frame_{name}_t{int(wanted[idx]):02d}.jpg"
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[OK] saved {out}")
        idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", type=float, nargs="+", default=[2, 8, 15],
                        help="Timestamps in seconds to export (default: 2 8 15)")
    parser.add_argument("--folder", default=".")
    args = parser.parse_args()

    videos = sorted(
        glob.glob(os.path.join(args.folder, "*.MOV"))
        + glob.glob(os.path.join(args.folder, "*.mov"))
        + glob.glob(os.path.join(args.folder, "*.mp4"))
        + glob.glob(os.path.join(args.folder, "*.MP4"))
    )
    if not videos:
        raise SystemExit("No videos found in this folder.")
    for v in videos:
        export(v, args.times)
    print("Done.")


if __name__ == "__main__":
    main()
