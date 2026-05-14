"""Live camera viewer for calibrating SO-101 camera angle / framing.

Shows the same view rerun would show during inference (camera 0, 640x480,
no rotation). Optional overlay PNG (with alpha) is alpha-blended on top so
you can align the workspace to a reference frame.

Usage:
    python camera_calibrate.py
    python camera_calibrate.py --camera-index 0 --overlay reference.png
    python camera_calibrate.py --no-crosshair
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def alpha_blend(bg: np.ndarray, fg: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
    """In-place alpha-blend `fg` (BGRA or BGR) onto `bg` at (x, y)."""
    h, w = fg.shape[:2]
    H, W = bg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return bg
    fg_clip = fg[y0 - y:y1 - y, x0 - x:x1 - x]
    if fg_clip.shape[2] == 4:
        a = fg_clip[..., 3:4].astype(np.float32) / 255.0
        bg[y0:y1, x0:x1] = (a * fg_clip[..., :3] + (1 - a) * bg[y0:y1, x0:x1]).astype(bg.dtype)
    else:
        bg[y0:y1, x0:x1] = fg_clip
    return bg


def draw_crosshair(img: np.ndarray) -> None:
    """Center crosshair + 1/3 rule-of-thirds lines for framing reference."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
    cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
    for fx in (1 / 3, 2 / 3):
        cv2.line(img, (int(w * fx), 0), (int(w * fx), h), (60, 60, 60), 1)
    for fy in (1 / 3, 2 / 3):
        cv2.line(img, (0, int(h * fy)), (w, int(h * fy)), (60, 60, 60), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--overlay", type=str, default=None,
                        help="Path to a PNG (RGBA preferred) to alpha-blend on top.")
    parser.add_argument("--overlay-x", type=int, default=0)
    parser.add_argument("--overlay-y", type=int, default=0)
    parser.add_argument("--no-crosshair", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened camera {args.camera_index}: requested {args.width}x{args.height}, "
          f"got {actual_w}x{actual_h}")

    overlay = None
    if args.overlay:
        overlay = cv2.imread(args.overlay, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise SystemExit(f"Could not read overlay: {args.overlay}")
        # cv2 reads PNG as BGRA; live frames are BGR — alpha_blend handles both.
        print(f"Loaded overlay: {Path(args.overlay).name} {overlay.shape}")

    print("Press ESC or 'q' to quit, 'c' to toggle crosshair, 's' to save snapshot.")
    show_crosshair = not args.no_crosshair
    snap_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        display = frame.copy()
        if overlay is not None:
            alpha_blend(display, overlay, args.overlay_x, args.overlay_y)
        if show_crosshair:
            draw_crosshair(display)

        cv2.imshow("camera_calibrate (ESC=quit, c=crosshair, s=snap)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("c"):
            show_crosshair = not show_crosshair
        if key == ord("s"):
            out = Path(f"snap_{snap_idx:03d}.png")
            cv2.imwrite(str(out), frame)
            print(f"Saved {out}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
