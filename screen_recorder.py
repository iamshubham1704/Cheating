import argparse
import datetime as dt
import time
from pathlib import Path

import cv2
import numpy as np
from mss import mss


FOURCC_MP4V = cv2.VideoWriter_fourcc(*"mp4v")


def list_monitors() -> None:
    with mss() as sct:
        for i, mon in enumerate(sct.monitors):
            if i == 0:
                name = "ALL (virtual)"
            else:
                name = f"Monitor {i}"
            w = mon.get("width", 0)
            h = mon.get("height", 0)
            print(f"[{i}] {name}: {w}x{h} at ({mon.get('left', 0)},{mon.get('top', 0)})")


def record_screen(output: Path, monitor_index: int = 1, fps: int = 20, show_preview: bool = True) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)

    with mss() as sct:
        monitors = sct.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            print("Invalid monitor index. Use --list to see available monitors.")
            return 2
        mon = monitors[monitor_index]
        width, height = mon["width"], mon["height"]

        writer = cv2.VideoWriter(str(output), FOURCC_MP4V, fps, (width, height))
        if not writer.isOpened():
            print("Error: cannot open VideoWriter. Check your codecs and output path.")
            return 3

        frame_interval = 1.0 / max(1, fps)
        next_time = time.time()
        print(f"Recording {width}x{height} @ {fps} FPS to {output} (press 'q' or Esc to stop)")

        try:
            while True:
                img = sct.grab(mon)
                frame = np.array(img)  # BGRA
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                writer.write(frame_bgr)

                if show_preview:
                    cv2.imshow("Screen Recorder", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break

                next_time += frame_interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.time()
        finally:
            writer.release()
            if show_preview:
                cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen recorder using MSS + OpenCV")
    parser.add_argument("--list", action="store_true", help="List monitors and exit")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (1..n), 0 = all")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument("--out", type=str, default="captures", help="Output directory or file (.mp4)")

    args = parser.parse_args()

    if args.__dict__["list"]:
        list_monitors()
        return 0

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".mp4":
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_path / f"screen_{args.monitor}_{timestamp}.mp4"

    return record_screen(out_path, monitor_index=args.monitor, fps=args.fps, show_preview=not args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
