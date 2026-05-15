"""
inference.py
─────────────
Main inference entry point.

Run the Universal VLA pipeline on:
  • A video file or webcam stream
  • A list of image frames
  • A simulation scene (mock mode)

Usage
─────
  # Interactive demo (mock frames, any robot):
  python inference.py --robot kuka_iiwa7 --command "Pick up the red cube."

  # From a video file:
  python inference.py --robot ur5 --video my_scene.mp4 \
                      --command "Move the blue block to the right."

  # From a webcam:
  python inference.py --robot franka_panda --webcam \
                      --command "Stack the orange cube on the green platform."

  # Dexterous hand:
  python inference.py --robot shadow_hand \
                      --command "Grasp the small sphere with a pinch grip."

  # List available robots:
  python inference.py --list-robots

  # Adapt to a new environment (provide calibration images):
  python inference.py --robot ur5 --adapt --calib-dir ./calib_images/ \
                      --command "Push the cylinder to the left."
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _load_frames_from_dir(path: str, limit: int = 32) -> List[np.ndarray]:
    """Load all images from a directory, sorted by name."""
    try:
        import cv2
    except ImportError:
        return []
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(sorted(glob.glob(str(Path(path) / ext))))
    frames = []
    for f in files[:limit]:
        img = cv2.imread(f)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def _load_frames_from_video(path: str, n_frames: int = 16) -> List[np.ndarray]:
    try:
        import cv2
    except ImportError:
        return []
    cap    = cv2.VideoCapture(path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
    step   = max(1, total // n_frames)
    frames = []
    idx    = 0
    while cap.isOpened() and len(frames) < n_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def _load_frames_from_webcam(n_frames: int = 16, fps: int = 10) -> List[np.ndarray]:
    try:
        import cv2
    except ImportError:
        return []
    cap    = cv2.VideoCapture(0)
    frames = []
    delay  = 1.0 / fps
    print(f"Capturing {n_frames} frames from webcam …")
    while len(frames) < n_frames:
        ret, bgr = cap.read()
        if ret:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        time.sleep(delay)
    cap.release()
    return frames


def _make_mock_frames(n: int = 16, h: int = 224, w: int = 224) -> List[np.ndarray]:
    """Generate synthetic frames for testing (no camera needed)."""
    frames = []
    # Add some colour blobs to simulate objects
    colours = [
        (200, 50,  50),   # red blob
        (50,  100, 200),  # blue blob
        (50,  180, 50),   # green blob
    ]
    for i in range(n):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 180   # grey background
        # Draw moving blobs
        for j, (r, g, b) in enumerate(colours):
            cx = int(w * 0.25 + j * w * 0.25 + i * 0.5)
            cy = int(h * 0.5)
            cx = max(20, min(w - 20, cx))
            y1, y2 = max(0, cy - 20), min(h, cy + 20)
            x1, x2 = max(0, cx - 20), min(w, cx + 20)
            frame[y1:y2, x1:x2] = (r, g, b)
        frames.append(frame)
    return frames


# ─────────────────────────────────────────────────────────
# Demo scene definitions
# ─────────────────────────────────────────────────────────

DEMO_SCENES = {
    "tabletop": [
        {"colour": "blue",   "shape": "block",    "position": (-0.10, 0.00, 0.65)},
        {"colour": "green",  "shape": "cube",     "position": ( 0.00, 0.00, 0.65)},
        {"colour": "red",    "shape": "sphere",   "position": ( 0.10, 0.00, 0.65)},
        {"colour": "yellow", "shape": "platform", "position": ( 0.00,-0.10, 0.65)},
        {"colour": "cyan",   "shape": "cylinder", "position": ( 0.15, 0.10, 0.65)},
        {"colour": "purple", "shape": "block",    "position": (-0.15, 0.10, 0.65)},
    ],
    "hand": [
        {"colour": "red",    "shape": "sphere",   "position": ( 0.05, 0.00, 0.65)},
        {"colour": "blue",   "shape": "cube",     "position": (-0.05, 0.00, 0.65)},
    ],
}

DEMO_COMMANDS = {
    "kuka_iiwa7":   "Move the blue block to the right of the green cube.",
    "ur5":          "Pick up the red sphere and place it on the yellow platform.",
    "franka_panda": "Stack the cyan cylinder on top of the purple block.",
    "shadow_hand":  "Grasp the red sphere using a pinch grip.",
    "simple_2dof":  "Push the blue block to the left.",
    "default":      "Move the blue block to the right of the green cube.",
}


# ─────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────

def run_inference(
    robot:       str  = "kuka_iiwa7",
    command:     str  = "",
    video:       Optional[str] = None,
    webcam:      bool = False,
    calib_dir:   Optional[str] = None,
    adapt:       bool = False,
    model_path:  Optional[str] = None,
    n_frames:    int  = 16,
    verbose:     bool = False,
    mock:        bool = True,
    scene_key:   str  = "tabletop",
):
    from universal_pipeline import UniversalPipeline

    print(f"\n{'='*62}")
    print(f"  RoboLang Universal Pipeline")
    print(f"  Robot   : {robot}")
    print(f"  Command : {command or DEMO_COMMANDS.get(robot, DEMO_COMMANDS['default'])}")
    print(f"{'='*62}\n")

    cmd = command or DEMO_COMMANDS.get(robot, DEMO_COMMANDS["default"])

    # ── Build pipeline ──────────────────────────────────────
    pipe = UniversalPipeline.for_robot(
        robot,
        model_path = model_path,
        adapt      = adapt,
    )
    print(pipe)

    # ── Load frames ─────────────────────────────────────────
    if webcam:
        frames = _load_frames_from_webcam(n_frames)
    elif video:
        frames = _load_frames_from_video(video, n_frames)
    else:
        frames = _make_mock_frames(n_frames)

    print(f"  Frames loaded : {len(frames)}")

    # ── Adapt to environment ─────────────────────────────────
    if adapt:
        calib_frames = frames
        if calib_dir:
            calib_frames = _load_frames_from_dir(calib_dir) or frames
        print(f"  Adapting to environment ({len(calib_frames)} calib frames) …")
        pipe.adapt_to_environment(calib_frames, verbose=verbose)

    # ── Select scene ────────────────────────────────────────
    scene = DEMO_SCENES.get(
        "hand" if "hand" in robot else scene_key,
        DEMO_SCENES["tabletop"]
    )

    # ── Run pipeline ─────────────────────────────────────────
    result = pipe.run(frames=frames, command=cmd, scene_info=scene)

    # ── Print results ────────────────────────────────────────
    print(result.summary())

    if verbose and result.motor_commands:
        print("\nMotor Commands:")
        for i, mc in enumerate(result.motor_commands):
            print(f"  [{i+1}] {mc}")

    if verbose and result.tracks:
        print("\nTracked Objects:")
        for t in result.tracks:
            print(f"  {t}")

    if verbose and result.trajectories:
        print("\nPredicted Trajectories:")
        for tp in result.trajectories:
            print(f"  Track#{tp.track_id} [{tp.colour}] "
                  f"→ final pos {tuple(round(v,3) for v in tp.final_position)} "
                  f"(horizon={tp.horizon_s:.1f}s)")

    return result


# ─────────────────────────────────────────────────────────
# Multi-robot demo
# ─────────────────────────────────────────────────────────

def run_multi_robot_demo():
    """
    Demonstrate the system across multiple robot morphologies
    with the SAME model weights.
    """
    robots = [
        ("simple_2dof",  "Move the blue block to the left."),
        ("kuka_iiwa7",   "Move the blue block to the right of the green cube."),
        ("ur5",          "Pick up the red sphere and place it on the yellow platform."),
        ("franka_panda", "Stack the cyan cylinder on top of the purple block."),
        ("shadow_hand",  "Grasp the red sphere."),
    ]

    print("\n" + "═"*62)
    print("  UNIVERSAL ROBOT DEMO — One Model, All Robots")
    print("═"*62)

    from universal_pipeline import UniversalPipeline

    for robot_name, cmd in robots:
        print(f"\n{'─'*50}")
        print(f"  Robot: {robot_name.upper()}")
        try:
            pipe   = UniversalPipeline.for_robot(robot_name, adapt=False)
            frames = _make_mock_frames(8)
            scene  = DEMO_SCENES.get(
                "hand" if "hand" in robot_name else "tabletop",
                DEMO_SCENES["tabletop"]
            )
            result = pipe.run(frames=frames, command=cmd, scene_info=scene)
            print(result.summary())
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "═"*62)
    print("  Demo complete.")
    print("═"*62)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RoboLang Universal VLA Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\nUsage")[1] if "\n\nUsage" in __doc__ else "",
    )
    ap.add_argument("--robot",       default="kuka_iiwa7",
                    help="Robot name or path to YAML config")
    ap.add_argument("--command",     default="",
                    help="Natural-language instruction")
    ap.add_argument("--video",       default=None,
                    help="Path to input video file")
    ap.add_argument("--webcam",      action="store_true",
                    help="Capture from webcam")
    ap.add_argument("--calib-dir",   default=None,
                    help="Directory of calibration images for adaptation")
    ap.add_argument("--adapt",       action="store_true",
                    help="Enable sim2real adaptation")
    ap.add_argument("--model-path",  default=None,
                    help="Path to VLA model checkpoint")
    ap.add_argument("--n-frames",    type=int, default=16,
                    help="Number of video frames to use")
    ap.add_argument("--verbose",     action="store_true")
    ap.add_argument("--demo",        action="store_true",
                    help="Run multi-robot demonstration")
    ap.add_argument("--list-robots", action="store_true",
                    help="List available robot presets")
    return ap


def main():
    ap   = _build_parser()
    args = ap.parse_args()

    if args.list_robots:
        from robot.robot_config import list_presets, get_robot
        print("\nAvailable robot presets:")
        for name in list_presets():
            r = get_robot(name)
            print(f"  {name:<20} {r.dof:>2} DOF  {r.description}")
        return

    if args.demo:
        run_multi_robot_demo()
        return

    run_inference(
        robot      = args.robot,
        command    = args.command,
        video      = args.video,
        webcam     = args.webcam,
        calib_dir  = args.calib_dir,
        adapt      = args.adapt,
        model_path = args.model_path,
        n_frames   = args.n_frames,
        verbose    = args.verbose,
    )


if __name__ == "__main__":
    main()
