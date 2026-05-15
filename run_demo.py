from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from vla.evaluation import DEFAULT_COMMANDS
from vla.language import CommandParser
from vla.perception import ColorObjectPerceiver
from vla.policy import RelationActionPlanner, TaskExecutor
from vla.simulation import ManipulationSimEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a VLA simulation demo with natural-language commands.")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output-video", type=str, default="artifacts/demo.mp4")
    parser.add_argument("--commands-file", type=str, default="")
    return parser.parse_args()


def load_commands(commands_file: str) -> list[str]:
    if not commands_file:
        return DEFAULT_COMMANDS
    path = Path(commands_file)
    commands = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return commands if commands else DEFAULT_COMMANDS


def main() -> None:
    args = parse_args()
    commands = load_commands(args.commands_file)

    env = ManipulationSimEnv(gui=args.gui, seed=42)
    parser = CommandParser()
    bounds = env.get_workspace_bounds()
    perceiver = ColorObjectPerceiver(workspace_x=bounds.x, workspace_y=bounds.y)
    executor = TaskExecutor(planner=RelationActionPlanner())

    frames = []
    successes = 0
    try:
        for idx, command in enumerate(commands):
            env.reset(seed=idx + 10)
            before = env.render_topdown(width=args.width, height=args.height)
            frames.append(before)

            try:
                intent = parser.parse(command)
            except ValueError as exc:
                print(f"[FAIL] {command} | parse error: {exc}")
                continue

            detections = perceiver.detect(before)
            success, _ = executor.execute(intent, detections, env)

            for _ in range(15):
                env.step()
                frames.append(env.render_topdown(width=args.width, height=args.height))

            status = "SUCCESS" if success else "FAIL"
            print(f"[{status}] {command}")
            if success:
                successes += 1

        output_video = Path(args.output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_video, frames, fps=args.fps)

        success_rate = successes / max(len(commands), 1)
        print(f"Completed {len(commands)} commands | success_rate={success_rate:.3f}")
        print(f"Saved demo video to: {output_video}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
