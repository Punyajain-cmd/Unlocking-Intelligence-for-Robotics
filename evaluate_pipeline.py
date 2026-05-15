from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from vla.evaluation import DEFAULT_COMMANDS, episode_results_to_dict, evaluate_pipeline
from vla.language import CommandParser
from vla.perception import ColorObjectPerceiver
from vla.policy import RelationActionPlanner, TaskExecutor
from vla.simulation import ManipulationSimEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the command-to-action VLA simulation pipeline.")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI.")
    parser.add_argument("--commands-file", type=str, default="")
    parser.add_argument("--output-json", type=str, default="artifacts/eval_metrics.json")
    return parser.parse_args()


def load_commands(commands_file: str) -> list[str]:
    if not commands_file:
        return DEFAULT_COMMANDS
    command_path = Path(commands_file)
    commands = [line.strip() for line in command_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return commands if commands else DEFAULT_COMMANDS


def main() -> None:
    args = parse_args()
    commands = load_commands(args.commands_file)

    env = ManipulationSimEnv(gui=args.gui, seed=123)
    parser = CommandParser()
    bounds = env.get_workspace_bounds()
    perceiver = ColorObjectPerceiver(workspace_x=bounds.x, workspace_y=bounds.y)
    executor = TaskExecutor(planner=RelationActionPlanner())

    try:
        metrics, episode_results = evaluate_pipeline(commands, env, parser, perceiver, executor)
    finally:
        env.close()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "episodes": episode_results_to_dict(episode_results),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
