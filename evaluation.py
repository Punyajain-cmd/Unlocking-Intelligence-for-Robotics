from __future__ import annotations

from dataclasses import asdict, dataclass

from .language import CommandParser
from .perception import ColorObjectPerceiver
from .policy import TaskExecutor
from .simulation import ManipulationSimEnv


DEFAULT_COMMANDS = [
    "Move the blue block to the right of the green cube.",
    "Move the red block to the left of the yellow cube.",
    "Move the yellow block in front of the blue cube.",
    "Move the green block behind the red cube.",
    "Move the red block next to the green cube.",
    "Move the blue block on top of the yellow cube.",
]


@dataclass(frozen=True)
class EpisodeResult:
    command: str
    parsed: bool
    success: bool
    message: str


def evaluate_pipeline(
    commands: list[str],
    env: ManipulationSimEnv,
    parser: CommandParser,
    perceiver: ColorObjectPerceiver,
    executor: TaskExecutor,
) -> tuple[dict[str, float], list[EpisodeResult]]:
    episode_results: list[EpisodeResult] = []
    parse_successes = 0
    task_successes = 0

    for idx, command in enumerate(commands):
        env.reset(seed=idx + 1)
        rgb = env.render_topdown()
        detections = perceiver.detect(rgb)

        try:
            intent = parser.parse(command)
            parsed = True
            parse_successes += 1
        except Exception as exc:
            parsed = False
            episode_results.append(
                EpisodeResult(command=command, parsed=False, success=False, message=f"Parse error: {exc}")
            )
            continue

        if intent.source_color not in detections or intent.target_color not in detections:
            episode_results.append(
                EpisodeResult(
                    command=command,
                    parsed=True,
                    success=False,
                    message="Perception missing required objects.",
                )
            )
            continue

        success, info = executor.execute(intent, detections, env)
        if success:
            task_successes += 1
        episode_results.append(
            EpisodeResult(
                command=command,
                parsed=True,
                success=success,
                message=f"Target XY={info.target_xy}",
            )
        )

    total = max(len(commands), 1)
    command_interpretation_accuracy = parse_successes / total
    task_success_rate = task_successes / total

    metrics = {
        "command_interpretation_accuracy": command_interpretation_accuracy,
        "task_success_rate": task_success_rate,
        "goal_condition_accuracy": task_success_rate,
        "task_completion_rate": task_success_rate,
    }
    return metrics, episode_results


def episode_results_to_dict(results: list[EpisodeResult]) -> list[dict[str, str | bool]]:
    return [asdict(result) for result in results]
