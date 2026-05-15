# RoboLang: Natural Language Guided Robotic Manipulation

## Project Overview

RoboLang is a modular Vision-Language-Action (VLA) framework for robotic manipulation. It enables a simulated robot arm (UR5/Kuka iiwa) to execute object manipulation tasks based on natural language commands.

### Example Commands
- "Move the blue block to the right of the green cube."
- "Pick up the red sphere and place it on the yellow platform."
- "Stack the orange cube on top of the purple block."

## Architecture

The pipeline connects: **CommandParser** → **ObjectDetector** → **SceneGraph** → **ActionGenerator** → **MotionPlanner** → **SimulationEnv**

## Tech Stack

- **Language**: Python 3.12
- **Deep Learning**: PyTorch 2.2 (CPU), Hugging Face Transformers, DINOv2, timm
- **Computer Vision**: OpenCV, Pillow, scikit-image, albumentations
- **NLP**: NLTK, spaCy, sentence-transformers
- **Robotics/Simulation**: PyBullet (physics engine), IKPy (Inverse Kinematics)
- **Utilities**: NumPy, SciPy, Pandas, Hydra, WandB, TensorBoard

## Package Structure

All source files live in the project root **and** are mirrored into their proper subpackages:

```
language/        command_parser.py, intent_classifier.py
vision/          object_detector.py, scene_graph.py
action/          action_generator.py, motion_planner.py
simulation/      pybullet_env.py, simulation.py
evaluation/      metrics.py, evaluate.py
data/            dataset.py, dataset_loader.py, augmentation.py
models/          vla_model.py, model.py
vla/             language.py, perception.py, policy.py, simulation.py, dataset.py, evaluation.py, model.py
```

## Running

```bash
# Run all 8 demo scenarios (mock mode, no PyBullet required)
python demo.py

# Run a single command
python demo.py --command "Move the blue block to the right of the green cube"

# Run verbose output
python demo.py --verbose

# Run the full pipeline on a command
python pipeline.py --command "Move the blue block to the right of the green cube"

# Run tests
pytest test_parser.py test_detector.py test_pipeline.py
```

## Workflow

The configured workflow runs `python demo.py` — it executes 8 natural language manipulation scenarios and prints a KPI summary.

## User Preferences

- Project runs in **mock mode** by default (no PyBullet physics simulator required)
- PyBullet is optional — all simulation files have graceful fallbacks
