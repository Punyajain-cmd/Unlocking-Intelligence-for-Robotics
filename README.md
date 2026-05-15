# RoboLang: Natural Language Guided Robotic Manipulation

A modular Vision-Language-Action (VLA) pipeline for robotic object manipulation driven by natural language commands.

---

## Overview

RoboLang integrates:
- **Language Understanding** (BERT-based command parser + intent/slot extraction)  
- **Visual Perception** (object detection with color/shape/position attributes)  
- **Action Generation** (symbolic + learned motion planning)  
- **Simulation** (PyBullet physics environment with UR5 robot arm)  
- **Evaluation** (Task Success Rate, Goal Condition Accuracy, Command Interpretation Accuracy)

### Example Commands
```
"Move the blue block to the right of the green cube."
"Pick up the red sphere and place it on the yellow platform."
"Push the cyan cylinder to the left side of the table."
"Stack the orange cube on top of the purple block."
"Grasp the small blue object near the edge."
```

---

## Architecture

```
NL Command
    │
    ▼
┌─────────────────────┐
│  CommandParser       │  BERT-based NER + dependency parsing
│  (language/)        │  → {action, object, target, relation}
└────────┬────────────┘
         │  Structured Intent
         ▼
┌─────────────────────┐
│  ObjectDetector      │  Color + shape segmentation + spatial indexing
│  (vision/)          │  → SceneGraph with 3D bounding boxes
└────────┬────────────┘
         │  Scene State
         ▼
┌─────────────────────┐
│  ActionGenerator     │  Grounding → IK-based trajectory generation
│  (action/)          │  → Action sequence: [grasp, move, release]
└────────┬────────────┘
         │  Trajectory
         ▼
┌─────────────────────┐
│  PyBulletEnv         │  Physics simulation with UR5 + gripper
│  (simulation/)      │  → Execution + feedback
└─────────────────────┘
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/robotic_nlp_manipulation.git
cd robotic_nlp_manipulation

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## Quick Start

### Run Demo (5+ Commands)
```bash
python demo.py
```

### Run Full Pipeline on a Single Command
```bash
python pipeline.py --command "Move the blue block to the right of the green cube" --render
```

### Train the VLA Model
```bash
python train.py --config configs/train_config.yaml --dataset open_x_embodiment
```

### Evaluate on Benchmarks
```bash
python evaluation/evaluate.py --split test --num_episodes 100
```

---

## Project Structure

```
robotic_nlp_manipulation/
├── config.py                  # Global configuration & hyperparameters
├── pipeline.py                # End-to-end pipeline orchestrator
├── demo.py                    # 5+ command demonstration script
├── train.py                   # VLA model training loop
│
├── language/
│   ├── command_parser.py      # BERT NER + intent/slot extraction
│   └── intent_classifier.py  # Action/relation classification head
│
├── vision/
│   ├── object_detector.py     # Color-based + DNN object detection
│   └── scene_graph.py         # 3D scene graph construction
│
├── action/
│   ├── action_generator.py    # Grounding → trajectory generation
│   └── motion_planner.py      # IK solver + collision avoidance
│
├── simulation/
│   ├── pybullet_env.py        # PyBullet simulation environment
│   └── robot_controller.py   # UR5 joint control & gripper
│
├── models/
│   ├── vla_model.py           # Full VLA neural network (OpenVLA-style)
│   └── encoders.py            # Visual + language encoders
│
├── evaluation/
│   ├── metrics.py             # TSR, GCA, CIA, TCR metrics
│   └── evaluate.py            # Evaluation harness
│
├── data/
│   ├── dataset_loader.py      # Open X-Embodiment / ALFRED loader
│   └── augmentation.py        # Data augmentation strategies
│
└── tests/
    ├── test_parser.py
    ├── test_detector.py
    └── test_pipeline.py
```

---

## KPI Targets

| Metric | Target | SOTA Baseline |
|--------|--------|---------------|
| Task Success Rate (TSR) | 80–85% | 70–75% (RT-1-X) |
| Goal Condition Accuracy (GCA) | 90% | 80–85% (VLA benchmarks) |
| Command Interpretation Accuracy (CIA) | 85% | 75–80% (CLIPort) |
| Task Completion Rate (TCR) | 80% | 65–70% (imitation learning) |
| Error Analysis Coverage | 100% | N/A |

---

## Citation

If you use this work, please cite:
```bibtex
@software{robolang2025,
  title={RoboLang: Natural Language Guided Robotic Manipulation},
  year={2025},
  url={https://github.com/your-org/robotic_nlp_manipulation}
}
```
