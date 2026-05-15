"""
config.py
─────────
Centralised configuration & hyperparameters for RoboLang.
All tuneable values live here so every module imports from one source.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = ROOT_DIR / "data"
MODEL_DIR  = ROOT_DIR / "checkpoints"
LOG_DIR    = ROOT_DIR / "logs"
ASSET_DIR  = ROOT_DIR / "assets"

for _d in [DATA_DIR, MODEL_DIR, LOG_DIR, ASSET_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Language / NLP
# ──────────────────────────────────────────────
@dataclass
class LanguageConfig:
    # HuggingFace model for command parsing
    bert_model: str = "bert-base-uncased"
    max_seq_len: int = 128
    # Recognised action verbs
    action_verbs: List[str] = field(default_factory=lambda: [
        "move", "pick", "place", "put", "push", "pull",
        "grasp", "grab", "lift", "stack", "slide", "rotate",
        "carry", "transfer", "drop", "release", "bring"
    ])
    # Spatial relation words
    spatial_relations: List[str] = field(default_factory=lambda: [
        "left", "right", "above", "below", "on", "under",
        "beside", "next to", "in front of", "behind",
        "on top of", "near", "far", "between", "inside"
    ])
    # Object colours the parser recognises
    colours: List[str] = field(default_factory=lambda: [
        "red", "blue", "green", "yellow", "orange", "purple",
        "pink", "cyan", "white", "black", "brown", "grey", "gray"
    ])
    # Object shape / type words
    shapes: List[str] = field(default_factory=lambda: [
        "block", "cube", "sphere", "ball", "cylinder", "cone",
        "box", "platform", "tray", "plate", "ring", "object"
    ])
    confidence_threshold: float = 0.75


# ──────────────────────────────────────────────
# Vision / Perception
# ──────────────────────────────────────────────
@dataclass
class VisionConfig:
    # Camera resolution
    image_width:  int = 640
    image_height: int = 480
    fov_degrees:  float = 60.0
    near_plane:   float = 0.1
    far_plane:    float = 10.0

    # Colour detection HSV ranges  {name: (lower_hsv, upper_hsv)}
    colour_ranges: Dict[str, Tuple] = field(default_factory=lambda: {
        "red":    ((0,   120,  70), (10,  255, 255)),
        "red2":   ((170, 120,  70), (180, 255, 255)),   # wrap-around
        "blue":   ((100, 150,  50), (140, 255, 255)),
        "green":  ((40,   40,  40), (80,  255, 255)),
        "yellow": ((20,  100, 100), (40,  255, 255)),
        "orange": ((10,  100, 100), (25,  255, 255)),
        "purple": ((130,  50,  50), (160, 255, 255)),
        "cyan":   ((80,  100, 100), (100, 255, 255)),
        "white":  ((0,     0, 200), (180,  30, 255)),
        "black":  ((0,     0,   0), (180, 255,  50)),
        "brown":  ((10,   60,  20), (20,  255, 200)),
        "grey":   ((0,     0,  50), (180,  50, 200)),
    })

    min_contour_area: int = 500       # px² – noise filter
    depth_scale:      float = 0.001   # metres per depth unit
    detection_conf:   float = 0.6


# ──────────────────────────────────────────────
# Action / Motion
# ──────────────────────────────────────────────
@dataclass
class ActionConfig:
    # IK solver
    ik_max_iter:     int   = 200
    ik_tolerance:    float = 1e-4

    # Trajectory parameters
    waypoints:            int   = 20      # steps per move segment
    pre_grasp_height:     float = 0.15    # m above object
    grasp_height_offset:  float = 0.005   # m into object for gripper
    place_height_offset:  float = 0.02    # m above target surface
    velocity_scale:       float = 0.5     # fraction of max joint vel

    # Gripper
    gripper_open_width:   float = 0.08    # m
    gripper_close_width:  float = 0.01    # m

    # Supported actions and their symbolic mappings
    action_map: Dict[str, str] = field(default_factory=lambda: {
        "move":    "pick_and_place",
        "pick":    "grasp",
        "place":   "place",
        "put":     "pick_and_place",
        "push":    "push",
        "pull":    "pull",
        "grasp":   "grasp",
        "grab":    "grasp",
        "lift":    "lift",
        "stack":   "stack",
        "slide":   "push",
        "rotate":  "rotate",
        "carry":   "pick_and_place",
        "transfer":"pick_and_place",
        "drop":    "place",
        "release": "place",
        "bring":   "pick_and_place",
    })


# ──────────────────────────────────────────────
# Simulation (PyBullet)
# ──────────────────────────────────────────────
@dataclass
class SimulationConfig:
    gravity:            float = -9.81
    timestep:           float = 1.0 / 240.0
    solver_iterations:  int   = 150
    use_gui:            bool  = True
    render_width:       int   = 1280
    render_height:      int   = 720

    # Table
    table_height:   float = 0.625   # m
    table_half_ext: float = 0.40    # m (half-size of square table)

    # Robot URDF (uses PyBullet built-in if not overridden)
    robot_urdf:     str   = "kuka_iiwa/model.urdf"
    robot_base_pos: Tuple = (0.0, -0.5, table_height)
    robot_base_orn: Tuple = (0.0, 0.0, 0.0, 1.0)   # quaternion

    # Object spawn region (metres, relative to table centre)
    spawn_x_range: Tuple = (-0.25, 0.25)
    spawn_y_range: Tuple = (-0.20, 0.20)
    spawn_z_offset: float = 0.02   # above table surface

    # Object sizes
    block_half_size:    float = 0.025
    sphere_radius:      float = 0.025
    cylinder_radius:    float = 0.022
    cylinder_height:    float = 0.050

    # Simulation speed
    sim_steps_per_action: int = 240   # 1 second of physics per action step


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Model
    visual_backbone:   str   = "dinov2_small"
    language_backbone: str   = "bert-base-uncased"
    hidden_dim:        int   = 512
    num_action_bins:   int   = 256
    num_transformer_layers: int = 6
    num_heads:         int   = 8
    dropout:           float = 0.1

    # Training loop
    batch_size:        int   = 32
    num_epochs:        int   = 50
    learning_rate:     float = 1e-4
    weight_decay:      float = 1e-5
    warmup_steps:      int   = 1000
    grad_clip:         float = 1.0
    mixed_precision:   bool  = True

    # Data
    train_split: float = 0.8
    val_split:   float = 0.1
    test_split:  float = 0.1
    num_workers: int   = 4

    # Checkpointing
    save_every:          int  = 5      # epochs
    eval_every:          int  = 1
    checkpoint_dir:      Path = MODEL_DIR
    resume_from:         Optional[str] = None

    # Logging
    use_wandb:           bool = False
    project_name:        str  = "robolang"
    experiment_name:     str  = "vla_baseline"


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
@dataclass
class EvalConfig:
    num_episodes:         int   = 100
    max_steps_per_task:   int   = 50
    success_threshold:    float = 0.05   # m positional tolerance
    angular_threshold:    float = 0.1    # rad orientation tolerance

    # KPI targets (for pass/fail reporting)
    target_tsr:  float = 0.82  # Task Success Rate
    target_gca:  float = 0.90  # Goal Condition Accuracy
    target_cia:  float = 0.85  # Command Interpretation Accuracy
    target_tcr:  float = 0.80  # Task Completion Rate


# ──────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────
@dataclass
class Config:
    language:   LanguageConfig   = field(default_factory=LanguageConfig)
    vision:     VisionConfig     = field(default_factory=VisionConfig)
    action:     ActionConfig     = field(default_factory=ActionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    eval:       EvalConfig       = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        for section, values in (raw or {}).items():
            if hasattr(cfg, section) and isinstance(values, dict):
                sec = getattr(cfg, section)
                for k, v in values.items():
                    if hasattr(sec, k):
                        setattr(sec, k, v)
        return cfg

    def to_yaml(self, path: str) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)


# Singleton default config
DEFAULT_CONFIG = Config()
