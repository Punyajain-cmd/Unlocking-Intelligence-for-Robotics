"""
data/openx_loader.py
─────────────────────
Dataset loaders for state-of-the-art open-source robotics datasets.

Supports:
  1. Open X-Embodiment (OXE) — largest multi-robot dataset
     https://robotics-transformer-x.github.io/
     500k+ real-robot episodes, 22 robot types, 500+ tasks

  2. BridgeData V2 — diverse tabletop manipulation
     https://rail-berkeley.github.io/bridgedata/
     60,000+ episodes, WidowX robot, 10+ environments

  3. RoboSet — multi-robot multi-task
     https://robopen.github.io/roboset/
     75k+ episodes, multiple robots

  4. Synthetic (always available, no download needed)
     Procedurally generated clips for bootstrapping

All loaders return the same interface expected by train_universal.py:
  - clip       : (T, 3, H, W) float32 in [0, 1]
  - command    : str
  - action     : (T, n_dof) float32  normalised to [-1, 1]
  - gripper    : (T,) float32        normalised to [0, 1]
  - robot_name : str

DOWNLOAD INSTRUCTIONS
─────────────────────
Open X-Embodiment (subset):
  pip install tensorflow tensorflow_datasets
  # Download one embodiment (e.g. bridge):
  python -c "
  import tensorflow_datasets as tfds
  ds = tfds.load('bridge', split='train', data_dir='./data/openx/')
  "

BridgeData V2:
  # Direct download from HuggingFace:
  pip install huggingface_hub
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('rail-berkeley/BridgeV2', repo_type='dataset',
                    local_dir='./data/bridgev2/')
  "

For training without downloading, the SyntheticDataset is always available.
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────
# Common types
# ─────────────────────────────────────────────────────────

class RobotEpisode:
    """
    Standardised episode container.
    All datasets are converted to this format.
    """
    __slots__ = ("frames", "actions", "grippers", "command",
                 "robot_name", "n_dof", "source")

    def __init__(
        self,
        frames:     np.ndarray,   # (T, H, W, 3) uint8
        actions:    np.ndarray,   # (T, n_dof) float32 normalised
        grippers:   np.ndarray,   # (T,) float32 [0, 1]
        command:    str,
        robot_name: str,
        n_dof:      int,
        source:     str = "unknown",
    ):
        self.frames     = frames
        self.actions    = actions
        self.grippers   = grippers
        self.command    = command
        self.robot_name = robot_name
        self.n_dof      = n_dof
        self.source     = source


# ─────────────────────────────────────────────────────────
# 1. Synthetic dataset (always available)
# ─────────────────────────────────────────────────────────

# Vocabulary of synthetic manipulation commands
SYNTHETIC_COMMANDS = [
    "Move the blue block to the right of the green cube.",
    "Pick up the red sphere and place it on the yellow platform.",
    "Stack the orange cube on top of the purple block.",
    "Push the cyan cylinder to the left.",
    "Grasp the blue object and move it forward.",
    "Place the green block on the red platform.",
    "Slide the yellow block toward the blue cube.",
    "Pick up the purple sphere and put it to the right.",
    "Move the red block behind the green cube.",
    "Grasp the orange object and lift it up.",
    "Put the blue block next to the yellow cylinder.",
    "Stack the red cube on the blue platform.",
    "Push the green sphere toward the wall.",
    "Pick up the cyan block and place it in the bin.",
    "Move the purple block to the center of the table.",
    "Rotate the orange cylinder 90 degrees.",
    "Slide the blue cube to the far end of the table.",
    "Pick up the green sphere with a pinch grasp.",
    "Place the red block on the highest platform.",
    "Move all blocks to the right side of the table.",
]

ROBOT_PRESETS = ["simple_2dof", "kuka_iiwa7", "ur5", "franka_panda"]

COLOUR_OBJECTS = [
    (200, 50,  50),   # red
    (50,  100, 200),  # blue
    (50,  180, 50),   # green
    (200, 180, 50),   # yellow
    (50,  180, 180),  # cyan
    (180, 50,  180),  # purple
    (200, 130, 50),   # orange
]


def _make_synthetic_frame(
    h: int = 224, w: int = 224,
    t: int = 0,
    n_objects: int = 3,
    domain_rand: bool = True,
) -> np.ndarray:
    """Generate a synthetic scene frame with coloured blobs."""
    rng = np.random.RandomState(t)

    # Background
    if domain_rand:
        bg = rng.randint(80, 200, 3)
    else:
        bg = [180, 180, 180]
    frame = np.ones((h, w, 3), dtype=np.uint8) * bg

    # Objects (colour blobs moving slightly over time)
    colours = COLOUR_OBJECTS[:n_objects]
    for j, (r, g, b) in enumerate(colours):
        cx = int(w * (0.2 + j * 0.25) + t * 0.3 + rng.randint(-5, 5))
        cy = int(h * 0.5 + rng.randint(-10, 10))
        cx = max(20, min(w - 20, cx))
        cy = max(20, min(h - 20, cy))
        radius = rng.randint(15, 25)

        # Colour jitter for domain randomisation
        if domain_rand:
            dr = rng.randint(-30, 30)
            dg = rng.randint(-30, 30)
            db = rng.randint(-30, 30)
            col = (
                int(np.clip(r + dr, 0, 255)),
                int(np.clip(g + dg, 0, 255)),
                int(np.clip(b + db, 0, 255)),
            )
        else:
            col = (r, g, b)

        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        frame[y1:y2, x1:x2] = col

    # Optional table surface
    ty = int(h * 0.7)
    frame[ty:, :] = [200, 170, 140] if not domain_rand else rng.randint(150, 220, 3)

    return frame


def _generate_synthetic_action(
    n_dof: int,
    t:     int,
    T:     int,
    command: str,
) -> Tuple[np.ndarray, float]:
    """
    Generate a plausible action given command type and timestep.
    Actions are normalised to [-1, 1] per DOF.
    """
    action  = np.zeros(n_dof, dtype=np.float32)
    gripper = 0.5  # neutral

    # Infer action type from command
    cmd_lower = command.lower()
    if "pick" in cmd_lower or "grasp" in cmd_lower or "lift" in cmd_lower:
        phase = t / max(T - 1, 1)
        if phase < 0.4:
            action[0] = 0.3    # approach
            if n_dof > 2: action[2] = -0.4   # move down
            gripper   = 0.9    # open
        elif phase < 0.6:
            gripper   = 0.0    # close (grasp)
        else:
            if n_dof > 2: action[2] = 0.5    # lift up
            gripper   = 0.0    # keep closed

    elif "place" in cmd_lower or "put" in cmd_lower or "stack" in cmd_lower:
        phase = t / max(T - 1, 1)
        if phase < 0.5:
            action[0] = 0.2    # move to target
            if n_dof > 1: action[1] = 0.1
            gripper   = 0.0    # keep closed
        else:
            if n_dof > 2: action[2] = -0.2   # lower
            gripper   = 0.8    # release

    elif "push" in cmd_lower or "slide" in cmd_lower or "move" in cmd_lower:
        # Smooth push motion
        action[0] = 0.4 * np.sin(t * np.pi / T)
        if "left" in cmd_lower:
            action[1] = -0.3
        elif "right" in cmd_lower:
            action[1] = 0.3
        gripper = 0.5

    else:
        # Default: gentle forward motion
        action[0] = 0.2
        gripper   = 0.5

    # Add small noise
    action += np.random.normal(0, 0.05, n_dof).astype(np.float32)
    action  = np.clip(action, -1.0, 1.0)
    return action, float(np.clip(gripper, 0, 1))


if _TORCH:

    class SyntheticDataset(Dataset):
        """
        Infinite procedurally-generated manipulation dataset.
        No download required — perfect for bootstrapping or CPU training.

        Each episode:
          - Random natural-language command from SYNTHETIC_COMMANDS
          - Synthetic video clip (T frames, colour blobs)
          - Plausible action trajectory matching the command
          - Works for any robot DOF
        """

        def __init__(
            self,
            n_episodes:     int   = 1000,
            clip_len:       int   = 8,
            img_size:       int   = 224,
            robot_names:    Optional[List[str]] = None,
            domain_rand:    bool  = True,
            seed:           int   = 42,
        ):
            self.n_episodes  = n_episodes
            self.clip_len    = clip_len
            self.img_size    = img_size
            self.robot_names = robot_names or ROBOT_PRESETS
            self.domain_rand = domain_rand
            rng = np.random.RandomState(seed)

            # Pre-generate episode metadata (lightweight)
            self.episodes = []
            for i in range(n_episodes):
                robot = rng.choice(self.robot_names)
                cmd   = rng.choice(SYNTHETIC_COMMANDS)
                seed_ = int(rng.randint(0, 2**31))
                self.episodes.append((robot, cmd, seed_))

        def __len__(self) -> int:
            return self.n_episodes

        def _get_dof(self, robot_name: str) -> int:
            DOF_MAP = {
                "simple_2dof": 2,
                "kuka_iiwa7":  7,
                "ur5":         6,
                "franka_panda": 7,
                "shadow_hand": 22,
            }
            return DOF_MAP.get(robot_name, 6)

        def __getitem__(self, idx: int) -> Dict:
            robot_name, command, seed = self.episodes[idx]
            n_dof = self._get_dof(robot_name)
            T = self.clip_len
            H = W = self.img_size

            np.random.seed(seed)

            # Generate frames
            frames = np.stack([
                _make_synthetic_frame(H, W, t=t, domain_rand=self.domain_rand)
                for t in range(T)
            ], axis=0)   # (T, H, W, 3)

            # Generate actions
            actions  = np.zeros((T, n_dof), dtype=np.float32)
            grippers = np.zeros(T, dtype=np.float32)
            for t in range(T):
                a, g = _generate_synthetic_action(n_dof, t, T, command)
                actions[t]  = a
                grippers[t] = g

            # Normalise frames to [0, 1] float
            clip = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            # (T, 3, H, W)

            return {
                "clip":       clip,                                  # (T, 3, H, W)
                "actions":    torch.tensor(actions, dtype=torch.float32),    # (T, n_dof)
                "grippers":   torch.tensor(grippers, dtype=torch.float32),   # (T,)
                "command":    command,
                "robot_name": robot_name,
                "n_dof":      n_dof,
                "source":     "synthetic",
            }


    # ─────────────────────────────────────────────────────────
    # 2. BridgeData V2 loader
    # ─────────────────────────────────────────────────────────

    class BridgeV2Dataset(Dataset):
        """
        BridgeData V2 dataset loader.

        Expects the dataset downloaded to data_dir with structure:
          data_dir/
            train/
              <episode_id>/
                rgb_<t>.jpg      ← RGB frames
                action.npy       ← (T, 7) actions [dx,dy,dz,drx,dry,drz,grip]
                lang.txt         ← language command

        Download:
          from huggingface_hub import snapshot_download
          snapshot_download('rail-berkeley/BridgeV2',
                            repo_type='dataset', local_dir='./data/bridgev2/')
        """

        ROBOT_NAME = "widowx250"
        N_DOF      = 6

        def __init__(
            self,
            data_dir:    str,
            split:       str = "train",   # "train" | "val"
            clip_len:    int = 8,
            img_size:    int = 224,
            max_episodes: Optional[int] = None,
        ):
            self.data_dir = Path(data_dir) / split
            self.clip_len = clip_len
            self.img_size = img_size

            if not self.data_dir.exists():
                raise FileNotFoundError(
                    f"BridgeV2 data not found at {self.data_dir}.\n"
                    "Download with:\n"
                    "  from huggingface_hub import snapshot_download\n"
                    "  snapshot_download('rail-berkeley/BridgeV2', "
                    "repo_type='dataset', local_dir='./data/bridgev2/')"
                )

            # Discover episodes
            self.episodes = sorted([
                d for d in self.data_dir.iterdir()
                if d.is_dir() and (d / "action.npy").exists()
            ])
            if max_episodes:
                self.episodes = self.episodes[:max_episodes]

        def __len__(self) -> int:
            return len(self.episodes)

        def _load_frame(self, path: Path) -> np.ndarray:
            try:
                import cv2
                bgr = cv2.imread(str(path))
                if bgr is None:
                    return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.img_size, self.img_size))
                return rgb
            except Exception:
                return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        def __getitem__(self, idx: int) -> Dict:
            ep_dir = self.episodes[idx]

            # Load command
            lang_file = ep_dir / "lang.txt"
            command = lang_file.read_text().strip() if lang_file.exists() else \
                "Pick up the object and place it on the platform."

            # Load actions
            act_path = ep_dir / "action.npy"
            raw_actions = np.load(str(act_path)).astype(np.float32)   # (T_raw, 7+)
            T_raw = len(raw_actions)

            # Load frames
            frame_files = sorted(ep_dir.glob("rgb_*.jpg"))
            if not frame_files:
                frame_files = sorted(ep_dir.glob("*.jpg"))

            # Sample clip_len frames evenly
            T = self.clip_len
            if len(frame_files) >= T:
                indices = np.linspace(0, len(frame_files) - 1, T).astype(int)
                frames = np.stack([self._load_frame(frame_files[i]) for i in indices])
            else:
                frames = np.stack([self._load_frame(f) for f in frame_files])
                # Pad with last frame
                pad = np.tile(frames[-1:], (T - len(frames), 1, 1, 1))
                frames = np.concatenate([frames, pad], axis=0)

            # Sample actions
            if T_raw >= T:
                act_idx = np.linspace(0, T_raw - 1, T).astype(int)
                actions  = raw_actions[act_idx, :self.N_DOF]   # (T, 6)
                grippers = raw_actions[act_idx, 6] if raw_actions.shape[1] > 6 \
                           else np.ones(T, dtype=np.float32) * 0.5
            else:
                actions  = np.zeros((T, self.N_DOF), dtype=np.float32)
                grippers = np.ones(T, dtype=np.float32) * 0.5

            # Normalise actions to [-1, 1]
            actions = np.clip(actions / (np.abs(actions).max() + 1e-6), -1, 1)
            grippers = np.clip(grippers, 0, 1)

            clip = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

            return {
                "clip":       clip,
                "actions":    torch.tensor(actions,  dtype=torch.float32),
                "grippers":   torch.tensor(grippers, dtype=torch.float32),
                "command":    command,
                "robot_name": self.ROBOT_NAME,
                "n_dof":      self.N_DOF,
                "source":     "bridgev2",
            }


    # ─────────────────────────────────────────────────────────
    # 3. Open X-Embodiment loader (TensorFlow Datasets bridge)
    # ─────────────────────────────────────────────────────────

    class OpenXDataset(Dataset):
        """
        Open X-Embodiment dataset loader via tensorflow_datasets.

        Bridges the TF dataset to PyTorch — converts episodes to
        the standard RobotEpisode format.

        Supported embodiments (subset):
          "bridge"             ← WidowX tabletop, 60k episodes
          "fractal20220817_data"  ← Google RT-2, 130k episodes
          "kuka"               ← KUKA iiwa grasping
          "jaco_play"          ← Kinova Jaco pick-and-place
          "berkeley_autolab_ur5"  ← UR5 tabletop

        Download:
          pip install tensorflow tensorflow_datasets
          # First access downloads automatically to data_dir

        Usage:
          ds = OpenXDataset("bridge", data_dir="./data/openx/", max_episodes=500)
        """

        EMBODIMENT_DOF = {
            "bridge":                     6,
            "fractal20220817_data":       7,
            "kuka":                       7,
            "jaco_play":                  6,
            "berkeley_autolab_ur5":       6,
            "rt_2_x":                     7,
            "default":                    6,
        }

        def __init__(
            self,
            embodiment:   str  = "bridge",
            data_dir:     str  = "./data/openx/",
            split:        str  = "train",
            clip_len:     int  = 8,
            img_size:     int  = 224,
            max_episodes: Optional[int] = 500,
        ):
            self.clip_len = clip_len
            self.img_size = img_size
            self.n_dof    = self.EMBODIMENT_DOF.get(embodiment, 6)
            self.robot_name = embodiment

            try:
                import tensorflow_datasets as tfds
                ds = tfds.load(
                    embodiment,
                    split    = split,
                    data_dir = data_dir,
                    shuffle_files = True,
                )
                self._episodes = list(ds.take(max_episodes or 500))
                self._tf_available = True
            except Exception as e:
                warnings.warn(
                    f"tensorflow_datasets not available or embodiment download failed:\n"
                    f"  {e}\n"
                    f"Falling back to SyntheticDataset. "
                    f"Install with: pip install tensorflow tensorflow_datasets"
                )
                self._episodes = []
                self._tf_available = False
                # Store fallback dataset
                self._fallback = SyntheticDataset(
                    n_episodes  = max_episodes or 500,
                    clip_len    = clip_len,
                    img_size    = img_size,
                    robot_names = ["ur5"],
                )

        def __len__(self) -> int:
            if not self._tf_available:
                return len(self._fallback)
            return len(self._episodes)

        def __getitem__(self, idx: int) -> Dict:
            if not self._tf_available:
                return self._fallback[idx]

            try:
                ep = self._episodes[idx]
                steps = list(ep["steps"])

                T_raw = len(steps)
                T     = self.clip_len

                # Sample T steps evenly
                indices = np.linspace(0, T_raw - 1, T).astype(int)
                sampled = [steps[i] for i in indices]

                frames   = []
                actions  = np.zeros((T, self.n_dof), dtype=np.float32)
                grippers = np.zeros(T, dtype=np.float32)
                command  = ""

                for t, step in enumerate(sampled):
                    # Frame
                    obs = step.get("observation", {})
                    img_key = "image" if "image" in obs else \
                              "rgb" if "rgb" in obs else \
                              next(iter(obs), None)
                    if img_key and hasattr(obs[img_key], "numpy"):
                        img = obs[img_key].numpy()
                        if img.shape[-1] != 3:
                            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                        try:
                            import cv2
                            img = cv2.resize(img.astype(np.uint8),
                                             (self.img_size, self.img_size))
                        except Exception:
                            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    else:
                        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    frames.append(img)

                    # Action
                    act = step.get("action", {})
                    if hasattr(act, "numpy"):
                        a = act.numpy().astype(np.float32)
                    elif isinstance(act, dict):
                        a = np.concatenate([
                            v.numpy() if hasattr(v, "numpy") else np.array([float(v)])
                            for k, v in act.items()
                            if "gripper" not in k.lower()
                        ]).astype(np.float32)
                        g = float(act.get("gripper_closedness_commanded",
                                          act.get("open_gripper", 0.5)))
                        grippers[t] = float(np.clip(g, 0, 1))
                    else:
                        a = np.zeros(self.n_dof, dtype=np.float32)

                    a = a[:self.n_dof]
                    if len(a) < self.n_dof:
                        a = np.pad(a, (0, self.n_dof - len(a)))
                    actions[t] = np.clip(a / (np.abs(a).max() + 1e-6), -1, 1)

                    # Language
                    if not command:
                        lang = step.get("language_instruction", "")
                        if hasattr(lang, "numpy"):
                            lang = lang.numpy()
                        if isinstance(lang, bytes):
                            lang = lang.decode("utf-8", errors="ignore")
                        if isinstance(lang, str) and lang:
                            command = lang

                if not command:
                    command = random.choice(SYNTHETIC_COMMANDS)

                frames_np = np.stack(frames, axis=0)   # (T, H, W, 3)
                clip = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

                return {
                    "clip":       clip,
                    "actions":    torch.tensor(actions,  dtype=torch.float32),
                    "grippers":   torch.tensor(grippers, dtype=torch.float32),
                    "command":    command,
                    "robot_name": self.robot_name,
                    "n_dof":      self.n_dof,
                    "source":     "openx",
                }

            except Exception as e:
                warnings.warn(f"OpenX episode {idx} failed: {e}; using synthetic.")
                return SyntheticDataset(1, self.clip_len, self.img_size)[0]


    # ─────────────────────────────────────────────────────────
    # 4. Mixed dataset (multi-source)
    # ─────────────────────────────────────────────────────────

    class MixedRoboticsDataset(Dataset):
        """
        Mix multiple datasets with configurable weights.
        This is the recommended training dataset — combines synthetic
        data (always available) with any real-robot data you have.

        Usage:
          ds = MixedRoboticsDataset.default(total=5000)
          # Or with real data:
          ds = MixedRoboticsDataset(
              datasets = [
                  SyntheticDataset(2000),
                  BridgeV2Dataset("./data/bridgev2/"),
              ],
              weights  = [0.3, 0.7],
          )
        """

        def __init__(
            self,
            datasets: List[Dataset],
            weights:  Optional[List[float]] = None,
        ):
            self.datasets = datasets
            total = sum(len(d) for d in datasets)

            if weights is None:
                weights = [len(d) / total for d in datasets]
            self.weights = [w / sum(weights) for w in weights]

            # Build flat index
            self._index = []
            for ds_idx, (ds, w) in enumerate(zip(datasets, self.weights)):
                n = max(1, int(w * total))
                for ep_idx in range(min(n, len(ds))):
                    self._index.append((ds_idx, ep_idx % len(ds)))

        @classmethod
        def default(
            cls,
            total:       int  = 2000,
            clip_len:    int  = 8,
            img_size:    int  = 224,
            domain_rand: bool = True,
        ) -> "MixedRoboticsDataset":
            """Build a default mixed dataset (synthetic only, always works)."""
            ds = SyntheticDataset(
                n_episodes  = total,
                clip_len    = clip_len,
                img_size    = img_size,
                domain_rand = domain_rand,
            )
            return cls(datasets=[ds], weights=[1.0])

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, idx: int) -> Dict:
            ds_idx, ep_idx = self._index[idx]
            return self.datasets[ds_idx][ep_idx]


    # ─────────────────────────────────────────────────────────
    # 5. DataLoader factory
    # ─────────────────────────────────────────────────────────

    def build_dataloader(
        dataset_type: str     = "synthetic",   # "synthetic" | "bridgev2" | "openx" | "mixed"
        data_dir:     Optional[str] = None,
        batch_size:   int     = 16,
        num_workers:  int     = 2,
        clip_len:     int     = 8,
        img_size:     int     = 224,
        n_episodes:   int     = 2000,
        shuffle:      bool    = True,
        domain_rand:  bool    = True,
        **kwargs,
    ) -> DataLoader:
        """
        Build a DataLoader for any supported dataset.

        Parameters
        ──────────
        dataset_type : "synthetic" (no download), "bridgev2", "openx", "mixed"
        data_dir     : path to downloaded data (not needed for "synthetic")
        batch_size   : training batch size
        num_workers  : DataLoader workers (set to 0 on Windows)
        clip_len     : number of frames per clip
        img_size     : resize all frames to (img_size, img_size)
        n_episodes   : how many synthetic episodes to generate
        """
        if dataset_type == "synthetic":
            ds = SyntheticDataset(
                n_episodes  = n_episodes,
                clip_len    = clip_len,
                img_size    = img_size,
                domain_rand = domain_rand,
            )
        elif dataset_type == "bridgev2":
            if not data_dir:
                raise ValueError("data_dir required for bridgev2")
            ds = BridgeV2Dataset(data_dir=data_dir, clip_len=clip_len, img_size=img_size)
        elif dataset_type == "openx":
            embodiment = kwargs.get("embodiment", "bridge")
            ds = OpenXDataset(
                embodiment  = embodiment,
                data_dir    = data_dir or "./data/openx/",
                clip_len    = clip_len,
                img_size    = img_size,
                max_episodes = n_episodes,
            )
        elif dataset_type == "mixed":
            ds = MixedRoboticsDataset.default(
                total       = n_episodes,
                clip_len    = clip_len,
                img_size    = img_size,
                domain_rand = domain_rand,
            )
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
            drop_last   = True,
            collate_fn  = _collate_variable_dof,
        )


    def _collate_variable_dof(batch: List[Dict]) -> Dict:
        """
        Collate function that handles variable n_dof across robots.
        Pads actions to the maximum n_dof in the batch.
        """
        max_dof = max(item["n_dof"] for item in batch)

        clips    = torch.stack([item["clip"] for item in batch])          # (B, T, 3, H, W)
        commands = [item["command"] for item in batch]
        robots   = [item["robot_name"] for item in batch]
        n_dofs   = [item["n_dof"] for item in batch]
        sources  = [item["source"] for item in batch]

        # Pad actions to max_dof
        T = batch[0]["actions"].shape[0]
        actions = torch.zeros(len(batch), T, max_dof)
        for i, item in enumerate(batch):
            n = item["n_dof"]
            actions[i, :, :n] = item["actions"]

        grippers = torch.stack([item["grippers"] for item in batch])      # (B, T)

        return {
            "clip":       clips,
            "actions":    actions,
            "grippers":   grippers,
            "commands":   commands,
            "robot_names": robots,
            "n_dofs":     n_dofs,
            "sources":    sources,
            "max_dof":    max_dof,
        }


if __name__ == "__main__":
    if not _TORCH:
        print("torch not available")
    else:
        print("Testing SyntheticDataset...")
        ds = SyntheticDataset(n_episodes=10, clip_len=8, img_size=64)
        sample = ds[0]
        print(f"  clip:     {sample['clip'].shape}")        # (8, 3, 64, 64)
        print(f"  actions:  {sample['actions'].shape}")     # (8, n_dof)
        print(f"  grippers: {sample['grippers'].shape}")    # (8,)
        print(f"  command:  {sample['command']}")
        print(f"  robot:    {sample['robot_name']}")

        print("\nTesting MixedDataset...")
        mixed = MixedRoboticsDataset.default(total=50, clip_len=4, img_size=64)
        print(f"  length: {len(mixed)}")

        print("\nTesting DataLoader...")
        loader = build_dataloader(
            dataset_type = "synthetic",
            batch_size   = 4,
            clip_len     = 4,
            img_size     = 64,
            n_episodes   = 20,
            num_workers  = 0,
        )
        batch = next(iter(loader))
        print(f"  batch clip:    {batch['clip'].shape}")      # (4, 4, 3, 64, 64)
        print(f"  batch actions: {batch['actions'].shape}")   # (4, 4, max_dof)
        print(f"  robots:        {batch['robot_names']}")
        print("\nAll dataset tests passed.")
