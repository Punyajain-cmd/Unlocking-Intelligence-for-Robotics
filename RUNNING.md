# RoboLang Universal VLA — Running Guide

## Table of Contents
1. [Running on Your Laptop](#1-running-on-your-laptop)
2. [Plugging into a Real Robot](#2-plugging-into-a-real-robot)
3. [Training the Model](#3-training-the-model)
4. [Command Reference](#4-command-reference)

---

## 1. Running on Your Laptop

### Prerequisites

**Python 3.10 or 3.12 recommended.**

```bash
# Clone the repo
git clone https://github.com/Punyajain-cmd/Unlocking-Intelligence-for-Robotics
cd Unlocking-Intelligence-for-Robotics

# Install dependencies (CPU-only — no GPU needed for the demo)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers timm opencv-python numpy scipy pillow nltk
pip install sentence-transformers ikpy

# Optional: GPU (CUDA 12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Run the demo (no camera needed — synthetic frames)

```bash
# Run all 8 classic manipulation scenarios + 5-robot universal VLA demo
python demo.py

# Run a single natural-language command (pick a robot)
python inference.py --robot kuka_iiwa7 \
    --command "Move the blue block to the right of the green cube."

python inference.py --robot ur5 \
    --command "Pick up the red sphere and place it on the yellow platform."

python inference.py --robot franka_panda \
    --command "Stack the cyan cylinder on top of the purple block."

python inference.py --robot shadow_hand \
    --command "Grasp the red sphere using a pinch grip."

# Show all available robot presets
python inference.py --list-robots

# Run verbose output (shows motor commands + trajectories)
python inference.py --robot ur5 \
    --command "Pick up the red sphere." \
    --verbose

# Run the multi-robot demo (all 5 robots, one model)
python inference.py --demo
```

### Run from a video file

```bash
# Provide any video; the pipeline extracts 16 frames automatically
python inference.py --robot ur5 \
    --video /path/to/scene.mp4 \
    --command "Move the blue block to the left."
```

### Run from your webcam

```bash
python inference.py --robot kuka_iiwa7 \
    --webcam \
    --command "Pick up the red object."
```

### Adapt to a new environment (sim2real)

```bash
# Point to a folder of calibration images from the real scene
python inference.py --robot ur5 \
    --adapt \
    --calib-dir ./my_scene_images/ \
    --command "Move the block to the platform."
```

### Run training (optional, improves quality)

```bash
# Train on synthetic data (fast, runs on CPU)
python train_universal.py --epochs 5 --batch 8

# Train with the best recipe (GPU recommended)
python train_universal.py --epochs 50 --batch 32 --pretrained --amp

# Resume from checkpoint
python train_universal.py --resume checkpoints/universal_vla.pt
```

---

## 2. Plugging into a Real Robot

The model outputs `MotorCommand` objects containing:
- `joint_positions` — target joint angles (radians) for all DOF
- `joint_velocities` — feedforward velocities (rad/s)
- `gripper_width` — gripper opening in metres (0 = closed, 0.08 = open)
- `control_mode` — "position" | "velocity" | "torque"

### Step 1: Register your robot

Either use a built-in preset (`kuka_iiwa7`, `ur5`, `franka_panda`, `shadow_hand`)
or write a YAML config for your robot:

```yaml
# my_robot.yaml
name: my_robot
dof: 6
description: "My 6-DOF arm"
control_mode: position
joints:
  - name: joint_1
    type: revolute
    axis: [0, 0, 1]
    limit: [-3.14, 3.14]
    max_vel: 2.0
    max_effort: 150.0
  # ... repeat for each joint
gripper_joints: [gripper_joint]
```

```python
from robot.robot_config import RobotConfig
cfg = RobotConfig.from_yaml("my_robot.yaml")
```

### Step 2: Build the pipeline

```python
from universal_pipeline import UniversalPipeline

# Built-in preset
pipe = UniversalPipeline.for_robot("ur5")

# Custom robot
pipe = UniversalPipeline(robot_cfg=cfg)

# With a trained checkpoint
pipe = UniversalPipeline.for_robot("ur5", model_path="checkpoints/universal_vla.pt")
```

### Step 3: Feed frames from your camera

```python
import cv2

cap = cv2.VideoCapture(0)          # or RTSP stream, RealSense, etc.
frames = []
for _ in range(16):
    ret, bgr = cap.read()
    if ret:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

result = pipe.run(
    frames  = frames,
    command = "Pick up the red cube and place it on the platform.",
)
```

### Step 4: Send motor commands to the robot

The pipeline produces a list of `MotorCommand` objects. Hook them to your
robot's SDK. Examples for common SDKs:

#### Universal Robots (UR5/UR10) — using `ur_rtde`
```python
import rtde_control, rtde_receive

rtde_c = rtde_control.RTDEControlInterface("192.168.1.100")

for cmd in result.motor_commands:
    q  = cmd.joint_positions.tolist()   # list of 6 joint angles (rad)
    qd = (cmd.joint_velocities.tolist()
          if cmd.joint_velocities is not None else [0.5]*6)
    rtde_c.moveJ(q, speed=0.3, acceleration=0.5)
```

#### Franka Panda — using `frankx` or `panda_robot`
```python
from panda_robot import PandaRobot

panda = PandaRobot()
for cmd in result.motor_commands:
    panda.move_to_joint_position(cmd.joint_positions)
    # gripper
    if cmd.gripper_width < 0.01:
        panda.grasp(width=0.0, force=20)
    else:
        panda.move_gripper(cmd.gripper_width)
```

#### Kuka iiwa — using `iiwa_ros` or `LBRiiwa`
```python
import rospy
from std_msgs.msg import Float64MultiArray

pub = rospy.Publisher('/iiwa/command/JointPosition',
                      Float64MultiArray, queue_size=1)

for cmd in result.motor_commands:
    msg = Float64MultiArray(data=cmd.joint_positions.tolist())
    pub.publish(msg)
    rospy.sleep(0.05)
```

#### ROS 2 Generic (any robot with a joint trajectory controller)
```python
import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

rclpy.init()
node = rclpy.create_node('robovla')
pub  = node.create_publisher(JointTrajectory, '/joint_trajectory', 10)

cmds = result.motor_commands
traj = JointTrajectory()
traj.joint_names = ['joint_1', 'joint_2', 'joint_3',
                    'joint_4', 'joint_5', 'joint_6']

for i, cmd in enumerate(cmds):
    pt = JointTrajectoryPoint()
    pt.positions = cmd.joint_positions.tolist()
    pt.time_from_start.sec = i  # 1 s per step (tune for your robot)
    traj.points.append(pt)

pub.publish(traj)
rclpy.spin_once(node, timeout_sec=1)
```

#### PyBullet simulation (testing before deploying)
```python
import pybullet as p

robot_id = p.loadURDF("ur5.urdf")

for cmd in result.motor_commands:
    for j, angle in enumerate(cmd.joint_positions):
        p.setJointMotorControl2(
            robot_id, j,
            controlMode = p.POSITION_CONTROL,
            targetPosition = angle,
        )
    p.stepSimulation()
```

### Step 5: Adapt to your real scene (one-time calibration)

Before running, capture 50–200 images of your workspace and call:

```python
import cv2, glob, numpy as np

calib_imgs = []
for f in sorted(glob.glob("calib/*.jpg"))[:100]:
    bgr = cv2.imread(f)
    calib_imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# One-shot domain adaptation — updates BatchNorm + AdaptiveLayerNorm
pipe.adapt_to_environment(calib_imgs, verbose=True)
```

This calibration takes ~2 seconds and dramatically reduces the sim2real gap.

### Latency tips for real-time deployment

- Use a GPU: latency drops from ~80 ms to ~8 ms per inference step.
- Set `use_temporal=False` for single-frame mode (~3x faster).
- Enable latency optimizer:
  ```python
  from models.latency_optimizer import LatencyOptimizer
  opt = LatencyOptimizer(pipe.vla_model)
  opt.enable_inference_cache(maxsize=64)
  opt.quantize_dynamic()   # INT8 quantization, 2-4x speedup
  ```

---

## 3. Training the Model

### On synthetic data (no GPU needed)
```bash
python train_universal.py --epochs 5 --batch 8
```

### With best settings (GPU recommended)
```bash
python train_universal.py \
    --epochs 50 \
    --batch 32 \
    --pretrained \
    --amp \
    --cosine-schedule \
    --grad-clip 1.0 \
    --ema-decay 0.999
```

### With Open X-Embodiment data (real robot demonstrations)
```bash
# Download a subset — see data/openx_loader.py for instructions
python train_universal.py \
    --dataset openx \
    --data-dir ./data/openx/ \
    --epochs 20 \
    --batch 16 \
    --pretrained \
    --amp
```

### Fine-tune on your own robot data

```bash
# Collect demonstrations as a numpy archive:
# episodes.npz: keys = frames (N,T,H,W,3), actions (N,T,DOF), commands (N,)
python train_universal.py \
    --dataset custom \
    --data-dir ./my_robot_demos/ \
    --robot ur5 \
    --finetune checkpoints/universal_vla.pt \
    --epochs 10 \
    --batch 8 \
    --lr 1e-4
```

---

## 4. Command Reference

| Script | Description |
|--------|-------------|
| `python demo.py` | 8 classic scenarios + 5-robot universal demo |
| `python inference.py --demo` | Multi-robot demo |
| `python inference.py --robot X --command "..."` | Single inference |
| `python inference.py --robot X --webcam --command "..."` | Webcam mode |
| `python inference.py --robot X --video V.mp4 --command "..."` | Video file |
| `python inference.py --robot X --adapt --calib-dir ./imgs/` | With adaptation |
| `python inference.py --list-robots` | List presets |
| `python train_universal.py` | Train (synthetic) |
| `python train_universal.py --pretrained --amp --epochs 50` | Best training |
| `python pipeline.py --command "..."` | Classic pipeline |

### Available robots

| Preset | DOF | Description |
|--------|-----|-------------|
| `simple_2dof` | 2 | Planar demo arm |
| `kuka_iiwa7` | 7 | KUKA LBR iiwa 7 |
| `ur5` | 6 | Universal Robots UR5 |
| `franka_panda` | 7 | Franka Emika Panda |
| `shadow_hand` | 23 | Shadow Dexterous Hand |
