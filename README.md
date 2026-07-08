# Mobile-Autonomous-Robotics

Techniques and principles for designing and developing mobile robots that interact
autonomously with their environment. Topics include sensors and actuators, kinematic
analysis, computer vision, state estimation and planning.

This repository contains the ROS 2 code, sensor characterisation tools, configuration
and reports for an autonomous **TurtleBot 4** search-and-return task (COMPSYS 732).

## Repository structure

```
.
├── scripts/                 # ROS 2 Python nodes
│   ├── autonomous_run.py    # Full autonomy FSM: search → report → return → done
│   ├── detect_and_stop.py   # Cube detection, wall following and obstacle avoidance
│   ├── lidar_logger.py      # LiDAR characterisation (snapshot / range / log modes)
│   └── odom_logger.py       # Odometry characterisation (linear / square modes)
├── config/
│   └── super_client_configuration_file.xml   # Fast DDS super-client config
└── reports/                 # Written reports
    ├── Report2_YanzhenHu.docx
    └── Evaluation of TurtleBot 4 Platform for Autonomous Stock.docx
```

## Scripts

| Script | Purpose |
| --- | --- |
| `autonomous_run.py` | Finite-state machine that sweeps a C-shaped arena, detects a cube, logs its position and returns to the start `(0, 0)`. |
| `detect_and_stop.py` | Vision-based cube detection with LiDAR obstacle avoidance and right-wall following. |
| `lidar_logger.py` | Captures/records LiDAR scans and reports mean, std-dev and % error vs ground truth. |
| `odom_logger.py` | Records odometry during a run and computes displacement/closing error metrics. |

## Requirements

- ROS 2 (with the TurtleBot 4 stack)
- Python 3 with `rclpy`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `cv_bridge`, `opencv-python`, `numpy`, `matplotlib`

## Usage

Replace `/TXX` with your robot namespace.

```bash
# Full autonomous run
~/ros2_venv/bin/python3 scripts/autonomous_run.py

# LiDAR characterisation
~/ros2_venv/bin/python3 scripts/lidar_logger.py --namespace /TXX --mode snapshot
~/ros2_venv/bin/python3 scripts/lidar_logger.py --namespace /TXX --mode range --ground-truth 1.0 --samples 50

# Odometry characterisation
~/ros2_venv/bin/python3 scripts/odom_logger.py --namespace /TXX --mode linear --target 1.0 --duration 30
~/ros2_venv/bin/python3 scripts/odom_logger.py --namespace /TXX --mode square --duration 60
```
