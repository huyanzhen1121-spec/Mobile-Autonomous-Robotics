#!/usr/bin/env python3
"""
Odometry Logger and Plotter
COMPSYS732

Records odometry data during a single test run and computes statistical
metrics. Run once per trial.

Modes:
  linear  — records a straight-line run; reports final displacement and error vs target
  square  — records a closed-loop square path; reports closing error

Usage (use venv to avoid NumPy 2.0 issues on lab PCs):
    ~/ros2_venv/bin/python3 odom_logger.py --namespace /TXX --mode linear --target 1.0 --duration 30
    ~/ros2_venv/bin/python3 odom_logger.py --namespace /TXX --mode square --duration 60
    ~/ros2_venv/bin/python3 odom_logger.py --namespace /TXX --mode linear --target 1.0 --duration 30 --trial 3
"""

import argparse
import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not found. Install with: pip3 install matplotlib --break-system-packages")
    sys.exit(1)


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle in degrees."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


class OdomLogger(Node):
    def __init__(self, namespace, mode, target, duration):
        super().__init__('odom_logger')
        self.namespace = namespace.rstrip('/')
        self.mode = mode
        self.target = target       # target distance in metres (linear mode)
        self.duration = duration
        self.data = []
        self.start_time = None
        self.recording = False

        topic = f"{self.namespace}/odom"
        self.subscription = self.create_subscription(Odometry, topic, self.odom_callback, 10)

        print(f"\n  Odometry Logger — Mode: {mode.upper()}")
        print(f"  Subscribing to: {topic}")
        if mode == 'linear':
            print(f"  Target distance: {target} m")
            print(f"  >> Start your test node now — robot should drive straight <<\n")
        else:
            print(f"  >> Start your test node now — robot should drive a 1m square <<\n")

        self.recording = True
        self.start_time = time.time()

    def odom_callback(self, msg):
        if not self.recording:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.recording = False
            return

        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        vel = msg.twist.twist

        self.data.append({
            'time':    round(elapsed, 4),
            'x':       round(pos.x, 6),
            'y':       round(pos.y, 6),
            'yaw_deg': round(quaternion_to_yaw(ori), 4),
            'vel_lin': round(vel.linear.x, 6),
            'vel_ang': round(vel.angular.z, 6),
        })

    def compute_stats(self):
        """Compute and return key metrics for this trial."""
        if not self.data:
            return {}
        last  = self.data[-1]
        first = self.data[0]

        if self.mode == 'linear':
            # Displacement along X axis (assuming straight-line motion in X)
            displacement = math.sqrt(
                (last['x'] - first['x'])**2 + (last['y'] - first['y'])**2
            )
            error_m  = displacement - self.target
            error_pct = (abs(error_m) / self.target) * 100.0 if self.target > 0 else 0.0
            raw_diff = last['yaw_deg'] - first['yaw_deg']
            heading_error = abs(((raw_diff + 180) % 360) - 180)
            return {
                'mode':           self.mode,
                'target_m':       self.target,
                'displacement_m': round(displacement, 4),
                'error_m':        round(error_m, 4),
                'error_pct':      round(error_pct, 2),
                'heading_error_deg': round(heading_error, 2),
                'samples':        len(self.data),
            }
        else:
            # Closing error — distance from final pose back to origin
            closing_error = math.sqrt(last['x']**2 + last['y']**2)
            raw_diff = last['yaw_deg'] - first['yaw_deg']
            heading_return = abs(((raw_diff + 180) % 360) - 180)
            return {
                'mode':              self.mode,
                'closing_error_m':   round(closing_error, 4),
                'final_x':           round(last['x'], 4),
                'final_y':           round(last['y'], 4),
                'heading_error_deg': round(heading_return, 2),
                'samples':           len(self.data),
            }

    def print_summary(self, stats):
        print(f"\n  --- Odometry Trial Summary ({self.mode.upper()}) ---")
        for k, v in stats.items():
            print(f"  {k:<25} {v}")
        print(f"  {'─'*40}\n")

    def save_csv(self, path):
        if not self.data:
            print("WARNING: No data collected.")
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        print(f"  Raw data CSV saved: {path}")

    def save_stats_csv(self, stats, path):
        """Append this trial's stats to a cumulative stats CSV."""
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
        print(f"  Stats appended to: {path}")

    def plot(self, path):
        if not self.data:
            return

        times   = [r['time']    for r in self.data]
        xs      = [r['x']       for r in self.data]
        ys      = [r['y']       for r in self.data]
        yaws    = [r['yaw_deg'] for r in self.data]
        vel_lin = [r['vel_lin'] for r in self.data]

        fig = plt.figure(figsize=(16, 10))
        title = ('Odometry — Linear Run' if self.mode == 'linear'
                 else 'Odometry — Square Path')
        fig.suptitle(f'{title} | TurtleBot 4 | Namespace: {self.namespace}',
                     fontsize=14, fontweight='bold')

        # Trajectory plot
        ax_traj = fig.add_subplot(1, 2, 1)
        ax_traj.plot(xs, ys, color='#2980B9', linewidth=2, label='Estimated path')
        ax_traj.plot(xs[0], ys[0], 'go', markersize=12, label='Start', zorder=5)
        ax_traj.plot(xs[-1], ys[-1], 'r^', markersize=12,
                     label=f'End ({xs[-1]:.3f}, {ys[-1]:.3f})', zorder=5)

        if self.mode == 'linear':
            # Draw ideal straight line
            ax_traj.plot([0, self.target], [0, 0], 'k--',
                         linewidth=1.2, label=f'Ideal {self.target}m straight')
            ax_traj.set_title(
                f'XY Trajectory\nTarget: {self.target}m  |  '
                f'Actual: {math.sqrt((xs[-1]-xs[0])**2+(ys[-1]-ys[0])**2):.4f}m', fontsize=12)
        else:
            # Draw ideal 1m square
            sq = plt.Polygon([[0,0],[1,0],[1,1],[0,1],[0,0]], fill=False,
                              edgecolor='gray', linestyle='--', linewidth=1.2,
                              label='Ideal 1m square')
            ax_traj.add_patch(sq)
            closing = math.sqrt(xs[-1]**2 + ys[-1]**2)
            ax_traj.plot([xs[-1], xs[0]], [ys[-1], ys[0]], 'r--',
                         linewidth=1, alpha=0.7, label=f'Closing error: {closing:.4f}m')
            ax_traj.set_title(f'XY Trajectory\nClosing error: {closing:.4f} m', fontsize=12)

        ax_traj.set_xlabel('X position (metres)', fontsize=11)
        ax_traj.set_ylabel('Y position (metres)', fontsize=11)
        ax_traj.legend(fontsize=9)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect('equal')

        margin = 0.3
        ref_pts_x = [0, self.target] if self.mode == 'linear' else [0, 1]
        ref_pts_y = [0, 0]           if self.mode == 'linear' else [0, 1]
        ax_traj.set_xlim(min(xs + ref_pts_x) - margin, max(xs + ref_pts_x) + margin)
        ax_traj.set_ylim(min(ys + ref_pts_y) - margin, max(ys + ref_pts_y) + margin)

        # Velocity plot
        ax_vel = fig.add_subplot(2, 2, 2)
        ax_vel.plot(times, vel_lin, color='#E74C3C', linewidth=1.2)
        ax_vel.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax_vel.set_ylabel('Linear Velocity (m/s)', fontsize=10)
        ax_vel.set_title('Linear Velocity over Time', fontsize=11)
        ax_vel.grid(True, alpha=0.3)

        # Heading plot
        ax_yaw = fig.add_subplot(2, 2, 4)
        ax_yaw.plot(times, yaws, color='#27AE60', linewidth=1.2)
        ax_yaw.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax_yaw.set_ylabel('Heading (degrees)', fontsize=10)
        ax_yaw.set_xlabel('Time (seconds)', fontsize=10)
        ax_yaw.set_title('Heading Angle over Time', fontsize=11)
        ax_yaw.grid(True, alpha=0.3)

        fig.text(0.01, 0.01,
                 f"Samples: {len(self.data)}  |  Duration: {times[-1]:.1f}s  |  "
                 f"Namespace: {self.namespace}",
                 fontsize=8, color='gray')

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Odometry Logger — Test Engineering')
    parser.add_argument('--namespace', type=str, required=True,
                        help='Robot namespace e.g. /T10')
    parser.add_argument('--mode', type=str, choices=['linear', 'square'],
                        default='linear',
                        help='linear: straight-line run | square: closed-loop square')
    parser.add_argument('--target', type=float, default=1.0,
                        help='Target distance in metres (linear mode only, default 1.0)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Recording duration in seconds')
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number — used in output filenames')
    args = parser.parse_args()

    base      = os.path.expanduser('~')
    label     = f"{args.mode}_t{args.trial}"
    csv_path  = os.path.join(base, f'odom_{label}.csv')
    plot_path = os.path.join(base, f'odom_{label}.png')
    stats_path = os.path.join(base, 'odom_stats.csv')   # cumulative across trials

    rclpy.init()
    node = OdomLogger(args.namespace, args.mode, args.target, args.duration)

    try:
        end_time = time.time() + args.duration + 2
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(node, timeout_sec=0.05)
            if not node.recording and node.data:
                break
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        stats = node.compute_stats()
        node.print_summary(stats)
        node.save_csv(csv_path)
        node.save_stats_csv(stats, stats_path)
        node.plot(plot_path)
        node.destroy_node()
        rclpy.shutdown()

    print(f"\n  Done!")
    print(f"  Plot:        eog {plot_path}")
    print(f"  Raw data:    {csv_path}")
    print(f"  Stats (all trials): {stats_path}\n")


if __name__ == '__main__':
    main()
