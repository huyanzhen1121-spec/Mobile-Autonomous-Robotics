#!/usr/bin/env python3
"""
LiDAR Logger and Plotter — Test Engineering Version
COMPSYS732

Three modes:
  snapshot  — captures one scan and saves a polar plot (exploration)
  range     — samples N readings at a stationary position; reports mean, std dev,
              and % error vs a known ground-truth distance
  log       — records scans over time and plots min distance vs time (motion test)

Usage (use venv to avoid NumPy 2.0 issues on lab PCs):
    ~/ros2_venv/bin/python3 lidar_logger.py --namespace /TXX --mode snapshot
    ~/ros2_venv/bin/python3 lidar_logger.py --namespace /TXX --mode range --ground-truth 1.0 --samples 50
    ~/ros2_venv/bin/python3 lidar_logger.py --namespace /TXX --mode range --ground-truth 1.0 --samples 50 --trial 2
    ~/ros2_venv/bin/python3 lidar_logger.py --namespace /TXX --mode log --duration 30
"""

import argparse
import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib/numpy not found.")
    print("Install with: pip3 install matplotlib numpy --break-system-packages")
    sys.exit(1)


class LidarLogger(Node):
    def __init__(self, namespace, mode, duration, ground_truth, n_samples):
        super().__init__('lidar_logger')
        self.namespace    = namespace.rstrip('/')
        self.mode         = mode
        self.duration     = duration
        self.ground_truth = ground_truth   # known distance in metres (range mode)
        self.n_samples    = n_samples      # number of scans to collect (range mode)

        self.latest_scan     = None
        self.log_data        = []
        self.range_samples   = []          # min-range readings for statistical analysis
        self.start_time      = None
        self.recording       = False
        self.snapshot_taken  = False

        topic = f"{self.namespace}/scan"
        self.subscription = self.create_subscription(LaserScan, topic, self.scan_callback, 10)

        print(f"\n  LiDAR Logger — Mode: {mode.upper()}")
        print(f"  Subscribing to: {topic}")

        if mode == 'snapshot':
            print(f"  Waiting for a scan...\n")
        elif mode == 'range':
            print(f"  Ground truth distance: {ground_truth} m")
            print(f"  Collecting {n_samples} scans — keep robot stationary")
            print(f"  Measuring forward-facing beam (0° = robot front)\n")
            self.recording = True
        else:
            print(f"  Recording for {duration} seconds — drive the robot\n")
            self.recording = True
            self.start_time = time.time()

    def scan_callback(self, msg):
        self.latest_scan = msg

        if self.mode == 'snapshot' and not self.snapshot_taken:
            self.snapshot_taken = True
            print(f"  Scan received — {len(msg.ranges)} beams, "
                  f"range {msg.range_min:.2f}–{msg.range_max:.2f} m")
            return

        if self.mode == 'range' and self.recording:
            n = len(msg.ranges)
            if n == 0:
                return
            # Forward beam: angle = 0 rad relative to robot front.
            # Compute index from angle_min and angle_increment.
            angle_increment = (msg.angle_max - msg.angle_min) / (n - 1)
            target_angle = math.pi  # radians — robot forward (LiDAR 0° = robot rear on TurtleBot 4)
            raw_idx = int(round((target_angle - msg.angle_min) / angle_increment))
            # Average a ±5 beam window around forward to reduce noise
            window = 5
            lo = max(0, raw_idx - window)
            hi = min(n - 1, raw_idx + window)
            window_ranges = [
                msg.ranges[i] for i in range(lo, hi + 1)
                if msg.range_min <= msg.ranges[i] <= msg.range_max
            ]
            if window_ranges:
                forward_range = sum(window_ranges) / len(window_ranges)
                self.range_samples.append(round(forward_range, 4))
                if len(self.range_samples) >= self.n_samples:
                    self.recording = False
                    print(f"  Collected {self.n_samples} samples.")
            return

        if self.mode == 'log' and self.recording:
            elapsed = time.time() - self.start_time
            if elapsed > self.duration:
                self.recording = False
                return
            valid = [r for r in msg.ranges
                     if msg.range_min <= r <= msg.range_max]
            min_range  = min(valid) if valid else float('nan')
            mean_range = sum(valid) / len(valid) if valid else float('nan')
            self.log_data.append({
                'time':        round(elapsed, 4),
                'min_range':   round(min_range, 4),
                'mean_range':  round(mean_range, 4),
                'valid_beams': len(valid),
            })

    # ── Statistical analysis (range mode) ──────────────────────────────

    def compute_range_stats(self):
        if not self.range_samples:
            return {}
        arr  = np.array(self.range_samples)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        err  = mean - self.ground_truth
        err_pct = (abs(err) / self.ground_truth) * 100.0 if self.ground_truth > 0 else 0.0
        return {
            'ground_truth_m': self.ground_truth,
            'n_samples':      len(self.range_samples),
            'mean_m':         round(mean, 4),
            'std_m':          round(std, 4),
            'error_m':        round(err, 4),
            'error_pct':      round(err_pct, 2),
            'min_reading':    round(float(np.min(arr)), 4),
            'max_reading':    round(float(np.max(arr)), 4),
        }

    def print_range_summary(self, stats):
        print(f"\n  --- LiDAR Range Test Summary ---")
        for k, v in stats.items():
            print(f"  {k:<25} {v}")
        print(f"  {'─'*40}\n")

    def save_range_stats_csv(self, stats, path):
        """Append this trial's stats to a cumulative stats CSV."""
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
        print(f"  Stats appended to: {path}")

    # ── Plots ───────────────────────────────────────────────────────────

    def plot_snapshot(self, path):
        if self.latest_scan is None:
            print("ERROR: No scan received.")
            return
        msg = self.latest_scan
        ranges = np.array(msg.ranges, dtype=float)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        ranges_clean = np.where(
            (ranges >= msg.range_min) & (ranges <= msg.range_max),
            ranges, np.nan)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10))
        fig.suptitle('LiDAR Scan Snapshot — TurtleBot 4\n(180° = forward / camera side)',
                     fontsize=14, fontweight='bold')
        ax.scatter(angles, ranges_clean, s=1.5, c='#E74C3C', alpha=0.7)
        ax.plot(angles, ranges_clean, color='#2980B9', linewidth=0.5, alpha=0.4)
        ax.plot(0, 0, 'ko', markersize=8, label='Robot')
        ax.set_rmax(min(msg.range_max, 6.0))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.grid(True, alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)

        valid = ranges_clean[~np.isnan(ranges_clean)]
        inf_count = int(np.sum(np.isinf(ranges) | (ranges > msg.range_max)))
        fig.text(0.01, 0.01,
                 f"Beams: {len(ranges)}  |  Valid: {len(valid)}  |  "
                 f"No-return: {inf_count}  |  Min: {np.nanmin(ranges_clean):.3f}m  |  "
                 f"Namespace: {self.namespace}",
                 fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Snapshot saved: {path}")

    def plot_range_histogram(self, path):
        if not self.range_samples:
            return
        arr  = np.array(self.range_samples)
        mean = np.mean(arr)
        std  = np.std(arr)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(arr, bins=20, color='#2980B9', edgecolor='white', alpha=0.8)
        ax.axvline(mean, color='#E74C3C', linewidth=2,
                   label=f'Mean: {mean:.4f} m')
        ax.axvline(self.ground_truth, color='#27AE60', linewidth=2,
                   linestyle='--', label=f'Ground truth: {self.ground_truth} m')
        ax.axvline(mean - std, color='gray', linewidth=1, linestyle=':',
                   label=f'±1 std dev: {std:.4f} m')
        ax.axvline(mean + std, color='gray', linewidth=1, linestyle=':')
        ax.set_xlabel('Measured Range (metres)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(
            f'LiDAR Range Accuracy — Ground Truth: {self.ground_truth} m\n'
            f'Mean: {mean:.4f} m  |  Std Dev: {std:.4f} m  |  '
            f'Error: {mean - self.ground_truth:+.4f} m  '
            f'({abs(mean - self.ground_truth)/self.ground_truth*100:.2f}%)',
            fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.text(0.01, 0.01,
                 f"Samples: {len(arr)}  |  Namespace: {self.namespace}",
                 fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Histogram saved: {path}")

    def plot_log(self, path):
        if not self.log_data:
            print("WARNING: No log data.")
            return
        times       = [r['time']        for r in self.log_data]
        min_ranges  = [r['min_range']   for r in self.log_data]
        mean_ranges = [r['mean_range']  for r in self.log_data]
        valid_beams = [r['valid_beams'] for r in self.log_data]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle('LiDAR Data During Motion — TurtleBot 4',
                     fontsize=14, fontweight='bold')

        axes[0].plot(times, min_ranges,  color='#E74C3C', linewidth=1.5,
                     label='Nearest obstacle')
        axes[0].plot(times, mean_ranges, color='#2980B9', linewidth=1.0,
                     linestyle='--', alpha=0.7, label='Mean range')
        axes[0].axhline(y=0.15, color='orange', linestyle=':', linewidth=1.2,
                        label='Min detectable (0.15 m)')
        axes[0].set_ylabel('Distance (metres)', fontsize=11)
        axes[0].set_title('Distance to Nearest Obstacle Over Time', fontsize=12)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

        axes[1].plot(times, valid_beams, color='#27AE60', linewidth=1.2)
        axes[1].set_ylabel('Valid Beam Count', fontsize=11)
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_title('Valid Beam Count Over Time', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        fig.text(0.01, 0.01,
                 f"Scans: {len(self.log_data)}  |  Duration: {times[-1]:.1f}s  |  "
                 f"Min range: {min(min_ranges):.3f}m  |  Namespace: {self.namespace}",
                 fontsize=8, color='gray')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Motion log saved: {path}")

    def save_log_csv(self, path):
        if not self.log_data:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"  Log CSV saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='LiDAR Logger — Test Engineering')
    parser.add_argument('--namespace',    type=str,   required=True,
                        help='Robot namespace e.g. /T10')
    parser.add_argument('--mode',         type=str,
                        choices=['snapshot', 'range', 'log'], default='snapshot')
    parser.add_argument('--ground-truth', type=float, default=1.0,
                        help='Known distance to wall in metres (range mode)')
    parser.add_argument('--samples',      type=int,   default=50,
                        help='Number of scans to collect (range mode)')
    parser.add_argument('--duration',     type=int,   default=30,
                        help='Recording duration in seconds (log mode)')
    parser.add_argument('--trial',        type=int,   default=1,
                        help='Trial number — used in output filenames')
    args = parser.parse_args()

    base       = os.path.expanduser('~')
    stats_path = os.path.join(base, 'lidar_range_stats.csv')

    rclpy.init()
    node = LidarLogger(args.namespace, args.mode, args.duration,
                       args.ground_truth, args.samples)

    try:
        if args.mode == 'snapshot':
            timeout = time.time() + 10
            while rclpy.ok() and not node.snapshot_taken and time.time() < timeout:
                rclpy.spin_once(node, timeout_sec=0.1)

        elif args.mode == 'range':
            timeout = time.time() + 60
            while rclpy.ok() and node.recording and time.time() < timeout:
                rclpy.spin_once(node, timeout_sec=0.05)

        else:
            end_time = time.time() + args.duration + 2
            while rclpy.ok() and time.time() < end_time:
                rclpy.spin_once(node, timeout_sec=0.05)
                if not node.recording and node.log_data:
                    break

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        if args.mode == 'snapshot':
            node.plot_snapshot(os.path.join(base, 'lidar_snapshot.png'))

        elif args.mode == 'range':
            stats = node.compute_range_stats()
            node.print_range_summary(stats)
            node.save_range_stats_csv(stats, stats_path)
            label = f"range_gt{args.ground_truth}m_t{args.trial}"
            node.plot_range_histogram(os.path.join(base, f'lidar_{label}.png'))

        else:
            label = f"log_t{args.trial}"
            node.save_log_csv(os.path.join(base, f'lidar_{label}.csv'))
            node.plot_log(os.path.join(base, f'lidar_{label}.png'))

        node.destroy_node()
        rclpy.shutdown()

    print(f"\n  Done!  Stats file: {stats_path}\n")


if __name__ == '__main__':
    main()
