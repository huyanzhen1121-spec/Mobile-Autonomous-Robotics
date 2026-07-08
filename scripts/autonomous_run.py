#!/usr/bin/env python3
"""
autonomous_run.py  ─  COMPSYS 732  |  TurtleBot 4  |  Demo Phase 2
====================================================================
Package : tb4_sensor_reader
Run     : ~/ros2_venv/bin/python3 -m tb4_sensor_reader.autonomous_run

FSM
───
  SEARCHING  → Lawnmower sweep of C-shaped arena, obstacle avoidance active
  REPORTING  → Cube confirmed: log odometry to terminal, save camera image
  RETURNING  → Proportional controller drives back to (0, 0)
  DONE       → Stop all motion, print final position + elapsed time

Hardware facts (from Task 7 / Task 8 lab docs)
───────────────────────────────────────────────
  Camera   CompressedImage  /TXX/oakd/rgb/image_raw/compressed  ~5 Hz
  LiDAR    LaserScan        /TXX/scan
  Odom     Odometry         /TXX/odom
  CmdVel   Twist            /TXX/cmd_vel

Scoring rubric coverage
───────────────────────
  1 Time              ─ systematic lawnmower finishes well under 10 min
  2 Detection/Pos     ─ terminal log + annotated JPEG saved to SAVE_DIR
  3 Return to Start   ─ P-controller, default tolerance = 0.25 m (5 pts)
  4 Obstacle Avoid    ─ LiDAR 60° front arc, turn toward open side
  5 Autonomy          ─ launch-to-DONE with zero human input
"""

import os
import math
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, CompressedImage
from cv_bridge import CvBridge


# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  edit only this section to match your robot and arena
# ══════════════════════════════════════════════════════════════════════════

NAMESPACE = '/T6'       # ← change to your robot number  (e.g. '/T3', '/T12')

# ── Lawnmower search pattern ─────────────────────────────────────────────
FORWARD_SPEED = 0.15    # m/s  (keep conservative on physical robot)
TURN_SPEED    = 0.50    # rad/s
LANE_LENGTH   = 1.80    # metres per forward sweep (≈ C-corridor depth)
LANE_SPACING  = 0.30    # metres between lanes  (must be < camera FOV width)
NUM_LANES     = 6       # how many lanes to sweep before giving up

# ── Obstacle avoidance (Task 7 pattern) ──────────────────────────────────
AVOID_DIST    = 0.55    # metres  (larger than sim – physical inertia)
FRONT_ARC_DEG = 60      # degrees each side of forward axis to check

# ── Red cube HSV thresholds (from Task 8 / Investigation C) ──────────────
# Red wraps around hue 0/180 → two ranges needed
RED_LOW1  = np.array([  0, 120, 70], dtype=np.uint8)
RED_HIGH1 = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LOW2  = np.array([170, 120, 70], dtype=np.uint8)
RED_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)
MIN_PIXELS = 500        # calibrate using your Task 8 results

# ── Return-to-start gains ─────────────────────────────────────────────────
KP_LIN         = 0.50   # proportional gain – linear velocity
KP_ANG         = 1.50   # proportional gain – angular velocity
GOAL_TOLERANCE = 0.25   # metres  (5-point scoring band)

# ── Image save directory (path printed to terminal for report) ────────────
SAVE_DIR = os.path.expanduser('~/ros2_ws/detection_output')


# ══════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════

def norm_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a >  math.pi:  a -= 2 * math.pi
    while a < -math.pi:  a += 2 * math.pi
    return a


# ══════════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════════

class AutonomousRun(Node):
    """
    Single-node autonomous run:
      camera detection  +  LiDAR avoidance  +  odometry return-to-start.
    """

    def __init__(self):
        super().__init__('autonomous_run')
        os.makedirs(SAVE_DIR, exist_ok=True)

        self.bridge     = CvBridge()
        self.start_time = time.time()

        # ── Publishers ──────────────────────────────────────────────────
        self.pub = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(
            Odometry,
            f'{NAMESPACE}/odom',
            self._cb_odom, 10)

        self.create_subscription(
            LaserScan,
            f'{NAMESPACE}/scan',
            self._cb_scan, 10)

        # Physical robot publishes CompressedImage (JPEG over WiFi, ~5 Hz)
        self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self._cb_image, 10)

        # ── Odometry state ──────────────────────────────────────────────
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # ── LiDAR state ─────────────────────────────────────────────────
        self.near_front = float('inf')
        self.near_left  = float('inf')
        self.near_right = float('inf')

        # ── Detection state ─────────────────────────────────────────────
        self.detected  = False
        self.det_x     = 0.0
        self.det_y     = 0.0
        self.img_saved = False

        # ── FSM state ───────────────────────────────────────────────────
        self.state = 'SEARCHING'

        # ── Lawnmower sub-state ─────────────────────────────────────────
        # phase:  FORWARD → TURN1 → LATERAL → TURN2 → (next lane)
        self.lane        = 0
        self.sweep_phase = 'FORWARD'
        self.sweep_dir   = 1          # +1 or -1 along corridor axis
        self.seg_start_x = 0.0
        self.seg_start_y = 0.0
        self.target_yaw  = 0.0

        # ── 10 Hz control timer ─────────────────────────────────────────
        self.timer = self.create_timer(0.1, self._loop)

        self.get_logger().info(
            f'[INIT] Autonomous node started  |  '
            f'namespace={NAMESPACE}  |  state=SEARCHING')

    # ══════════════════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════════════════

    def _cb_odom(self, msg: Odometry):
        """Cache robot pose from odometry (Task 6 / Task 7 pattern)."""
        p = msg.pose.pose
        self.x = p.position.x
        self.y = p.position.y
        q = p.orientation
        # Quaternion → yaw  (Task 6 pose_reader pattern)
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y  + q.z * q.z))

    def _cb_scan(self, msg: LaserScan):
        """
        Compute nearest obstacle in front / left / right arcs.
        Index arithmetic matches Task 7 reactive_physical.py exactly.
        """
        inc     = msg.angle_increment
        n       = len(msg.ranges)
        # Forward beam index (Task 7: front_i = int(round(-angle_min / inc)))
        front_i = int(round(-msg.angle_min / inc))
        half_f  = int(round(math.radians(FRONT_ARC_DEG) / inc))
        half_s  = int(round(math.radians(90.0)          / inc))

        def _arc_min(lo: int, hi: int) -> float:
            lo = max(0, lo); hi = min(n - 1, hi)
            vals = [r for r in msg.ranges[lo : hi + 1]
                    if msg.range_min < r < msg.range_max]
            return min(vals) if vals else float('inf')

        self.near_front = _arc_min(front_i - half_f, front_i + half_f)
        self.near_left  = _arc_min(front_i,           front_i + half_s)
        self.near_right = _arc_min(front_i - half_s,  front_i)

    def _cb_image(self, msg: CompressedImage):
        """
        HSV red-cube detection using CompressedImage + cv_bridge.
        Follows Task 8 camera_detector.py / detect_and_stop.py pattern.

        Key differences from simulation:
          • Topic  : CompressedImage (not raw Image)
          • Decoder: compressed_imgmsg_to_cv2  (not imgmsg_to_cv2)
          • cv2.waitKey(1) is mandatory after imshow
        """
        if self.state != 'SEARCHING':
            return   # skip camera processing once cube is found

        # Decode CompressedImage → BGR numpy array  (Task 8 standard)
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'[IMAGE] Decode error: {e}')
            return

        # HSV thresholding (same two-range approach as Task 8)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))

        # Morphological denoising (reduces false positives)
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        pixels = cv2.countNonZero(mask)

        if pixels >= MIN_PIXELS and not self.detected:
            # ── Capture detection position (odom) ──────────────────────
            self.detected = True
            self.det_x    = self.x
            self.det_y    = self.y

            # ── Save annotated image (Rubric 2 evidence) ───────────────
            if not self.img_saved:
                annotated = img.copy()
                annotated[mask > 0] = [0, 0, 255]   # highlight red region
                ts       = time.strftime('%Y%m%d_%H%M%S')
                path     = os.path.join(SAVE_DIR, f'cube_detection_{ts}.jpg')
                cv2.imwrite(path, annotated)
                self.img_saved = True
                # Terminal output required for report
                self.get_logger().info(
                    f'[DETECTION] Image saved → {path}')

            # Terminal output required for Rubric 2 (odometry at detection)
            self.get_logger().info(
                f'[DETECTION] RED CUBE CONFIRMED  '
                f'pixels={pixels}  '
                f'odometry  x={self.det_x:.3f} m  y={self.det_y:.3f} m')

        # ── Debug overlay window ───────────────────────────────────────
        # Shows pixel count and current state; useful during testing
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay,
                    f'px={pixels}  state={self.state}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if pixels >= MIN_PIXELS:
            cv2.putText(overlay, 'DETECTED',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)   # ← mandatory: prevents window from freezing (Task 8)

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop  (10 Hz)
    # ══════════════════════════════════════════════════════════════════════

    def _loop(self):
        elapsed = time.time() - self.start_time
        cmd     = Twist()

        if self.state == 'SEARCHING':
            if self.detected:
                self._publish_stop()
                self.state = 'REPORTING'
                return
            cmd = self._lawnmower()

        elif self.state == 'REPORTING':
            self._publish_stop()
            # Required terminal output (Rubric 2 + report S3)
            self.get_logger().info(
                f'[REPORTING] '
                f'Detection position: '
                f'x={self.det_x:.3f} m  y={self.det_y:.3f} m  '
                f'| elapsed={elapsed:.1f} s')
            self.state = 'RETURNING'
            self.get_logger().info('[STATE] REPORTING → RETURNING')
            return

        elif self.state == 'RETURNING':
            cmd = self._return_to_start(elapsed)

        elif self.state == 'DONE':
            self._publish_stop()
            return

        self.pub.publish(cmd)

    # ══════════════════════════════════════════════════════════════════════
    #  Lawnmower sweep  (systematic coverage of C-shaped arena)
    # ══════════════════════════════════════════════════════════════════════

    def _lawnmower(self) -> Twist:
        """
        Four-phase lawnmower:
          FORWARD  → drive LANE_LENGTH along corridor axis
          TURN1    → rotate 90° toward next lane
          LATERAL  → advance LANE_SPACING perpendicular to corridor
          TURN2    → rotate 90° back to corridor axis (opposite direction)
          repeat with reversed sweep_dir
        """
        cmd = Twist()

        if self.lane >= NUM_LANES:
            self.get_logger().warn(
                '[SEARCH] All lanes swept — cube not found. Stopping.')
            self.state = 'DONE'
            return cmd

        # ── Obstacle avoidance overrides lawnmower during FORWARD ───────
        if (self.sweep_phase == 'FORWARD'
                and self.near_front <= AVOID_DIST):
            cmd.linear.x  = 0.0
            cmd.angular.z = (TURN_SPEED
                             if self.near_left >= self.near_right
                             else -TURN_SPEED)
            self.get_logger().warn(
                f'[AVOID] front={self.near_front:.2f} m  '
                f'turning {"LEFT" if cmd.angular.z > 0 else "RIGHT"}')
            return cmd

        # ── FORWARD phase ────────────────────────────────────────────────
        if self.sweep_phase == 'FORWARD':
            driven = math.hypot(self.x - self.seg_start_x,
                                self.y - self.seg_start_y)
            if driven < LANE_LENGTH:
                cmd.linear.x = FORWARD_SPEED * self.sweep_dir
            else:
                # Lane complete → start 90° turn toward next lane
                sign = 1 if self.sweep_dir > 0 else -1
                self.target_yaw  = norm_angle(
                    self.yaw + sign * math.pi / 2.0)
                self.sweep_phase = 'TURN1'

        # ── TURN1 phase (first 90° turn) ─────────────────────────────────
        elif self.sweep_phase == 'TURN1':
            err = norm_angle(self.target_yaw - self.yaw)
            if abs(err) > 0.06:   # ~3.4° deadband
                cmd.angular.z = math.copysign(TURN_SPEED, err)
            else:
                # Aligned → drive one lane-width laterally
                self.seg_start_x = self.x
                self.seg_start_y = self.y
                self.sweep_phase = 'LATERAL'

        # ── LATERAL phase (step to next lane) ────────────────────────────
        elif self.sweep_phase == 'LATERAL':
            driven = math.hypot(self.x - self.seg_start_x,
                                self.y - self.seg_start_y)
            if driven < LANE_SPACING:
                cmd.linear.x = FORWARD_SPEED
            else:
                # Second 90° turn back to corridor direction (reversed)
                sign = 1 if self.sweep_dir > 0 else -1
                self.target_yaw  = norm_angle(
                    self.yaw + sign * math.pi / 2.0)
                self.sweep_phase = 'TURN2'

        # ── TURN2 phase (second 90° turn) ────────────────────────────────
        elif self.sweep_phase == 'TURN2':
            err = norm_angle(self.target_yaw - self.yaw)
            if abs(err) > 0.06:
                cmd.angular.z = math.copysign(TURN_SPEED, err)
            else:
                # Next lane ready: reverse sweep direction
                self.sweep_dir   = -self.sweep_dir
                self.seg_start_x = self.x
                self.seg_start_y = self.y
                self.sweep_phase = 'FORWARD'
                self.lane       += 1
                self.get_logger().info(
                    f'[SEARCH] Starting lane {self.lane}/{NUM_LANES}  '
                    f'pos=({self.x:.2f},{self.y:.2f})')

        return cmd

    # ══════════════════════════════════════════════════════════════════════
    #  Return to start  (proportional controller)
    # ══════════════════════════════════════════════════════════════════════

    def _return_to_start(self, elapsed: float) -> Twist:
        """
        Proportional heading + linear controller targeting (0, 0).
        Matches the detect_and_stop.py pattern from Task 8.
        """
        cmd = Twist()
        dx   = -self.x
        dy   = -self.y
        dist = math.hypot(dx, dy)

        if dist < GOAL_TOLERANCE:
            self.get_logger().info(
                f'[DONE] Origin reached.  '
                f'final_pos=({self.x:.3f}, {self.y:.3f}) m  '
                f'error={dist:.3f} m  '
                f'elapsed={elapsed:.1f} s')
            self.state = 'DONE'
            self._publish_stop()
            return cmd

        goal_heading = math.atan2(dy, dx)
        heading_err  = norm_angle(goal_heading - self.yaw)

        # Rotate in place if not aligned; combine forward + steer once aligned
        if abs(heading_err) > 0.25:
            cmd.angular.z = KP_ANG * heading_err
        else:
            cmd.linear.x  = min(KP_LIN * dist, FORWARD_SPEED)
            cmd.angular.z = KP_ANG * heading_err

        self.get_logger().info(
            f'[RETURN] dist={dist:.2f} m  '
            f'heading_err={math.degrees(heading_err):.1f}°  '
            f'pos=({self.x:.2f},{self.y:.2f})')
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _publish_stop(self):
        self.pub.publish(Twist())


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousRun()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        cv2.destroyAllWindows()   # clean up imshow windows
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
