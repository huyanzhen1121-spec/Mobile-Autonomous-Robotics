import math
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# ─── Configuration ────────────────────────────────────────────────────────────
NAMESPACE      = ''  # Change to your robot namespace
FORWARD_SPEED  = 0.15    # m/s
TURN_SPEED     = 0.5     # rad/s
AVOID_DISTANCE = 0.45    # metres — obstacle avoidance threshold

WALL_DISTANCE  = 0.35    # metres — target distance from right wall
WALL_KP        = 1.2     # proportional gain for wall following

APPROACH_SPEED    = 0.08    # m/s — slow creep toward cube
APPROACH_KP_ANG   = 0.005   # proportional gain: pixel offset → angular.z
APPROACH_STOP_DIST  = 0.40   # metres — distance threshold to stop approaching
APPROACH_STOP_PIX   = 10000  # pixels — pixel count threshold to stop approaching

STUCK_TIME          = 15.0    # seconds without movement before escape
STUCK_DIST_THRESH   = 0.05   # metres — less than this = considered not moving
ESCAPE_ANGLE        = 135    # degrees to turn when stuck

RETURN_KP_ANG  = 1.5     # proportional gain for return heading
RETURN_KP_LIN  = 0.6     # proportional gain for return speed
RETURN_MAX_SPD = 0.18    # m/s — max speed while returning
RETURN_THRESH  = 0.20    # metres — close enough to (0,0)

RED_LOW1  = np.array([0,   150, 100])
RED_HIGH1 = np.array([8,   255, 255])
RED_LOW2  = np.array([172, 150, 100])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 2000


# ─── Node ─────────────────────────────────────────────────────────────────────
class DetectAndStop(Node):

    def __init__(self):
        super().__init__('detect_and_stop')

        self.bridge       = CvBridge()
        self.last_overlay = None
        self.img_width    = 640       # updated on first frame

        # Approach tracking
        self.cube_cx      = None      # pixel x of cube centroid
        self.cube_pixels  = 0

        self.pub = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10
        )
        self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan', self.scan_callback, 10
        )
        self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10
        )
        self.create_subscription(
            Odometry,
            f'{NAMESPACE}/odom',
            self.odom_callback, 10
        )

        # Laser distances
        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')

        # Odometry
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # State machine
        self.cube_detected   = False
        self.state           = 'SEARCHING'
        self.turn_target_yaw = None   # target yaw for 180° spin

        # Stuck detection
        self.last_moved_x    = 0.0
        self.last_moved_y    = 0.0
        self.last_moved_time = self.get_clock().now()
        self.prev_state      = None   # state before ESCAPING
        self.escape_target_yaw = None

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Detect-and-stop node started — SEARCHING')

    # ── Laser scan callback ───────────────────────────────────────────────────
    def scan_callback(self, msg: LaserScan):
        inc    = msg.angle_increment
        arc_r  = math.radians(60)
        side_r = math.radians(90)

        front_r = math.radians(90)
        front_i = int(round(front_r / inc))
        half_a  = int(round(arc_r  / inc))
        side_a  = int(round(side_r / inc))
        n       = len(msg.ranges)

        def arc_min(lo, hi):
            lo = max(0, lo); hi = min(n - 1, hi)
            vals = [r for r in msg.ranges[lo:hi + 1]
                    if msg.range_min < r < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_left  = arc_min(front_i,          front_i + side_a)
        self.nearest_right = arc_min(front_i - side_a, front_i)

    # ── Odometry callback ─────────────────────────────────────────────────────
    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    # ── Camera callback ───────────────────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        img  = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]
        self.img_width = w

        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        )
        pixels = cv2.countNonZero(mask)
        self.cube_pixels = pixels

        # Find centroid of largest red contour for approach steering
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cube_cx = None
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= MIN_PIXELS:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    self.cube_cx = int(M['m10'] / M['m00'])
                x, y, bw, bh = cv2.boundingRect(largest)
                cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                if self.cube_cx is not None:
                    cv2.circle(overlay, (self.cube_cx, y + bh // 2), 6,
                               (0, 255, 0), -1)

        # HUD text
        state_colour = (0, 255, 0) if self.state == 'SEARCHING' else (0, 0, 255)
        cv2.putText(overlay, f'State: {self.state}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_colour, 2)
        cv2.putText(overlay, f'Red pixels: {pixels}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay,
                    f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}) m',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay,
                    f'Front: {self.nearest_front:.2f} m',
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if pixels >= MIN_PIXELS:
            cv2.putText(overlay, 'DETECTED', (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        self.last_overlay = overlay

        if self.state == 'SEARCHING' and pixels >= MIN_PIXELS:
            self.cube_detected = True

    # ── Stop helper ───────────────────────────────────────────────────────────
    def stop(self):
        self.pub.publish(Twist())

    # ── Wall-following command ─────────────────────────────────────────────────
    def wall_follow_cmd(self) -> Twist:
        cmd = Twist()
        error = self.nearest_right - WALL_DISTANCE
        cmd.linear.x  = FORWARD_SPEED
        cmd.angular.z = max(-TURN_SPEED,
                            min(TURN_SPEED, -WALL_KP * error))
        return cmd

    # ── Approach command ───────────────────────────────────────────────────────
    def approach_cmd(self) -> Twist:
        """
        Creep toward the cube using visual centroid for steering.
        Returns None if the cube is no longer visible.
        """
        cmd = Twist()

        if self.cube_cx is None:
            # Lost sight — spin slowly to re-acquire
            cmd.linear.x  = 0.0
            cmd.angular.z = TURN_SPEED * 0.4
            self.get_logger().warn('Approach: cube lost — spinning to re-acquire')
            return cmd

        # Pixel offset from image centre → angular correction
        offset = self.cube_cx - (self.img_width / 2)
        cmd.angular.z = -APPROACH_KP_ANG * offset
        cmd.angular.z = max(-TURN_SPEED * 0.6,
                            min(TURN_SPEED * 0.6, cmd.angular.z))
        cmd.linear.x  = APPROACH_SPEED
        return cmd

    # ── Obstacle-aware return command ─────────────────────────────────────────
    def return_cmd(self) -> Twist:
        cmd = Twist()
        dx = 0.0 - self.current_x
        dy = 0.0 - self.current_y
        target_yaw  = math.atan2(dy, dx)
        heading_err = target_yaw - self.current_yaw
        while heading_err >  math.pi: heading_err -= 2 * math.pi
        while heading_err < -math.pi: heading_err += 2 * math.pi
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if self.nearest_front <= AVOID_DISTANCE:
            # Obstacle blocking — turn away, always keep moving forward slightly
            # so robot doesn't get stuck spinning in place
            cmd.linear.x = 0.05
            if self.nearest_left >= self.nearest_right:
                cmd.angular.z = TURN_SPEED
                self.get_logger().warn('Return: obstacle — turning LEFT')
            else:
                cmd.angular.z = -TURN_SPEED
                self.get_logger().warn('Return: obstacle — turning RIGHT')
        else:
            # Phase 1: large heading error → turn in place first
            if abs(heading_err) > math.radians(30):
                cmd.linear.x  = 0.0
                cmd.angular.z = max(-TURN_SPEED,
                                    min(TURN_SPEED, RETURN_KP_ANG * heading_err))
            # Phase 2: roughly aligned → drive forward with gentle correction
            else:
                cmd.linear.x  = min(RETURN_MAX_SPD, RETURN_KP_LIN * dist)
                cmd.angular.z = max(-TURN_SPEED * 0.5,
                                    min(TURN_SPEED * 0.5,
                                        RETURN_KP_ANG * heading_err))
        return cmd

    # ── Stuck detection ───────────────────────────────────────────────────────
    def check_stuck(self):
        """
        Call once per control loop tick (only in SEARCHING/RETURNING).
        If robot hasn't moved STUCK_DIST_THRESH in STUCK_TIME seconds,
        trigger ESCAPING state which turns 135° then resumes.
        """
        dist_moved = math.sqrt(
            (self.current_x - self.last_moved_x) ** 2 +
            (self.current_y - self.last_moved_y) ** 2
        )
        now = self.get_clock().now()

        if dist_moved >= STUCK_DIST_THRESH:
            # Made progress — reset reference point and timer
            self.last_moved_x    = self.current_x
            self.last_moved_y    = self.current_y
            self.last_moved_time = now
        else:
            elapsed = (now - self.last_moved_time).nanoseconds / 1e9
            if elapsed >= STUCK_TIME:
                self.get_logger().warn(
                    f'STUCK detected ({elapsed:.1f}s without movement) — escaping'
                )
                self.prev_state = self.state
                target = self.current_yaw + math.radians(ESCAPE_ANGLE)
                if target > math.pi:
                    target -= 2 * math.pi
                self.escape_target_yaw = target
                self.state = 'ESCAPING'
                # Reset timer so we don't re-trigger immediately after escape
                self.last_moved_time = now
                self.last_moved_x    = self.current_x
                self.last_moved_y    = self.current_y

    # ── Control loop (10 Hz) ──────────────────────────────────────────────────
    def control_loop(self):
        if self.last_overlay is not None:
            cv2.imshow('Detection', self.last_overlay)
            cv2.waitKey(1)

        if self.state == 'DONE':
            return

        # ── SEARCHING ─────────────────────────────────────────────────────────
        if self.state == 'SEARCHING':
            self.check_stuck()
            if self.state == 'ESCAPING':
                return

            if self.cube_detected:
                self.state = 'APPROACHING'
                self.get_logger().info('Cube detected — switching to APPROACHING')
                return

            if self.nearest_front <= AVOID_DISTANCE:
                cmd = Twist()
                cmd.linear.x = 0.0
                if self.nearest_left >= self.nearest_right:
                    cmd.angular.z = TURN_SPEED
                    self.get_logger().warn('Obstacle — turning LEFT')
                else:
                    cmd.angular.z = -TURN_SPEED
                    self.get_logger().warn('Obstacle — turning RIGHT')
            else:
                cmd = self.wall_follow_cmd()
                self.get_logger().info(
                    f'Wall-follow | right={self.nearest_right:.2f} m '
                    f'front={self.nearest_front:.2f} m '
                    f'pos=({self.current_x:.2f}, {self.current_y:.2f})'
                )
            self.pub.publish(cmd)

        # ── APPROACHING ───────────────────────────────────────────────────────
        elif self.state == 'APPROACHING':
            # Arrived: pixels > 10000 AND front distance < 0.4 m
            if (self.cube_pixels >= APPROACH_STOP_PIX and
                    self.nearest_front <= APPROACH_STOP_DIST):
                self.stop()
                self.state = 'DETECTED'
                self.get_logger().info(
                    f'Arrived at cube — front={self.nearest_front:.2f} m')
                self.get_logger().info(
                    f'Detected position: x={self.current_x:.3f} m  '
                    f'y={self.current_y:.3f} m'
                )
                ts = self.get_clock().now().to_msg().sec
                if self.last_overlay is not None:
                    cv2.imwrite(f'/home/ubuntu/detection_{ts}.jpg',
                                self.last_overlay)
                    self.get_logger().info(f'Image saved: detection_{ts}.jpg')
                return

            # Lost the cube for too long — fall back to searching
            if self.cube_pixels < MIN_PIXELS // 2:
                self.cube_detected = False
                self.state = 'SEARCHING'
                self.get_logger().warn('Cube lost during approach — back to SEARCHING')
                return

            cmd = self.approach_cmd()
            self.get_logger().info(
                f'Approaching | front={self.nearest_front:.2f} m '
                f'pixels={self.cube_pixels} '
                f'cx={self.cube_cx} '
                f'pos=({self.current_x:.2f}, {self.current_y:.2f})'
            )
            self.pub.publish(cmd)

        # ── DETECTED: record then start 180° turn ─────────────────────────────
        elif self.state == 'DETECTED':
            self.stop()
            # Set target yaw = current yaw + 180°, normalised to [-π, π]
            target = self.current_yaw + math.pi
            if target > math.pi:
                target -= 2 * math.pi
            self.turn_target_yaw = target
            self.state = 'TURNING'
            self.get_logger().info(
                f'Starting 180° turn — '
                f'from yaw={math.degrees(self.current_yaw):.1f}° '
                f'to {math.degrees(target):.1f}°'
            )

        # ── TURNING: spin 180° in place before returning ───────────────────────
        elif self.state == 'TURNING':
            err = self.turn_target_yaw - self.current_yaw
            # Normalise to [-π, π]
            while err >  math.pi: err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi

            if abs(err) < math.radians(8):   # within 8° — close enough
                self.stop()
                self.state = 'RETURNING'
                self.get_logger().info('180° turn complete — switching to RETURNING')
            else:
                cmd = Twist()
                cmd.linear.x  = 0.0
                # Turn direction follows sign of error
                cmd.angular.z = TURN_SPEED if err > 0 else -TURN_SPEED
                self.pub.publish(cmd)
                self.get_logger().info(
                    f'Turning | err={math.degrees(err):.1f}°'
                )

        # ── RETURNING ─────────────────────────────────────────────────────────
        elif self.state == 'RETURNING':
            self.check_stuck()
            if self.state == 'ESCAPING':
                return

            dx   = 0.0 - self.current_x
            dy   = 0.0 - self.current_y
            dist = math.sqrt(dx ** 2 + dy ** 2)

            if dist < RETURN_THRESH:
                self.stop()
                self.state = 'DONE'
                self.get_logger().info(
                    f'Arrived at origin — final pos '
                    f'({self.current_x:.3f}, {self.current_y:.3f}), '
                    f'error={dist:.3f} m'
                )
                return

            cmd = self.return_cmd()
            self.get_logger().info(
                f'Returning | dist={dist:.2f} m '
                f'front={self.nearest_front:.2f} m '
                f'pos=({self.current_x:.2f}, {self.current_y:.2f})'
            )
            self.pub.publish(cmd)

        # ── ESCAPING: turn 135° then resume previous state ────────────────────
        elif self.state == 'ESCAPING':
            err = self.escape_target_yaw - self.current_yaw
            while err >  math.pi: err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi

            if abs(err) < math.radians(8):
                self.stop()
                self.state = self.prev_state
                self.get_logger().info(
                    f'Escape complete — resuming {self.prev_state}'
                )
            else:
                cmd = Twist()
                cmd.linear.x  = 0.0
                cmd.angular.z = TURN_SPEED if err > 0 else -TURN_SPEED
                self.pub.publish(cmd)
                self.get_logger().info(
                    f'Escaping | err={math.degrees(err):.1f}°'
                )


# ─── Entry point ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = DetectAndStop()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
