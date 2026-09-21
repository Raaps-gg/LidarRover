import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import serial
import math

class RoverMasterNode(Node):
    def __init__(self):
        super().__init__('rover_master_node')

        # Robot & Serial Parameters
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_base', 0.81)      # account tire scrub 0.47(base) * 1.861(adjust)
        self.declare_parameter('wheel_radius', 0.127)
        self.declare_parameter('ticks_per_rev', 90)

        # PS5 Controller Parameters
        self.declare_parameter('axis_linear', 1)        # Left stick up/down
        self.declare_parameter('axis_angular', 0)       # Left stick left/right
        self.declare_parameter('modeSwitch_button', 0)  # button X
        self.declare_parameter('speed_linear', 0.8)     # m/s
        self.declare_parameter('speed_angular', 3.5)    # rad/s

        self.manual_mode = True
        self.prev_button_state = 0
        self.port = self.get_parameter('port').value    # using ROS2 parameter for expandability
        self.baud = self.get_parameter('baudrate').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.ticks_per_rev = self.get_parameter('ticks_per_rev').value

        # Open Serial to Arduino Due
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.02)
            self.get_logger().info(f"Connected to Arduino Due on {self.port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open {self.port}: {e}")
            raise e

        # Velocity State (Multiplexer)
        self.target_vx = 0.0
        self.target_wz = 0.0
        self.last_nav_time = self.get_clock().now()
        self.nav_vx = 0.0
        self.nav_wz = 0.0
        self.joy_active = False

        # Odometry Tracking State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_ticks = None
        self.prev_right_ticks = None
        self.prev_odom_time = self.get_clock().now()

        # Subscribers & Publishers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.nav_sub = self.create_subscription(Twist, '/cmd_vel', self.nav_callback, 10)
        self.active_cmd_pub = self.create_publisher(Twist, '/rover/active_cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Main loop timer (20 Hz / 50 ms)
        self.create_timer(0.05, self.control_and_serial_loop)

    def joy_callback(self, msg: Joy):
        toggle_idx = self.get_parameter('modeSwitch_button').value

        # 1. Detect X Button Press (Rising Edge: 0 -> 1)
        if len(msg.buttons) > toggle_idx:
            current_btn = msg.buttons[toggle_idx]
            if current_btn == 1 and self.prev_button_state == 0:
                self.manual_mode = not self.manual_mode
                mode_str = "MANUAL" if self.manual_mode else "NAV2 / AUTO"
                self.get_logger().info(f"Switched control mode to: {mode_str}")
            self.prev_button_state = current_btn

        # 2. In Manual Mode, map analog stick to velocity
        if self.manual_mode:
            axis_lin = self.get_parameter('axis_linear').value
            axis_ang = self.get_parameter('axis_angular').value
            speed_lin = self.get_parameter('speed_linear').value
            speed_ang = self.get_parameter('speed_angular').value

            if len(msg.axes) > max(axis_lin, axis_ang):
                self.target_vx = msg.axes[axis_lin] * speed_lin
                self.target_wz = msg.axes[axis_ang] * speed_ang

    def nav_callback(self, msg: Twist):
        self.last_nav_time = self.get_clock().now()

        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Open-loop skid-steer scrub breakaway compensation
        MIN_SCRUB_RAD = 1.4  # Minimum angular speed to physically break tire scrub

        is_pivoting_in_place = abs(linear_x) < 0.05
        
        if is_pivoting_in_place:
        # 1.
            if abs(angular_z) <= 0.20:
                angular_z = 0.0
        # 2.
            elif abs(angular_z) < MIN_SCRUB_RAD:
                angular_z = math.copysign(MIN_SCRUB_RAD, angular_z)

        self.nav_vx = linear_x
        self.nav_wz = angular_z

    def control_and_serial_loop(self):
        now = self.get_clock().now()

        # 1. Check Mode: If in Auto Mode, read from Nav2
        if not self.manual_mode:
            # Fall back to Nav2 cmd_vel if received within 0.5s watchdog window
            if (now - self.last_nav_time).nanoseconds / 1e9 < 0.5:
                self.target_vx = self.nav_vx
                self.target_wz = self.nav_wz
            else:
                self.target_vx = 0.0
                self.target_wz = 0.0

        # Publish active velocity for Foxglove/visualization
        active_twist = Twist()
        active_twist.linear.x = float(self.target_vx)
        active_twist.angular.z = float(self.target_wz)
        self.active_cmd_pub.publish(active_twist)

        # Transmit velocity command to Arduino Due ('v <linear_x> <angular_z>\n')
        if self.ser.is_open:
            cmd = f"v {self.target_vx:.3f} {self.target_wz:.3f}\n"
            self.ser.write(cmd.encode('utf-8'))

        # 2. Read Telemetry from Arduino Due ('e <left> <right>\n')
        while self.ser.is_open and self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 3 and parts[0] == 'e':
                    left_ticks = int(parts[1])
                    right_ticks = int(parts[2])
                    self.update_odometry(left_ticks, right_ticks)
            except (ValueError, UnicodeDecodeError):
                continue

    def update_odometry(self, left_ticks, right_ticks):
        current_time = self.get_clock().now()

        if self.prev_left_ticks is None:
            self.prev_left_ticks = left_ticks
            self.prev_right_ticks = right_ticks
            self.prev_odom_time = current_time
            return

        dt = (current_time - self.prev_odom_time).nanoseconds / 1e9
        if dt <= 0:
            return

        d_left_ticks = (left_ticks - self.prev_left_ticks)
        d_right_ticks = (right_ticks - self.prev_right_ticks)

        d_left = (d_left_ticks / float(self.ticks_per_rev)) * (2.0 * math.pi * self.wheel_radius)
        d_right = (d_right_ticks / float(self.ticks_per_rev)) * (2.0 * math.pi * self.wheel_radius)

        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks
        self.prev_odom_time = current_time

        d_dist = (d_right + d_left) / 2.0
        d_theta = (d_right - d_left) / self.wheel_base

        if d_dist != 0.0:
            self.x += d_dist * math.cos(self.theta + (d_theta / 2.0))
            self.y += d_dist * math.sin(self.theta + (d_theta / 2.0))
        self.theta += d_theta

        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        # Broadcast odom -> base_link Transform
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

        # Publish /odom topic
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = d_dist / dt
        odom.twist.twist.angular.z = d_theta / dt
        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = RoverMasterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
