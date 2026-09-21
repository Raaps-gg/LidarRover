import struct
import math
import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from foo import RoverOdometry, convert_to_servo, cmd_vel_to_servo

class RoverBaseNode(Node):
    def __init__(self):
        super().__init__('rover_base_node')

        # 1. Initialize Math Tracker from foo.py
        self.odom_tracker = RoverOdometry(wheel_radius=0.12, wheel_base=0.47, ticks_per_rev=90)

        # 2. ROS 2 Publishers & Broadcasters
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)	#10 outgoing msg
        self.tf_broadcaster = TransformBroadcaster(self)

        # 3. ROS 2 Subscriptions (Control Inputs)
        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.sub_cmd_vel = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # 4. SINGLE Serial Connection
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0)
            self.get_logger().info("Connected to Arduino over Serial.")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.ser = None

        # System State
        self.mode = 0  # 0 = Manual (Joy) | 1 = Autonomous (Nav2)
        self.manual_left = 90
        self.manual_right = 90
        self.nav_left = 90
        self.nav_right = 90
        self.last_button_state = 0

        # Command Watchdog Timeout (0.5 seconds)
        self.last_cmd_vel_time = self.get_clock().now()
        self.cmd_vel_timeout = 0.5

        # Buffer for incoming encoder bytes
        self.buffer = bytearray()

        # Main Loop Timer (50 Hz / 0.02s)
        self.timer = self.create_timer(0.02, self.control_and_read_loop)

    def joy_callback(self, msg):
        button_state = msg.buttons[0]
        if button_state == 1 and self.last_button_state == 0:
            self.mode = 1 if self.mode == 0 else 0
            mode_name = "AUTONOMOUS (Nav2)" if self.mode == 1 else "MANUAL (Joystick)"
            self.get_logger().info(f"Mode Switched -> {mode_name}")

        self.last_button_state = button_state

        steering, throttle = msg.axes[0], msg.axes[1]
        self.manual_left, self.manual_right = convert_to_servo(throttle, steering)

    def cmd_vel_callback(self, msg):
        self.nav_left, self.nav_right = cmd_vel_to_servo(msg.linear.x, msg.angular.z)
        self.last_cmd_vel_time = self.get_clock().now()

    def control_and_read_loop(self):
        if not self.ser or not self.ser.is_open:
            return

        # Watchdog: Reset autonomous commands to neutral if messages stop
        time_since_cmd = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds / 1e9
        if self.mode == 1 and time_since_cmd > self.cmd_vel_timeout:
            self.nav_left = 90
            self.nav_right = 90
        # ==========================================
        # STEP A: Send Motor Packet to Arduino (TX)
        # ==========================================
        left = self.manual_left if self.mode == 0 else self.nav_left
        right = self.manual_right if self.mode == 0 else self.nav_right
        
        # Write 3-byte command packet [0xFF, Left, Right]
        self.ser.write(bytes([0xFF, left, right]))

        # ==========================================
        # STEP B: Read Encoder Feedback from Arduino (RX)
        # ==========================================
        if self.ser.in_waiting > 0:
            self.buffer.extend(self.ser.read(self.ser.in_waiting))

        if len(self.buffer) > 200:
            self.buffer.clear()
            return

        while len(self.buffer) >= 9:
            if self.buffer[0] == 0xFF:
                data_bytes = self.buffer[1:9]
                left_ticks, right_ticks = struct.unpack('<ii', data_bytes)
                del self.buffer[:9]

                now = self.get_clock().now()
                time_sec = now.nanoseconds / 1e9

                # Odometry Math
                x, y, theta, v, w = self.odom_tracker.update(left_ticks, right_ticks, time_sec)
                
                qz = math.sin(theta / 2.0)
                qw = math.cos(theta / 2.0)

                # Broadcast TF
                t = TransformStamped()
                t.header.stamp = now.to_msg()
                t.header.frame_id = 'odom'
                t.child_frame_id = 'base_link'
                t.transform.translation.x = x
                t.transform.translation.y = y
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw
                self.tf_broadcaster.sendTransform(t)

                # Publish /odom
                odom_msg = Odometry()
                odom_msg.header.stamp = now.to_msg()
                odom_msg.header.frame_id = 'odom'
                odom_msg.child_frame_id = 'base_link'
                odom_msg.pose.pose.position.x = x
                odom_msg.pose.pose.position.y = y
                odom_msg.pose.pose.orientation.z = qz
                odom_msg.pose.pose.orientation.w = qw
                odom_msg.twist.twist.linear.x = v
                odom_msg.twist.twist.angular.z = w
                self.odom_pub.publish(odom_msg)
            else:
                del self.buffer[0]

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            # Send neutral stop command [0xFF, 90, 90] before closing port
            try:
                self.ser.write(bytes([0xFF, 90, 90]))
            except Exception:
                pass
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RoverBaseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
