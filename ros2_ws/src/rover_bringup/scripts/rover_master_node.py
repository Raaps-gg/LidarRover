#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool, String

import subprocess
import signal
import serial
import math
import json
import psutil
import os
import time

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
        self.last_joy_time = self.get_clock().now()
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

        # LED Status Tracker (0: Manual/Red, 1: Nav2/Blue, 2: Explore/Green)
        self.current_led_mode = -1

        # Subscribers & Publishers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.nav_sub = self.create_subscription(Twist, '/cmd_vel', self.nav_callback, 10)
        self.active_cmd_pub = self.create_publisher(Twist, '/rover/active_cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # System Health Publisher & 1 Hz Timer
        self.stats_pub = self.create_publisher(String, '/rover/system_stats', 10)
        self.create_timer(1.0, self.publish_system_stats)

        # Explore-Lite SubProcess Handler
        self.explore_proc = None
        self.explore_sub = self.create_subscription(Bool, '/rover/toggle_explore', self.explore_callback, 10)
        self.explore_status_pub = self.create_publisher(Bool, '/rover/explore_status', 10)
    
        self.create_timer(1.0, self.broadcast_explore_status)

        # Main loop timer (20 Hz / 50 ms)
        self.create_timer(0.05, self.control_and_serial_loop)
    
    def explore_callback(self, msg: Bool):
        should_run = msg.data

        if should_run:
            # Check if its already running
            if self.explore_proc is None or self.explore_proc.poll() is not None:
                self.get_logger().info("Starting Explore-Lite...")
                # os.setsid 
                self.explore_proc = subprocess.Popen(
                    ["ros2", "launch", "explore_lite", "explore.launch.py"],
                    preexec_fn=os.setsid
                )
        else:
            # if running, terminate whole process
            if self.explore_proc and self.explore_proc.poll() is None:
                self.get_logger().info("Terminating Explore-Lite process group...")
                try:
                    os.killpg(os.getpgid(self.explore_proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.explore_proc = None
 
        self.broadcast_explore_status()
  
    def broadcast_explore_status(self):
        is_running = self.explore_proc is not None and self.explore_proc.poll() is None
        status_msg = Bool()
        status_msg.data = is_running
        self.explore_status_pub.publish(status_msg)

    def publish_system_stats(self):
        # 1. Per-Core CPU & System Memory
        cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
        cpu_avg = round(sum(cpu_cores) / len(cpu_cores), 1) if cpu_cores else 0.0
        vm = psutil.virtual_memory()

        # 2. SoC Temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = round(float(f.read().strip()) / 1000.0, 1)
        except Exception:
            temp_c = 0.0

        # 3. Load Average & Uptime
        load_avg = [round(x, 2) for x in os.getloadavg()]
        uptime_sec = int(time.time() - psutil.boot_time())
        hrs, rem = divmod(uptime_sec, 3600)
        mins, secs = divmod(rem, 60)
        uptime_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        # 4. Process Listing for Main Tab (Top 8 by CPU)
        proc_list = []
        try:
            for p in psutil.process_iter(['pid', 'username', 'cpu_percent', 'memory_percent', 'name', 'io_counters']):
                proc_list.append(p.info)
        except Exception:
            pass

        # Sort Main Processes
        top_procs = sorted(proc_list, key=lambda x: x.get('cpu_percent') or 0.0, reverse=True)[:8]
        clean_procs = []
        for p in top_procs:
            clean_procs.append({
                'pid': p.get('pid', 0),
                'user': (p.get('username') or 'root')[:8],
                'cpu': round(p.get('cpu_percent') or 0.0, 1),
                'mem': round(p.get('memory_percent') or 0.0, 1),
                'cmd': (p.get('name') or 'unknown')[:20]
            })

        # Sort I/O Processes (Top 8 by total disk activity)
        top_io = []
        for p in proc_list:
            io = p.get('io_counters')
            if io:
                top_io.append({
                    'pid': p.get('pid', 0),
                    'user': (p.get('username') or 'root')[:8],
                    'read_mb': round(io.read_bytes / (1024 * 1024), 1),
                    'write_mb': round(io.write_bytes / (1024 * 1024), 1),
                    'cmd': (p.get('name') or 'unknown')[:20]
                })
        clean_io = sorted(top_io, key=lambda x: (x['read_mb'] + x['write_mb']), reverse=True)[:8]

        payload = {
            'cpu_avg': cpu_avg,
            'cores': cpu_cores,
            'mem_used': round(vm.used / (1024 * 1024)),
            'mem_total': round(vm.total / (1024 * 1024)),
            'mem_percent': vm.percent,
            'temp': temp_c,
            'load': load_avg,
            'uptime': uptime_str,
            'tasks': len(proc_list),
            'procs': clean_procs,
            'io_procs': clean_io
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.stats_pub.publish(msg)
        
    def joy_callback(self, msg: Joy):
        self.last_joy_time = self.get_clock().now()
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

        # Evaluate Active Operational State
        if self.explore_proc is not None and self.explore_proc.poll() is None:
            active_mode = 2  # Green: Explore-Lite Running
        elif not self.manual_mode:
            active_mode = 1  # Blue: Nav2 Autonomous Navigation
        else:
            active_mode = 0  # Red: Manual Teleoperation

        # Send 'c <mode>\n' to Arduino Due only on state transitions or periodically
        if active_mode != self.current_led_mode:
            self.current_led_mode = active_mode
            if self.ser.is_open:
                self.ser.write(f"c {self.current_led_mode}\n".encode('utf-8'))

        # 1. Watchdog- nav2
        if not self.manual_mode:
            if (now - self.last_nav_time).nanoseconds / 1e9 < 0.5:
                self.target_vx = self.nav_vx
                self.target_wz = self.nav_wz
            else:
                self.target_vx = 0.0
                self.target_wz = 0.0
        # 2. Watchdog- manual
        else: # if no command for 0.5 seconds, stop
            if (now - self.last_joy_time).nanoseconds / 1e9 >= 0.5:
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
        if node.explore_proc and node.explore_proc.poll() is None:
            try:
                os.killpg(os.getpgid(node.explore_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
