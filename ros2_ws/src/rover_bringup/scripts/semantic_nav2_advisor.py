#!/usr/bin/env python3
"""Provide person-aware speed advice to Nav2 from YOLO and RPLIDAR.

This is an advisory, not a collision-prevention mechanism. Nav2's existing
costmap and Collision Monitor remain responsible for obstacle handling.
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from nav2_msgs.msg import SpeedLimit
from std_msgs.msg import String


def person_wedges(detections, image_width, hfov_deg, min_confidence,
                  roi_min, roi_max, camera_yaw_deg, angular_margin_deg):
    """Return approximate LiDAR-frame bearing intervals for visible people."""
    wedges = []
    width = max(float(image_width), 1.0)
    camera_half_fov_rad = math.radians(float(hfov_deg)) / 2.0
    yaw_offset = math.radians(float(camera_yaw_deg))
    margin = math.radians(float(angular_margin_deg))
    for detection in detections:
        center_x = float(detection.bbox.center.position.x)
        box_width = float(detection.bbox.size_x)
        if not (roi_min * width <= center_x <= roi_max * width):
            continue
        matched = any(
            str(result.hypothesis.class_id).strip().lower() == 'person'
            and float(result.hypothesis.score) >= min_confidence
            for result in detection.results
        )
        if not matched:
            continue
        x_left = max(0.0, center_x - box_width / 2.0)
        x_right = min(width, center_x + box_width / 2.0)
        # Image x increases right; ROS LiDAR bearing increases counter-clockwise.
        bearing_left = yaw_offset + (1.0 - 2.0 * x_left / width) * camera_half_fov_rad
        bearing_right = yaw_offset + (1.0 - 2.0 * x_right / width) * camera_half_fov_rad
        wedges.append((min(bearing_left, bearing_right) - margin,
                       max(bearing_left, bearing_right) + margin))
    return wedges


def nearest_person_wedge_range(scan, wedges):
    """Find closest valid scan return in any camera-person angular interval."""
    if not wedges or not math.isfinite(scan.angle_increment) or scan.angle_increment == 0:
        return None
    nearest = math.inf
    for index, value in enumerate(scan.ranges):
        distance = float(value)
        if not (math.isfinite(distance) and scan.range_min <= distance <= scan.range_max):
            continue
        angle = scan.angle_min + index * scan.angle_increment
        # Normalize to [-pi, pi] to handle scans that cross the wrap boundary.
        for low, high in wedges:
            difference = math.atan2(math.sin(angle - (low + high) / 2.0),
                                    math.cos(angle - (low + high) / 2.0))
            if abs(difference) <= (high - low) / 2.0:
                nearest = min(nearest, distance)
                break
    return nearest if math.isfinite(nearest) else None


class SemanticNav2Advisor(Node):
    def __init__(self):
        super().__init__('semantic_nav2_advisor')
        self.declare_parameter('enabled', False)
        self.declare_parameter('camera_timeout_sec', 1.5)
        self.declare_parameter('scan_timeout_sec', 1.0)
        self.declare_parameter('image_width_px', 640.0)
        self.declare_parameter('camera_hfov_deg', 70.0)
        self.declare_parameter('camera_yaw_offset_deg', 0.0)
        self.declare_parameter('bearing_margin_deg', 5.0)
        self.declare_parameter('zone_min_x', 0.20)
        self.declare_parameter('zone_max_x', 0.80)
        self.declare_parameter('min_person_confidence', 0.45)
        self.declare_parameter('slowdown_range_m', 1.5)
        self.declare_parameter('release_range_m', 1.8)
        self.declare_parameter('slowdown_percent', 60.0)
        self.declare_parameter('clear_hold_sec', 1.0)

        self.last_camera_at = None
        self.last_scan_at = None
        self.person_wedges = []
        self.scan = None
        self.nearest = None
        self.slowing = False
        self.last_near_at = None
        self.was_enabled = False

        self.create_subscription(Detection2DArray, '/camera/detections', self.on_camera, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.speed_pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        self.status_pub = self.create_publisher(String, '/rover/semantic_nav2_status', 10)
        self.create_timer(0.2, self.update)
        self.get_logger().info('Semantic Nav2 advisor started (disabled by default)')

    def p(self, key):
        return self.get_parameter(key).value

    def on_camera(self, msg):
        self.person_wedges = person_wedges(
            msg.detections, self.p('image_width_px'), self.p('camera_hfov_deg'),
            self.p('min_person_confidence'), self.p('zone_min_x'), self.p('zone_max_x'),
            self.p('camera_yaw_offset_deg'), self.p('bearing_margin_deg')
        )
        self.last_camera_at = time.monotonic()

    def on_scan(self, msg):
        self.scan = msg
        self.last_scan_at = time.monotonic()

    def update(self):
        now = time.monotonic()
        enabled = bool(self.p('enabled'))
        camera_fresh = self.last_camera_at is not None and (
            now - self.last_camera_at < float(self.p('camera_timeout_sec')))
        scan_fresh = self.last_scan_at is not None and (
            now - self.last_scan_at < float(self.p('scan_timeout_sec')))
        has_person = bool(self.person_wedges) and camera_fresh
        self.nearest = (
            nearest_person_wedge_range(self.scan, self.person_wedges)
            if camera_fresh and scan_fresh and has_person else None
        )

        if not enabled:
            self.slowing = False
            self.last_near_at = None
        elif not (camera_fresh and scan_fresh):
            # Retain any previous slowdown during a sensor interruption.
            # Nav2's own LiDAR collision monitoring remains essential.
            pass
        elif self.nearest is not None and self.nearest <= float(self.p('slowdown_range_m')):
            self.slowing = True
            self.last_near_at = now
        elif self.slowing:
            far_enough = self.nearest is None or self.nearest >= float(self.p('release_range_m'))
            hold_over = self.last_near_at is not None and (
                now - self.last_near_at >= float(self.p('clear_hold_sec')))
            if far_enough and hold_over:
                self.slowing = False

        limit = SpeedLimit()
        limit.header.stamp = self.get_clock().now().to_msg()
        limit.header.frame_id = 'base_link'
        limit.percentage = True
        limit.speed_limit = float(self.p('slowdown_percent')) if enabled and self.slowing else 0.0
        # When disabled, clear a previously active limit once. Avoid repeatedly
        # overriding unrelated Nav2 speed-limit providers while disabled.
        if enabled or self.was_enabled:
            self.speed_pub.publish(limit)
        self.was_enabled = enabled

        if not enabled:
            state = 'DISABLED'
        elif not camera_fresh or not scan_fresh:
            state = 'SENSOR_STALE'
        elif self.slowing:
            state = 'PERSON_NEAR_SLOWDOWN'
        elif has_person:
            state = 'PERSON_VISIBLE_NAV2_NORMAL'
        else:
            state = 'NAV2_NORMAL'

        status = String()
        status.data = json.dumps({
            'state': state,
            'enabled': enabled,
            'camera_fresh': camera_fresh,
            'scan_fresh': scan_fresh,
            'person_in_camera_region': has_person,
            'nearest_matching_scan_m': round(self.nearest, 3) if self.nearest is not None else None,
            'speed_limit_percent': limit.speed_limit,
            'approximate_bearing_match': True,
        })
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticNav2Advisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
