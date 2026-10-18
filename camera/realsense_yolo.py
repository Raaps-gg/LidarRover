import cv2
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "/home/teamrover/yolo26n.pt"

# Distance zones
CLOSE_DISTANCE = 1.0
CAUTION_DISTANCE = 2.0

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# RealSense
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    30
)

config.enable_stream(
    rs.stream.depth,
    640, 480,
    rs.format.z16,
    30
)

align = rs.align(rs.stream.color)

pipeline.start(config)

# -----------------------------
# FPS
# -----------------------------
prev_time = time.time()
fps = 0.0

try:

    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # -----------------------------
        # YOLO tracking
        # -----------------------------
        results = model.track(
            color_image,
            device=0,
            imgsz=640,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        annotated = color_image.copy()

        # -----------------------------
        # Process detections
        # -----------------------------
        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1 = max(0, min(639, x1))
                y1 = max(0, min(479, y1))
                x2 = max(0, min(639, x2))
                y2 = max(0, min(479, y2))

                # Center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # -----------------------------
                # Stable depth
                # -----------------------------
                radius = 2
                depths = []

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):

                        px = cx + dx
                        py = cy + dy

                        if 0 <= px < 640 and 0 <= py < 480:

                            d = depth_frame.get_distance(px, py)

                            if d > 0:
                                depths.append(d)

                if depths:
                    distance = float(np.median(depths))
                else:
                    distance = 0.0

                # -----------------------------
                # Object information
                # -----------------------------
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                name = model.names[cls]

                if box.id is not None:
                    track_id = int(box.id[0])
                    object_name = f"{name} #{track_id}"
                else:
                    object_name = name

                # -----------------------------
                # LEFT / CENTER / RIGHT
                # -----------------------------
                if cx < 213:
                    position = "LEFT"
                elif cx < 427:
                    position = "CENTER"
                else:
                    position = "RIGHT"

                # -----------------------------
                # Distance color
                # -----------------------------
                if distance <= 0:
                    color = (255, 255, 255)
                    zone = "UNKNOWN"

                elif distance < CLOSE_DISTANCE:
                    color = (0, 0, 255)
                    zone = "CLOSE"

                elif distance < CAUTION_DISTANCE:
                    color = (0, 255, 255)
                    zone = "CAUTION"

                else:
                    color = (0, 255, 0)
                    zone = "CLEAR"

                # -----------------------------
                # Draw bounding box
                # -----------------------------
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # -----------------------------
                # Label
                # -----------------------------
                if distance > 0:

                    label = (
                        f"{object_name} "
                        f"{conf:.2f} "
                        f"{distance:.2f}m "
                        f"{position}"
                    )

                else:

                    label = (
                        f"{object_name} "
                        f"{conf:.2f} "
                        f"depth N/A "
                        f"{position}"
                    )

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )

                label_y = max(y1 - 8, th + 5)

                cv2.rectangle(
                    annotated,
                    (x1, label_y - th - 5),
                    (x1 + tw + 5, label_y + 2),
                    color,
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2
                )

        # -----------------------------
        # FPS
        # -----------------------------
        current_time = time.time()
        elapsed = current_time - prev_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            fps = fps * 0.9 + instant_fps * 0.1

        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # -----------------------------
        # Display
        # -----------------------------
        cv2.imshow(
            "RealSense + YOLO + Depth + Tracking",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()
