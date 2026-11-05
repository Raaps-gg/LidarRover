import cv2
import time
import threading
import rclpy

from flask import Flask, Response, jsonify
from ultralytics import YOLO
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

app = Flask(__name__)

MODEL_PATH = "/home/lidarrover/yolo26n.pt"
CAMERA_INDEX = 0

condition = threading.Condition()
latest_frame = None
latest_status = {
    "camera_online": False,
    "person_detected": None,
    "objects": [],
    "fps": 0.0
}
last_update = 0.0
frame_number = 0
stop_event = threading.Event()


def camera_loop(node):
    global latest_frame, latest_status, last_update, frame_number

    detection_pub = node.create_publisher(
        Detection2DArray,
        "/camera/detections",
        10
    )

    person_pub = node.create_publisher(
        Bool,
        "/camera/person_detected",
        10
    )

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Opening Logitech C922...")
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not camera.isOpened():
        print("ERROR: Camera unavailable")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    previous_time = time.perf_counter()

    try:
        while not stop_event.is_set():
            success, frame = camera.read()

            if not success:
                print("Camera frame capture failed")
                break

            results = model.predict(
                frame,
                imgsz=256,
                conf=0.4,
                device="cpu",
                verbose=False
            )

            result = results[0]

            msg = Detection2DArray()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "logitech_c922"

            objects = []
            person_detected = False

            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                name = model.names[cls]

                x1, y1, x2, y2 = [
                    float(x) for x in box.xyxy[0].tolist()
                ]

                detection = Detection2D()
                detection.header = msg.header

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(name)
                hypothesis.hypothesis.score = confidence

                detection.results.append(hypothesis)

                detection.bbox.center.position.x = (x1 + x2) / 2
                detection.bbox.center.position.y = (y1 + y2) / 2
                detection.bbox.center.theta = 0.0

                detection.bbox.size_x = x2 - x1
                detection.bbox.size_y = y2 - y1

                msg.detections.append(detection)

                objects.append({
                    "class": name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2]
                })

                if name == "person":
                    person_detected = True

            detection_pub.publish(msg)

            status_msg = Bool()
            status_msg.data = person_detected
            person_pub.publish(status_msg)

            annotated = result.plot()

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 0.001)
            previous_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            encoded, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )

            if not encoded:
                continue

            with condition:
                latest_frame = buffer.tobytes()
                latest_status = {
                    "camera_online": True,
                    "person_detected": person_detected,
                    "objects": objects,
                    "fps": round(fps, 1)
                }
                last_update = time.monotonic()
                frame_number += 1
                condition.notify_all()

    finally:
        camera.release()
        print("Camera released")


def generate_frames():
    previous_frame = -1

    while True:
        with condition:
            condition.wait_for(
                lambda: frame_number != previous_frame
                or stop_event.is_set(),
                timeout=2.0
            )

            if stop_event.is_set():
                return

            if latest_frame is None or frame_number == previous_frame:
                continue

            data = latest_frame
            previous_frame = frame_number

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + data
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <body style="background:#111827;color:white;text-align:center;">
      <h2>MavSentry Object Detection</h2>
      <img src="/video_feed" style="width:90%;max-width:900px;">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    with condition:
        data = dict(latest_status)
        data["objects"] = list(latest_status["objects"])
        online = (
            latest_frame is not None
            and time.monotonic() - last_update < 2.0
        )

    data["camera_online"] = online

    if not online:
        data["person_detected"] = None
        data["objects"] = []
        data["fps"] = 0.0

    return jsonify(data)


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("mavsentry_camera_detector")

    worker = threading.Thread(
        target=camera_loop,
        args=(node,),
        daemon=True
    )

    worker.start()

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            debug=False
        )
    finally:
        stop_event.set()
        with condition:
            condition.notify_all()
        worker.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown()
