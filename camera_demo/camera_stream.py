import cv2
import time
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = "/home/lidarrover/yolo26n.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)


def generate_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not camera.isOpened():
        print("ERROR: Could not open Logitech C922")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    previous_time = time.perf_counter()

    try:
        while True:
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

            annotated = results[0].plot()

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

            success, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    finally:
        camera.release()


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MavSentry - Live Object Detection</title>
        <style>
            body {
                background: #111827;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 30px;
            }
            img {
                width: 90%;
                max-width: 960px;
                border: 2px solid #3b82f6;
                border-radius: 12px;
            }
        </style>
    </head>
    <body>
        <h1>MavSentry - Live Object Detection</h1>
        <p>Logitech C922 | Raspberry Pi 5 | YOLO</p>
        <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Starting camera server on port 5000...")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
