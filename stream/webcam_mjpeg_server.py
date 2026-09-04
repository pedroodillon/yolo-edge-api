#!/usr/bin/env python3
"""
Servidor MJPEG com YOLO para webcam USB.

Exibe a Anker PowerConf C200 no navegador com bounding boxes,
confiança e OSD, usando uma página ampliada.
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify


sys.path.insert(0, str(Path(__file__).parent.parent))

from stream.v3_optimized import RealtimeDetector


app = Flask(__name__)

_camera = None
_detector = None
_latest_jpg = b""
_lock = threading.Lock()


class USBCamera:
    """Captura V4L2 em thread com buffer de apenas um frame."""

    def __init__(self, device, width, height, fps):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self._capture = None
        self._buffer = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="USBCameraThread",
        )

        self.frames_captured = 0
        self.frames_dropped = 0

    def start(self):
        self._capture = cv2.VideoCapture(
            self.device,
            cv2.CAP_V4L2,
        )

        self._capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self._capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )
        self._capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a webcam {self.device}"
            )

        actual_width = int(
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        actual_height = int(
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        print(
            f"[USBCamera] {self.device} — "
            f"{actual_width}x{actual_height} @ "
            f"{actual_fps:.1f} FPS"
        )

        self._running.set()
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running.is_set():
            success, frame = self._capture.read()

            if not success:
                time.sleep(0.02)
                continue

            self.frames_captured += 1

            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            self._buffer.put(frame)

    def read(self, timeout=2.0):
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running.clear()

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._capture is not None:
            self._capture.release()


def frame_producer():
    """Captura, executa YOLO, desenha OSD e comprime em JPEG."""
    global _latest_jpg

    while True:
        frame = _camera.read(timeout=2.0)

        if frame is None:
            continue

        annotated = _detector.process(frame)

        success, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

        if success:
            with _lock:
                _latest_jpg = encoded.tobytes()


def generate_mjpeg():
    while True:
        with _lock:
            jpg = _latest_jpg

        if not jpg:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg
            + b"\r\n"
        )

        time.sleep(0.025)


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">
        <title>YOLOv8 - Anker C200</title>
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                background: #0b0b0b;
                color: #f2f2f2;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            header {
                width: 100%;
                padding: 12px 20px;
                background: #151515;
                border-bottom: 1px solid #333;
                text-align: center;
            }

            h1 {
                margin: 0;
                font-size: 1.35rem;
            }

            .viewer {
                width: 96vw;
                max-width: 1600px;
                margin: 14px auto;
                display: flex;
                justify-content: center;
            }

            .viewer img {
                width: 100%;
                max-height: calc(100vh - 100px);
                object-fit: contain;
                background: #000;
                border: 2px solid #444;
                border-radius: 6px;
                box-shadow: 0 0 24px rgba(0, 0, 0, 0.7);
            }

            footer {
                padding: 0 10px 12px;
                color: #999;
                font-size: 0.85rem;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>YOLOv8 — Anker PowerConf C200 — Tempo Real</h1>
        </header>

        <main class="viewer">
            <img src="/stream" alt="Stream YOLO em tempo real">
        </main>

        <footer>
            Bounding boxes, confiança e OSD processados
            na Raspberry Pi 5.
        </footer>
    </body>
    </html>
    """


@app.route("/stream")
def stream():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        stream="active",
        camera="Anker PowerConf C200",
        device=_camera.device if _camera else None,
        frame_count=(
            _detector._frame_idx
            if _detector is not None
            else 0
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO MJPEG para webcam USB"
    )
    parser.add_argument(
        "--device",
        default="/dev/video0",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
    )
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--infer-every",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--infer-size",
        type=int,
        default=416,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main():
    global _camera, _detector

    args = parse_args()

    _camera = USBCamera(
        args.device,
        args.width,
        args.height,
        args.fps,
    ).start()

    _detector = RealtimeDetector(
        args.model,
        args.conf,
        args.infer_every,
        args.infer_size,
    )

    producer = threading.Thread(
        target=frame_producer,
        daemon=True,
        name="YOLOProducer",
    )
    producer.start()

    time.sleep(1.0)

    print("=" * 64)
    print("YOLO MJPEG — ANKER POWERCONF C200")
    print("=" * 64)
    print(f"Navegador : http://192.168.0.5:{args.port}/")
    print(f"Health    : http://192.168.0.5:{args.port}/health")
    print("=" * 64)

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        _camera.stop()


if __name__ == "__main__":
    main()
