#!/usr/bin/env python3
"""
Preview MJPEG bruto para captura de dataset.

Endpoints:
  /          Página de visualização
  /stream    Stream MJPEG contínuo
  /snapshot  Um único frame JPEG limpo
  /health    Estado do servidor
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify


app = Flask(__name__)

_lock = threading.Lock()
_latest_jpg = b""
_camera = None
_frame_count = 0


def capture_loop():
    """Captura continuamente e mantém somente o frame mais recente."""
    global _latest_jpg, _frame_count

    while True:
        success, frame = _camera.read()

        if not success:
            time.sleep(0.05)
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

        if success:
            with _lock:
                _latest_jpg = encoded.tobytes()
                _frame_count += 1


def generate_mjpeg():
    """Gera a resposta multipart utilizada pelo navegador."""
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

        time.sleep(0.033)


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Preview bruto - Dataset de EPI</title>
        <style>
            body {
                margin: 0;
                background: #111;
                color: #eee;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding-top: 24px;
            }
            h1 {
                margin-bottom: 12px;
            }
            img {
                width: min(90vw, 1280px);
                border: 2px solid #555;
                border-radius: 6px;
            }
            p {
                color: #aaa;
            }
        </style>
    </head>
    <body>
        <h1>Anker C200 - Preview bruto</h1>
        <img src="/stream">
        <p>Imagem limpa, sem YOLO, bounding boxes ou OSD.</p>
    </body>
    </html>
    """


@app.route("/stream")
def stream():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/snapshot")
def snapshot():
    with _lock:
        jpg = _latest_jpg

    if not jpg:
        return Response(status=503)

    return Response(jpg, mimetype="image/jpeg")


@app.route("/health")
def health():
    with _lock:
        count = _frame_count

    return jsonify(
        status="ok",
        camera="Anker PowerConf C200",
        frame_count=count,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview bruto da webcam USB"
    )
    parser.add_argument(
        "--device",
        default="/dev/video8",
        help="Dispositivo V4L2 da webcam",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


def main():
    global _camera

    args = parse_args()

    _camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    _camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    _camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    _camera.set(cv2.CAP_PROP_FPS, args.fps)
    _camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not _camera.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a webcam: {args.device}"
        )

    actual_width = int(_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = _camera.get(cv2.CAP_PROP_FPS)

    print("=" * 60)
    print("PREVIEW BRUTO PARA DATASET")
    print("=" * 60)
    print(f"Dispositivo : {args.device}")
    print(f"Resolução   : {actual_width}x{actual_height}")
    print(f"FPS         : {actual_fps:.1f}")
    print(f"Navegador   : http://192.168.0.5:{args.port}/")
    print(f"Snapshot    : http://192.168.0.5:{args.port}/snapshot")
    print("=" * 60)

    producer = threading.Thread(
        target=capture_loop,
        daemon=True,
        name="CaptureThread",
    )
    producer.start()

    time.sleep(1.0)

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        _camera.release()


if __name__ == "__main__":
    main()
