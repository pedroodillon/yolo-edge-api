#!/usr/bin/env python3
"""
Captura imagens do endpoint /snapshot do raw_server.py e salva em
dataset/raw/. Frames borrados são descartados automaticamente.
"""

import argparse
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


OUTPUT_DIR = Path("dataset/raw")


def fetch_snapshot(url):
    """Obtém o frame mais recente do servidor de preview."""
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read()

    frame = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    return frame


def calculate_sharpness(frame):
    """Calcula a nitidez pela variância do Laplaciano."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Captura de frames para dataset de EPI"
    )
    parser.add_argument(
        "--snapshot-url",
        default="http://127.0.0.1:5001/snapshot",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Quantidade de imagens válidas a salvar",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Intervalo entre capturas automáticas",
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=80.0,
        help="Limite mínimo de nitidez",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Captura uma imagem a cada ENTER",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    discarded = 0
    failed = 0

    print("=" * 62)
    print("CAPTURA DE FRAMES PARA DATASET DE EPI")
    print("=" * 62)
    print(f"Fonte                : {args.snapshot_url}")
    print(f"Meta                 : {args.total} imagens válidas")
    print(f"Intervalo            : {args.interval:.1f} s")
    print(f"Limite de nitidez    : {args.sharpness:.1f}")
    print(f"Diretório            : {OUTPUT_DIR.resolve()}")
    print(
        "Modo                 : "
        + ("manual" if args.manual else "automático")
    )
    print("Pressione Ctrl+C para encerrar antecipadamente.")
    print("=" * 62)

    try:
        while saved < args.total:
            if args.manual:
                input(
                    f"[{saved:>3}/{args.total}] "
                    "Pressione ENTER para capturar..."
                )
            elif saved > 0 or discarded > 0 or failed > 0:
                time.sleep(args.interval)

            try:
                frame = fetch_snapshot(args.snapshot_url)
            except (
                urllib.error.URLError,
                TimeoutError,
            ) as error:
                failed += 1
                print(f"\n[AVISO] Falha de conexão: {error}")
                continue

            if frame is None:
                failed += 1
                print("\n[AVISO] Frame inválido.")
                continue

            sharpness = calculate_sharpness(frame)

            if sharpness < args.sharpness:
                discarded += 1
                print(
                    f"\rSalvos: {saved:>3}/{args.total} | "
                    f"Descartados: {discarded:>3} | "
                    f"Falhas: {failed:>2} | "
                    f"Nitidez: {sharpness:>7.1f} — DESCARTADO",
                    end="",
                    flush=True,
                )
                continue

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )[:-3]

            output_path = (
                OUTPUT_DIR / f"frame_{timestamp}.jpg"
            )

            success = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            if not success:
                failed += 1
                print("\n[AVISO] Não foi possível salvar o frame.")
                continue

            saved += 1

            print(
                f"\rSalvos: {saved:>3}/{args.total} | "
                f"Descartados: {discarded:>3} | "
                f"Falhas: {failed:>2} | "
                f"Nitidez: {sharpness:>7.1f}",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuário.")

    finally:
        print("\n" + "=" * 62)
        print("RELATÓRIO FINAL DA CAPTURA")
        print("=" * 62)
        print(f"Frames salvos       : {saved}")
        print(f"Frames descartados  : {discarded}")
        print(f"Falhas de captura   : {failed}")
        print(f"Diretório de saída  : {OUTPUT_DIR.resolve()}")
        print("=" * 62)


if __name__ == "__main__":
    main()
