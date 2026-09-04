"""
Experimento E2: resize simples versus letterbox.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from preprocessing.utils.evaluate import evaluate_pipeline
from preprocessing.utils.letterbox import (
    adjust_bboxes,
    letterbox,
)

TARGET = 416


def preproc_naive_resize(frame: np.ndarray) -> np.ndarray:
    """Redimensiona para um quadrado, distorcendo a proporção."""
    return cv2.resize(frame, (TARGET, TARGET))


def preproc_letterbox(frame: np.ndarray) -> np.ndarray:
    """Redimensiona preservando a proporção e adicionando padding."""
    frame_letterboxed, _, _ = letterbox(
        frame,
        target_size=TARGET,
    )
    return frame_letterboxed


def demo_bbox_adjustment():
    """Demonstra o retorno de uma bbox ao espaço original."""
    images_dir = Path(
        "dataset/exports/epi-v1/valid/images"
    )
    image_path = min(images_dir.glob("*.jpg"))
    frame = cv2.imread(str(image_path))

    original_height, original_width = frame.shape[:2]

    frame_letterboxed, scale, (pad_w, pad_h) = letterbox(
        frame,
        target_size=TARGET,
    )

    bbox_letterboxed = np.array(
        [[60, 90, 200, 310]],
        dtype=float,
    )

    bbox_original = adjust_bboxes(
        bbox_letterboxed,
        scale,
        pad_w,
        pad_h,
    )

    print(
        f"Frame original: {original_width}x"
        f"{original_height}"
    )
    print(
        f"Frame letterboxed: {TARGET}x{TARGET} "
        f"(scale={scale:.4f}, "
        f"pad_w={pad_w}, pad_h={pad_h})"
    )
    print(
        "Bbox no espaço letterboxed:",
        bbox_letterboxed[0].astype(int).tolist(),
    )
    print(
        "Bbox ajustada ao original:",
        bbox_original[0].astype(int).tolist(),
    )

    cv2.rectangle(
        frame_letterboxed,
        tuple(bbox_letterboxed[0, :2].astype(int)),
        tuple(bbox_letterboxed[0, 2:].astype(int)),
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        frame,
        tuple(bbox_original[0, :2].astype(int)),
        tuple(bbox_original[0, 2:].astype(int)),
        (0, 255, 0),
        2,
    )

    cv2.imwrite(
        "preprocessing/outputs/e2_bbox_letterboxed.jpg",
        frame_letterboxed,
    )
    cv2.imwrite(
        "preprocessing/outputs/e2_bbox_original.jpg",
        frame,
    )


if __name__ == "__main__":
    print("=" * 65)
    print("E2 - Resize simples versus letterbox")
    print("=" * 65)

    results = [
        evaluate_pipeline(
            None,
            "E2-baseline",
        ),
        evaluate_pipeline(
            preproc_naive_resize,
            "E2-A: resize simples",
        ),
        evaluate_pipeline(
            preproc_letterbox,
            "E2-B: letterbox",
        ),
    ]

    print("\n--- Demonstração do ajuste de coordenadas ---")
    demo_bbox_adjustment()

    print("\n--- Resumo E2 ---")
    baseline_map = results[0]["map50"]

    for result in results[1:]:
        delta = result["map50"] - baseline_map

        print(
            f"{result['label']:30s} "
            f"mAP@0.5={result['map50']:.4f} "
            f"delta={delta:+.4f}"
        )
