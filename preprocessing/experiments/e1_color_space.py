"""
Experimento E1: impacto da conversão de espaço de cor
BGR para RGB.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from preprocessing.utils.evaluate import evaluate_pipeline


def preproc_bgr_raw(frame: np.ndarray) -> np.ndarray:
    """Mantém o frame recebido no formato BGR."""
    return frame


def preproc_rgb_correct(frame: np.ndarray) -> np.ndarray:
    """Converte explicitamente o frame de BGR para RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preproc_rgb_flip(frame: np.ndarray) -> np.ndarray:
    """Inverte os canais por indexação NumPy."""
    return frame[:, :, ::-1]


if __name__ == "__main__":
    print("=" * 65)
    print("E1 - Impacto da conversão de espaço de cor")
    print("=" * 65)

    results = [
        evaluate_pipeline(
            None,
            "E1-baseline (Ultralytics padrão)",
        ),
        evaluate_pipeline(
            preproc_bgr_raw,
            "E1-A: BGR sem conversão",
        ),
        evaluate_pipeline(
            preproc_rgb_correct,
            "E1-B: RGB correto (cvtColor)",
        ),
        evaluate_pipeline(
            preproc_rgb_flip,
            "E1-C: RGB por NumPy flip",
        ),
    ]

    print("\n--- Resumo E1 ---")
    baseline_map = results[0]["map50"]

    for result in results[1:]:
        delta = result["map50"] - baseline_map

        print(
            f"{result['label']:35s} "
            f"mAP@0.5={result['map50']:.4f} "
            f"delta={delta:+.4f}"
        )
