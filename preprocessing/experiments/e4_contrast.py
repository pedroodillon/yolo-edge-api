"""
Experimento E4: equalização global versus CLAHE em imagens
com iluminação degradada.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

import preprocessing.utils.evaluate as evaluate_module
from preprocessing.utils.evaluate import evaluate_pipeline

DARK_DATASET_YAML = (
    "dataset/exports/epi-v1-dark/data.yaml"
)


def rgb_only(frame: np.ndarray) -> np.ndarray:
    """Converte para RGB sem corrigir o contraste."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def equalize_hist_lab(frame: np.ndarray) -> np.ndarray:
    """Aplica equalização global no canal L do espaço LAB."""
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(frame_lab)

    equalized_lightness = cv2.equalizeHist(lightness)

    equalized_lab = cv2.merge(
        [
            equalized_lightness,
            channel_a,
            channel_b,
        ]
    )

    return cv2.cvtColor(
        equalized_lab,
        cv2.COLOR_LAB2RGB,
    )


def clahe_lab(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Aplica CLAHE no canal L do espaço LAB."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(frame_lab)

    clahe_lightness = clahe.apply(lightness)

    clahe_lab_image = cv2.merge(
        [
            clahe_lightness,
            channel_a,
            channel_b,
        ]
    )

    return cv2.cvtColor(
        clahe_lab_image,
        cv2.COLOR_LAB2RGB,
    )


if __name__ == "__main__":
    print("=" * 65)
    print("E4 - Contraste em imagens com iluminação degradada")
    print("=" * 65)

    original_dataset_yaml = evaluate_module.DATASET_YAML
    evaluate_module.DATASET_YAML = DARK_DATASET_YAML

    try:
        results = [
            evaluate_pipeline(
                rgb_only,
                "E4-A: sem equalização",
            ),
            evaluate_pipeline(
                equalize_hist_lab,
                "E4-B: equalizeHist global",
            ),
            evaluate_pipeline(
                clahe_lab,
                "E4-C: CLAHE clip=2 tile=8",
            ),
        ]
    finally:
        evaluate_module.DATASET_YAML = (
            original_dataset_yaml
        )

    print("\n--- Resumo E4 ---")
    baseline_map = results[0]["map50"]

    for result in results:
        delta = result["map50"] - baseline_map

        print(
            f"{result['label']:35s} "
            f"mAP@0.5={result['map50']:.4f} "
            f"delta={delta:+.4f} "
            f"preproc={result['preproc_ms']:.3f}ms"
        )
