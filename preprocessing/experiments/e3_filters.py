"""
Experimento E3: impacto dos filtros de suavização na
detecção de EPIs.
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from preprocessing.utils.evaluate import evaluate_pipeline


def preproc_rgb_only(frame: np.ndarray) -> np.ndarray:
    """Converte para RGB sem aplicar filtros."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preproc_gaussian_3x3(frame: np.ndarray) -> np.ndarray:
    """Aplica GaussianBlur 3x3 com sigma 0.8."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(
        frame_rgb,
        (3, 3),
        sigmaX=0.8,
    )


def preproc_gaussian_5x5(frame: np.ndarray) -> np.ndarray:
    """Aplica GaussianBlur 5x5 com sigma 1.5."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(
        frame_rgb,
        (5, 5),
        sigmaX=1.5,
    )


def preproc_median_3(frame: np.ndarray) -> np.ndarray:
    """Aplica medianBlur com kernel 3."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.medianBlur(frame_rgb, 3)


def benchmark_filter_cost(number_of_frames: int = 200):
    """Mede o custo médio dos filtros em frames 640x480."""
    test_frame = np.random.randint(
        0,
        255,
        (480, 640, 3),
        dtype=np.uint8,
    )

    filters = [
        (
            "cvtColor apenas",
            lambda frame: cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            ),
        ),
        (
            "GaussianBlur 3x3",
            lambda frame: cv2.GaussianBlur(
                frame,
                (3, 3),
                0.8,
            ),
        ),
        (
            "GaussianBlur 5x5",
            lambda frame: cv2.GaussianBlur(
                frame,
                (5, 5),
                1.5,
            ),
        ),
        (
            "medianBlur kernel=3",
            lambda frame: cv2.medianBlur(frame, 3),
        ),
        (
            "bilateralFilter",
            lambda frame: cv2.bilateralFilter(
                frame,
                9,
                75,
                75,
            ),
        ),
    ]

    print(
        "\n--- Custo por filtro "
        "(média de 200 frames 640x480) ---"
    )

    for filter_name, filter_function in filters:
        start_time = time.perf_counter()

        for _ in range(number_of_frames):
            filter_function(test_frame)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) / number_of_frames * 1000

        print(
            f"{filter_name:24s}: "
            f"{elapsed_ms:.3f} ms/frame"
        )


if __name__ == "__main__":
    print("=" * 65)
    print("E3 - Filtragem: Gaussiano versus Mediana")
    print("=" * 65)

    results = [
        evaluate_pipeline(
            None,
            "E3-baseline",
        ),
        evaluate_pipeline(
            preproc_rgb_only,
            "E3-A: sem filtro",
        ),
        evaluate_pipeline(
            preproc_gaussian_3x3,
            "E3-B: GaussianBlur 3x3",
        ),
        evaluate_pipeline(
            preproc_gaussian_5x5,
            "E3-C: GaussianBlur 5x5",
        ),
        evaluate_pipeline(
            preproc_median_3,
            "E3-D: medianBlur kernel=3",
        ),
    ]

    benchmark_filter_cost()

    print("\n--- Resumo E3 ---")
    baseline_map = results[0]["map50"]

    for result in results[1:]:
        delta = result["map50"] - baseline_map

        print(
            f"{result['label']:35s} "
            f"mAP@0.5={result['map50']:.4f} "
            f"delta={delta:+.4f} "
            f"preproc={result['preproc_ms']:.3f}ms"
        )
