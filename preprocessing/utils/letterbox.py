"""
Funções auxiliares para redimensionamento com letterbox e
correção de bounding boxes.
"""


import cv2
import numpy as np


def letterbox(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Redimensiona a imagem preservando sua proporção e
    adiciona padding para formar uma imagem quadrada.

    Returns:
        frame_lb: imagem redimensionada com padding.
        scale: fator uniforme de escala aplicado.
        (pad_w, pad_h): padding esquerdo e superior.
    """
    height, width = frame.shape[:2]

    scale = min(
        target_size / height,
        target_size / width,
    )

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2

    frame_lb = np.full(
        (target_size, target_size, 3),
        pad_color,
        dtype=np.uint8,
    )

    frame_lb[
        pad_h : pad_h + new_height,
        pad_w : pad_w + new_width,
    ] = resized

    return frame_lb, scale, (pad_w, pad_h)


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> np.ndarray:
    """
    Mapeia bounding boxes do espaço letterboxed para o
    espaço da imagem original.
    """
    boxes = boxes_xyxy.copy().astype(float)

    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale

    return boxes
