"""
Módulo central de pré-processamento de imagens para o
projeto yolo-edge-api.

Integra-se ao pipeline de tempo real e à API REST.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from preprocessing.utils.letterbox import letterbox


@dataclass
class PreprocessConfig:
    """Configuração do pipeline de pré-processamento."""

    infer_size: int = 320
    convert_rgb: bool = True
    use_letterbox: bool = True
    gaussian_blur: bool = False
    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.8
    median_blur: bool = False
    median_ksize: int = 3
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    clahe_space: str = "lab"
    normalize: bool = False


@dataclass
class PreprocessResult:
    """Imagem processada e metadados da transformação."""

    frame: np.ndarray
    scale: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_w: int = 0
    pad_h: int = 0
    orig_size: tuple[int, int] = (0, 0)


class Preprocessor:
    """
    Encapsula um pipeline configurável de pré-processamento.

    Não mantém estado mutável específico entre os frames.
    """

    def __init__(
        self,
        config: PreprocessConfig | None = None,
    ):
        self.cfg = config or PreprocessConfig()

        if self.cfg.clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(
                    self.cfg.clahe_tile,
                    self.cfg.clahe_tile,
                ),
            )

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """Aplica o pipeline completo a um frame BGR."""

        orig_h, orig_w = frame.shape[:2]
        output = frame.copy()

        if self.cfg.clahe:
            output = self._apply_clahe(output)

        if self.cfg.convert_rgb:
            output = cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB,
            )

        if self.cfg.gaussian_blur:
            kernel = self.cfg.gaussian_ksize
            output = cv2.GaussianBlur(
                output,
                (kernel, kernel),
                sigmaX=self.cfg.gaussian_sigma,
            )
        elif self.cfg.median_blur:
            output = cv2.medianBlur(
                output,
                self.cfg.median_ksize,
            )

        if self.cfg.use_letterbox:
            output, scale, (pad_w, pad_h) = letterbox(
                output,
                self.cfg.infer_size,
            )
            scale_x = scale
            scale_y = scale

        else:
            output = cv2.resize(
                output,
                (
                    self.cfg.infer_size,
                    self.cfg.infer_size,
                ),
            )

            scale_x = self.cfg.infer_size / orig_w
            scale_y = self.cfg.infer_size / orig_h
            scale = min(scale_x, scale_y)
            pad_w = 0
            pad_h = 0

        if self.cfg.normalize:
            output = output.astype(np.float32) / 255.0

        return PreprocessResult(
            frame=output,
            scale=scale,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_size=(orig_h, orig_w),
        )

    def adjust_boxes(
        self,
        boxes_xyxy: np.ndarray,
        result: PreprocessResult,
    ) -> np.ndarray:
        """
        Ajusta bounding boxes do espaço processado para o
        espaço da imagem original.
        """
        boxes = boxes_xyxy.copy().astype(float)

        boxes[:, [0, 2]] -= result.pad_w
        boxes[:, [1, 3]] -= result.pad_h

        boxes[:, [0, 2]] /= result.scale_x
        boxes[:, [1, 3]] /= result.scale_y

        return boxes

    def _apply_clahe(
        self,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        """Aplica CLAHE somente ao canal de luminância."""

        if self.cfg.clahe_space == "lab":
            lab = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2LAB,
            )
            lightness, channel_a, channel_b = cv2.split(lab)
            lightness_clahe = self._clahe.apply(lightness)

            merged = cv2.merge(
                [
                    lightness_clahe,
                    channel_a,
                    channel_b,
                ]
            )

            return cv2.cvtColor(
                merged,
                cv2.COLOR_LAB2BGR,
            )

        hsv = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2HSV,
        )
        hue, saturation, value = cv2.split(hsv)
        value_clahe = self._clahe.apply(value)

        merged = cv2.merge(
            [
                hue,
                saturation,
                value_clahe,
            ]
        )

        return cv2.cvtColor(
            merged,
            cv2.COLOR_HSV2BGR,
        )


CONFIG_DEFAULT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
    gaussian_blur=False,
    clahe=False,
)


CONFIG_LOW_LIGHT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
    clahe=True,
    clahe_clip=2.0,
    clahe_tile=8,
    clahe_space="lab",
)


CONFIG_HIGH_QUALITY = PreprocessConfig(
    infer_size=640,
    convert_rgb=True,
    use_letterbox=True,
    gaussian_blur=False,
    clahe=False,
)
