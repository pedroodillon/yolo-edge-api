"""
Testes unitários do módulo preprocessing/preprocessor.py.
"""

import numpy as np

from preprocessing.preprocessor import (
    PreprocessConfig,
    Preprocessor,
)


def make_frame(
    height=480,
    width=640,
    dtype=np.uint8,
):
    return np.random.randint(
        0,
        255,
        (height, width, 3),
        dtype=dtype,
    )


class TestPreprocessorOutput:
    def test_output_shape_letterbox(self):
        pp = Preprocessor(
            PreprocessConfig(infer_size=416)
        )
        result = pp.process(make_frame())

        assert result.frame.shape == (416, 416, 3)

    def test_output_dtype_uint8(self):
        pp = Preprocessor(
            PreprocessConfig(normalize=False)
        )
        result = pp.process(make_frame())

        assert result.frame.dtype == np.uint8

    def test_output_dtype_float32_when_normalized(self):
        pp = Preprocessor(
            PreprocessConfig(normalize=True)
        )
        result = pp.process(make_frame())

        assert result.frame.dtype == np.float32
        assert result.frame.max() <= 1.0

    def test_scale_and_padding_set(self):
        pp = Preprocessor(
            PreprocessConfig(
                infer_size=416,
                use_letterbox=True,
            )
        )
        result = pp.process(
            make_frame(height=480, width=640)
        )

        assert result.scale > 0
        assert result.orig_size == (480, 640)

    def test_letterbox_padding_symmetric(self):
        pp = Preprocessor(
            PreprocessConfig(
                infer_size=416,
                use_letterbox=True,
            )
        )
        result = pp.process(
            make_frame(height=416, width=416)
        )

        assert result.pad_w == 0
        assert result.pad_h == 0


class TestBboxAdjustment:
    def test_adjust_removes_letterbox_offset(self):
        pp = Preprocessor(
            PreprocessConfig(infer_size=416)
        )
        result = pp.process(
            make_frame(height=480, width=640)
        )

        boxes_letterboxed = np.array(
            [[10, 50, 100, 200]],
            dtype=float,
        )

        boxes_original = pp.adjust_boxes(
            boxes_letterboxed,
            result,
        )

        if result.pad_h > 0:
            assert (
                boxes_original[0, 1]
                < boxes_letterboxed[0, 1]
            )


class TestPreprocessorConfigs:
    def test_config_low_light_applies_clahe(self):
        from preprocessing.preprocessor import (
            CONFIG_LOW_LIGHT,
        )

        pp = Preprocessor(CONFIG_LOW_LIGHT)
        result = pp.process(make_frame())

        assert result.frame.shape[2] == 3

    def test_config_default_no_filter(self):
        from preprocessing.preprocessor import CONFIG_DEFAULT

        pp = Preprocessor(CONFIG_DEFAULT)

        assert not pp.cfg.gaussian_blur
        assert not pp.cfg.median_blur
        assert not pp.cfg.clahe


class TestNonUniformScale:
    def test_adjust_boxes_without_letterbox_uses_separate_axis_scales(
        self,
    ):
        pp = Preprocessor(
            PreprocessConfig(
                infer_size=416,
                use_letterbox=False,
            )
        )
        result = pp.process(
            make_frame(height=480, width=640)
        )

        assert result.scale_x != result.scale_y

        boxes_resized = np.array(
            [[0, 0, 416, 416]],
            dtype=float,
        )

        boxes_original = pp.adjust_boxes(
            boxes_resized,
            result,
        )

        assert abs(boxes_original[0, 2] - 640) < 1e-6
        assert abs(boxes_original[0, 3] - 480) < 1e-6
