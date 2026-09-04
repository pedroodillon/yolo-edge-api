"""
preprocessing/utils/evaluate.py

Avalia o mAP@0.5 de um pipeline de pré-processamento no
dataset epi-v1.
"""

import shutil
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


DATASET_YAML = "dataset/exports/epi-v1/data.yaml"
MODEL_PATH = "models/yolov8n.pt"


def evaluate_pipeline(
    preprocess_fn: Callable | None = None,
    label: str = "baseline",
    split: str = "val",
    verbose: bool = False,
) -> dict:
    """
    Avalia o mAP@0.5 do modelo com uma função de
    pré-processamento opcional.

    Args:
        preprocess_fn: função que recebe um frame BGR NumPy
            e retorna um frame transformado. Se None, usa o
            comportamento padrão da Ultralytics.
        label: nome usado para identificar o experimento.
        split: split do dataset a avaliar ("val" ou "test").
        verbose: controla a saída detalhada da validação.

    Returns:
        Dicionário contendo map50, map50_95 e o tempo médio
        de pré-processamento em milissegundos.

    Nota:
        O model.val() relê as imagens do disco e aplica sua
        própria conversão de canais. Portanto, o efeito visual
        do experimento E1 será conferido separadamente.
    """
    model = YOLO(MODEL_PATH)

    if preprocess_fn is None:
        metrics = model.val(
            data=DATASET_YAML,
            split=split,
            verbose=verbose,
        )
        preproc_ms = 0.0

    else:
        split_dirname = {
            "val": "valid",
            "test": "test",
            "train": "train",
        }.get(split, split)

        dataset_dir = Path(DATASET_YAML).parent
        src_images_dir = dataset_dir / split_dirname / "images"
        src_labels_dir = dataset_dir / split_dirname / "labels"

        images = sorted(src_images_dir.glob("*.jpg"))
        images += sorted(src_images_dir.glob("*.png"))

        safe_label = "".join(
            char if char.isalnum() else "_" for char in label
        )
        tmp_root = (
            Path("preprocessing/outputs/_tmp_eval") / safe_label
        )

        if tmp_root.exists():
            shutil.rmtree(tmp_root)

        tmp_images_dir = tmp_root / "images"
        tmp_labels_dir = tmp_root / "labels"
        tmp_images_dir.mkdir(parents=True, exist_ok=True)
        tmp_labels_dir.mkdir(parents=True, exist_ok=True)

        preproc_times = []

        for img_path in images:
            frame = cv2.imread(str(img_path))

            t0 = time.perf_counter()
            frame_proc = preprocess_fn(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            preproc_times.append(elapsed_ms)

            cv2.imwrite(
                str(tmp_images_dir / img_path.name),
                frame_proc,
            )

            label_src = src_labels_dir / f"{img_path.stem}.txt"
            if label_src.exists():
                shutil.copy(
                    label_src,
                    tmp_labels_dir / label_src.name,
                )

        with open(DATASET_YAML, encoding="utf-8") as file:
            base_cfg = yaml.safe_load(file)

        tmp_yaml_cfg = {
            "path": str(tmp_root.resolve()),
            "train": "images",
            "val": "images",
            "test": "images",
            "names": base_cfg["names"],
        }

        tmp_yaml = tmp_root / "data.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as file:
            yaml.safe_dump(tmp_yaml_cfg, file)

        metrics = model.val(
            data=str(tmp_yaml),
            split="val",
            verbose=verbose,
        )

        preproc_ms = (
            float(np.mean(preproc_times))
            if preproc_times
            else 0.0
        )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print(
        f"[{label:30s}] "
        f"mAP@0.5={map50:.4f} "
        f"mAP@0.5:0.95={map50_95:.4f} "
        f"preproc={preproc_ms:.1f}ms"
    )

    return {
        "label": label,
        "map50": map50,
        "map50_95": map50_95,
        "preproc_ms": preproc_ms,
    }
