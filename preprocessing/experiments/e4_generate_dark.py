"""
Gera uma versão escurecida do split de validação para o
experimento E4.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path("dataset/exports/epi-v1/valid")
DESTINATION = Path("dataset/exports/epi-v1-dark/valid")
GAMMA = 2.2


def main():
    destination_images = DESTINATION / "images"
    destination_labels = DESTINATION / "labels"

    destination_images.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    for label_path in (SOURCE / "labels").glob("*.txt"):
        shutil.copy(
            label_path,
            destination_labels / label_path.name,
        )

    gamma_table = np.array(
        [
            ((value / 255.0) ** GAMMA) * 255
            for value in range(256)
        ],
        dtype=np.uint8,
    )

    generated_images = 0

    for image_path in (SOURCE / "images").glob("*.jpg"):
        image = cv2.imread(str(image_path))
        dark_image = cv2.LUT(image, gamma_table)

        cv2.imwrite(
            str(destination_images / image_path.name),
            dark_image,
        )

        generated_images += 1

    copied_labels = len(
        list(destination_labels.glob("*.txt"))
    )

    print(f"Gamma aplicado: {GAMMA}")
    print(f"Imagens escurecidas: {generated_images}")
    print(f"Labels copiados: {copied_labels}")
    print(f"Destino: {DESTINATION.parent}")


if __name__ == "__main__":
    main()
