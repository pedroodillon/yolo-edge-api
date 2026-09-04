"""
Mede o baseline de validação antes dos experimentos de
pré-processamento.
"""

import sys

sys.path.insert(0, ".")

from preprocessing.utils.evaluate import evaluate_pipeline

baseline = evaluate_pipeline(
    preprocess_fn=None,
    label="baseline (sem preproc)",
)

print(f"\nBaseline mAP@0.5 = {baseline['map50']:.4f}")
print("Anote este valor: ele será a referência dos experimentos.")
