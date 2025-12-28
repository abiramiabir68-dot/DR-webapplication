from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, ds, class_names: list[str]) -> dict:
    y_true = []
    y_pred = []

    for batch in ds:
        if len(batch) == 3:
            x, y, _sw = batch
        else:
            x, y = batch

        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=-1).tolist())
        y_pred.extend(np.argmax(p, axis=-1).tolist())

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=False
    )
    cm = confusion_matrix(y_true, y_pred)

    return {"report_text": report_text, "confusion_matrix": cm}
