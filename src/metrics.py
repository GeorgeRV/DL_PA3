from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def classification_metrics(y_true, y_pred, class_names: Sequence[str]) -> Dict:
    labels = list(range(len(class_names)))
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_examples": int(len(y_true)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(y_true, y_pred, class_names: Sequence[str], output_dir: Path, prefix: str = "test_public"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv", index=True, index_label="true_label"
    )

    # A matriz 21x21 é densa; rótulos menores tornam a figura legível nos slides.
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Classe prevista",
        ylabel="Classe verdadeira",
        title="Matriz de confusão — test_public",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_csv: Path, output_path: Path, title: str):
    history = pd.read_csv(history_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_eval_loss"], label="Treino")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validação")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_f1_macro"], label="Treino")
    axes[1].plot(history["epoch"], history["val_f1_macro"], label="Validação")
    axes[1].set_title("F1 macro")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("F1 macro")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: Dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
