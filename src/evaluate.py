from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import load_datasets
from src.model import build_model, count_parameters
from src.metrics import classification_metrics, save_confusion_matrix, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia checkpoints no test_public.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pa3_dados"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/parte0"))
    parser.add_argument("--models", choices=["both", "cnn_scratch", "resnet18"], default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    class_names, _, _, _, test_ds = load_datasets(args.data_dir)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()
    selected_models = ["cnn_scratch", "resnet18"] if args.models == "both" else [args.models]

    for model_name in selected_models:
        checkpoint_path = args.output_dir / model_name / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado: {checkpoint_path}. Execute primeiro o treinamento."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_classes = checkpoint.get("class_names")
        if saved_classes is not None and list(saved_classes) != list(class_names):
            raise ValueError("A ordem das classes no checkpoint não corresponde a classes.csv.")

        model = build_model(model_name, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        parameter_count = count_parameters(model)

        y_true, y_pred = [], []
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(device)
                labels_device = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels_device)
                predictions = logits.argmax(dim=1)
                batch_size = labels_device.size(0)
                total_loss += float(loss.item()) * batch_size
                total += batch_size
                y_true.extend(labels.tolist())
                y_pred.extend(predictions.cpu().tolist())

        metrics = classification_metrics(y_true, y_pred, class_names)
        metrics["cross_entropy_loss"] = total_loss / max(total, 1)
        metrics["num_parameters"] = parameter_count
        metrics["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
        metrics["checkpoint_val_f1_macro"] = float(checkpoint.get("best_val_f1_macro", float("nan")))
        metrics["test_split"] = "test_public"

        result_dir = args.output_dir / model_name
        save_metrics(metrics, result_dir / "test_public_metrics.json")
        save_confusion_matrix(y_true, y_pred, class_names, result_dir, prefix="test_public")

        print(f"\n=== {model_name} — test_public ===")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")
        print(f"Acurácia: {metrics['accuracy']:.4f}")
        print(f"Cross-entropy: {metrics['cross_entropy_loss']:.4f}")
        print(f"Parâmetros: {parameter_count:,}")
        print(f"Resultados: {result_dir}")


if __name__ == "__main__":
    main()
