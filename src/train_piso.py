from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.dataset import load_datasets
from src.model import build_model, count_parameters
from src.metrics import save_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prioriza reprodutibilidade; pode reduzir um pouco a velocidade na GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: nn.Module, model_name: str) -> torch.optim.Optimizer:
    if model_name == "resnet18":
        # Fine-tuning integral: backbone e nova camada final são atualizados.
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
        head_params = list(model.fc.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 1e-4},
                {"params": head_params, "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_true, all_pred = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels_device)
        predictions = logits.argmax(dim=1)

        batch_size = labels_device.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(predictions.cpu().numpy().tolist())

    class_labels = list(range(num_classes))
    return {
        "loss": total_loss / max(total_examples, 1),
        "f1_macro": float(f1_score(all_true, all_pred, labels=class_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(all_true, all_pred)),
    }


def train_model(
    model_name: str,
    class_names,
    train_dataset,
    train_eval_dataset,
    val_dataset,
    output_dir: Path,
    args,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    use_pretrained = model_name == "resnet18"
    try:
        model = build_model(model_name, num_classes=len(class_names), pretrained=use_pretrained)
    except Exception as exc:
        if use_pretrained:
            raise RuntimeError(
                "Não foi possível carregar os pesos ImageNet da ResNet-18. "
                "Na primeira execução, o torchvision precisa baixar os pesos; "
                "verifique a conexão e tente novamente."
            ) from exc
        raise

    model = model.to(device)
    parameter_count = count_parameters(model)
    print(f"\n[{model_name}] Parâmetros treináveis/totais: {parameter_count:,}")
    if parameter_count > 12_000_000:
        raise RuntimeError(f"Modelo excede o limite de 12 milhões: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, model_name)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )

    history = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_stats = evaluate(model, train_eval_loader, criterion, device, len(class_names))
        val_stats = evaluate(model, val_loader, criterion, device, len(class_names))
        scheduler.step(val_stats["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_stats["loss"],
            "train_f1_macro": train_stats["f1_macro"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_f1_macro": val_stats["f1_macro"],
            "val_accuracy": val_stats["accuracy"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print(
            f"[{model_name}] Época {epoch:03d}/{args.epochs} | "
            f"loss treino {train_loss:.4f} | loss val {val_stats['loss']:.4f} | "
            f"F1 treino {train_stats['f1_macro']:.4f} | F1 val {val_stats['f1_macro']:.4f}"
        )

        is_better = (
            val_stats["f1_macro"] > best_f1 + 1e-8
            or (
                abs(val_stats["f1_macro"] - best_f1) <= 1e-8
                and val_stats["loss"] < best_val_loss
            )
        )
        if is_better:
            best_f1 = val_stats["f1_macro"]
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "num_classes": len(class_names),
                    "class_names": list(class_names),
                    "state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_val_f1_macro": best_f1,
                    "best_val_loss": best_val_loss,
                    "num_parameters": parameter_count,
                    "seed": args.seed,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"[{model_name}] Early stopping: F1 de validação não melhorou por "
                f"{args.patience} épocas."
            )
            break

    history_path = output_dir / "history.csv"
    save_training_curves(
        history_path,
        output_dir / "training_curves.png",
        title=f"Parte 0 — {model_name}",
    )
    print(
        f"[{model_name}] Melhor época: {best_epoch}; "
        f"F1 macro de validação: {best_f1:.4f}; checkpoint: {checkpoint_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Treina os dois pisos supervisionados da Parte 0.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pa3_dados"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/parte0"))
    parser.add_argument("--models", choices=["both", "cnn_scratch", "resnet18"], default="both")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda ou mps")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Dispositivo: {device}")
    print(f"Seed: {args.seed}")

    class_names, train_ds, train_eval_ds, val_ds, _ = load_datasets(
        args.data_dir, image_size=args.image_size, include_test=False
    )
    print(f"Classes: {len(class_names)}")
    print(f"Treino rotulado: {len(train_ds)} imagens")
    print(f"Validação: {len(val_ds)} imagens")
    print("O test_public não será usado no treinamento nem na seleção de época.")

    selected_models = ["cnn_scratch", "resnet18"] if args.models == "both" else [args.models]
    for model_name in selected_models:
        train_model(
            model_name=model_name,
            class_names=class_names,
            train_dataset=train_ds,
            train_eval_dataset=train_eval_ds,
            val_dataset=val_ds,
            output_dir=args.output_dir / model_name,
            args=args,
            device=device,
        )

    print("\nTreinamento concluído. Avalie os melhores checkpoints no test_public apenas agora.")


if __name__ == "__main__":
    main()
