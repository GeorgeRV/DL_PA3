from __future__ import annotations

import argparse
import json
import random
import warnings
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.dataset import ImagePathIndex, TrapDataset, SquarePad, load_class_mapping, make_transforms
from src.model import build_model, count_parameters

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class UnlabeledTwoViews(Dataset):
    """Imagens de images/train que não pertencem ao CSV rotulado do orçamento."""
    def __init__(self, image_root: Path, labeled_ids: List[str], image_size: int):
        if not image_root.exists():
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {image_root}")
        labeled = {normalise_image_id(x) for x in labeled_ids}
        paths = sorted(
            p for p in image_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.items = [(p.stem, p) for p in paths if normalise_image_id(p.stem) not in labeled]
        if not self.items:
            raise RuntimeError("Nenhuma imagem não rotulada encontrada em images/train.")

        base = [SquarePad(fill=(0, 0, 0)), transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
        self.weak = transforms.Compose(base + [
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.strong = transforms.Compose(base + [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.3, 0.08)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, path = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.weak(image), self.strong(image), image_id


def normalise_image_id(value) -> str:
    token = str(value).strip()
    return Path(token).stem if Path(token).suffix.lower() in IMAGE_EXTENSIONS else token


def normalise_label(value) -> str:
    token = str(value).strip().casefold()
    if re.fullmatch(r"-?\d+\.0+", token):
        token = token.split(".", 1)[0]
    return token


def read_and_validate_labels(csv_path: Path, class_names: List[str], aliases: Dict[str, int],
                             expected_budget: int | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV de rótulos não encontrado: {csv_path}")
    frame = pd.read_csv(csv_path, dtype={"image_id": str})
    required = {"image_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} precisa conter as colunas {sorted(required)}.")
    frame = frame[["image_id", "label"]].copy()
    frame["image_id"] = frame["image_id"].astype(str).str.strip()
    if frame["image_id"].duplicated().any():
        duplicated = frame.loc[frame.image_id.duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"IDs duplicados em {csv_path}: {duplicated}")
    frame["class_idx"] = frame["label"].map(lambda x: aliases.get(normalise_label(x), -1))
    if (frame.class_idx < 0).any():
        bad = frame.loc[frame.class_idx < 0, "label"].unique().tolist()
        raise ValueError(f"Rótulos não reconhecidos em {csv_path}: {bad}")
    counts = frame.groupby("class_idx").size().reindex(range(len(class_names)), fill_value=0)
    if expected_budget is not None and not (counts == expected_budget).all():
        raise ValueError(
            f"{csv_path} deveria ter exatamente {expected_budget} imagens por classe. "
            f"Contagens: {counts.tolist()}"
        )
    return frame


def make_budget_csvs(args, class_names, aliases, output_dir: Path) -> Dict[Tuple[int, int], Path]:
    """Usa CSVs rotulos_{1,5,20}.csv ou amostra conjuntos aninhados de uma fonte >=20/classe."""
    labels_dir = args.data_dir / "labels"
    seeds = {1: [args.seed], 5: args.seeds_5, 20: [args.seed]}
    output: Dict[Tuple[int, int], Path] = {}

    if args.source_labels_csv is not None:
        source = read_and_validate_labels(args.source_labels_csv, class_names, aliases)
        counts = source.groupby("class_idx").size().reindex(range(len(class_names)), fill_value=0)
        if (counts < 20).any():
            raise ValueError("O CSV fonte precisa ter ao menos 20 exemplos em cada classe. "
                             f"Contagens: {counts.tolist()}")
        for seed in sorted(set(sum(seeds.values(), []))):
            rng = np.random.default_rng(seed)
            selected_by_budget = {1: [], 5: [], 20: []}
            for class_idx in range(len(class_names)):
                group = source[source.class_idx == class_idx]
                # Uma permutação por classe: os conjuntos 1, 5 e 20 são aninhados.
                chosen = rng.permutation(group.index.to_numpy())[:20]
                for budget in (1, 5, 20):
                    selected_by_budget[budget].extend(chosen[:budget].tolist())
            for budget in (1, 5, 20):
                if seed not in seeds[budget]:
                    continue
                subset = source.loc[selected_by_budget[budget], ["image_id", "label"]].copy()
                path = output_dir / "label_subsets" / f"rotulos_{budget}_seed_{seed}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                subset.to_csv(path, index=False)
                read_and_validate_labels(path, class_names, aliases, budget)
                output[(budget, seed)] = path
    else:
        for budget, budget_seeds in seeds.items():
            path = labels_dir / f"rotulos_{budget}.csv"
            read_and_validate_labels(path, class_names, aliases, budget)
            for seed in budget_seeds:
                output[(budget, seed)] = path
        warnings.warn(
            "Usando rotulos_{1,5,20}.csv existentes. Nos três runs do orçamento 5, "
            "o subconjunto de imagens será o mesmo; a seed varia a inicialização, "
            "embaralhamento e augmentações. Para variar também a amostra, informe "
            "--source-labels-csv com pelo menos 20 rótulos por classe.", RuntimeWarning)
    return output


def build_data(args, csv_path: Path, class_names, aliases, seed: int):
    train_tf, eval_tf = make_transforms(args.image_size)
    image_index = ImagePathIndex(args.data_dir / "images")
    train_ds = TrapDataset(csv_path, image_index, aliases, "train", train_tf)
    val_ds = TrapDataset(args.data_dir / "labels" / "val.csv", image_index, aliases, "val", eval_tf)
    # test_public não é carregado nem consultado durante o treinamento.
    # O conjunto não rotulado é sempre criado a partir do CSV específico do orçamento.
    labeled_ids = pd.read_csv(csv_path, dtype={"image_id": str})["image_id"].astype(str).tolist()
    unlabeled_ds = UnlabeledTwoViews(args.data_dir / "images" / "train", labeled_ids, args.image_size)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = args.device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=False, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    unl_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size * args.mu, shuffle=True,
                            num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    return train_ds, train_loader, val_loader, unl_loader, None


@torch.no_grad()
def evaluate(model, loader, device, nclasses: int, per_class: bool = False):
    model.eval()
    y_true, y_pred = [], []
    loss_total, n_total = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels_device)
        predictions = logits.argmax(dim=1)
        n = labels_device.size(0)
        loss_total += float(loss.item()) * n
        n_total += n
        y_true.extend(labels.tolist())
        y_pred.extend(predictions.cpu().tolist())
    labels_range = list(range(nclasses))
    result = {
        "loss": loss_total / max(n_total, 1),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels_range,
                                   average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if per_class:
        scores = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
        result["f1_by_class"] = scores.tolist()
    return result


def make_model(class_names, device):
    model = build_model("resnet18", num_classes=len(class_names), pretrained=True).to(device)
    nparams = count_parameters(model)
    if nparams > 12_000_000:
        raise RuntimeError(f"ResNet-18 excede o limite de 12 milhões: {nparams:,}")
    return model, nparams


def train_experiment(method: str, args, budget: int, seed: int, csv_path: Path,
                     class_names, aliases, output_dir: Path):
    seed_everything(seed)
    _, train_loader, val_loader, unl_loader, _ = build_data(
        args, csv_path, class_names, aliases, seed)
    model, nparams = make_model(class_names, args.device)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")],
         "lr": args.lr_backbone},
        {"params": model.fc.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    ce = nn.CrossEntropyLoss()
    ce_none = nn.CrossEntropyLoss(reduction="none")
    # Prior uniforme coerente com os subconjuntos estratificados (mesmo nº de rótulos/classe).
    target_prior = torch.full((len(class_names),), 1.0 / len(class_names), device=args.device)
    marginal = target_prior.clone()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    history, best_f1, best_val_accuracy, best_val_loss, stale = [], -1.0, float("nan"), float("nan"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = sup_total = unsup_total = accepted_total = 0.0
        steps = 0
        if method == "supervised":
            for images, labels, _ in train_loader:
                images = images.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = ce(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                sup_total += float(loss.item())
                steps += 1
        else:
            labeled_iter = iter(train_loader)
            # FixMatch usa os batches não rotulados como relógio; o loader rotulado é reciclado.
            for weak, strong, _ in unl_loader:
                try:
                    images_l, labels_l, _ = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(train_loader)
                    images_l, labels_l, _ = next(labeled_iter)
                images_l = images_l.to(args.device, non_blocking=True)
                labels_l = labels_l.to(args.device, non_blocking=True)
                weak = weak.to(args.device, non_blocking=True)
                strong = strong.to(args.device, non_blocking=True)

                logits_l = model(images_l)
                loss_x = ce(logits_l, labels_l)
                with torch.no_grad():
                    model.eval()
                    probabilities = torch.softmax(model(weak), dim=1)
                    if method == "fixmatch_aligned":
                        marginal = 0.999 * marginal + 0.001 * probabilities.mean(dim=0)
                        probabilities = probabilities * (target_prior / marginal.clamp_min(1e-6))
                        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    confidence, pseudo_labels = probabilities.max(dim=1)
                    mask = (confidence >= args.threshold).float()
                    model.train()

                logits_u = model(strong)
                loss_u = (ce_none(logits_u, pseudo_labels) * mask).mean()
                loss = loss_x + args.lambda_u * loss_u
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                sup_total += float(loss_x.item())
                unsup_total += float(loss_u.item())
                accepted_total += float(mask.mean().item())
                steps += 1

        val = evaluate(model, val_loader, args.device, len(class_names))
        scheduler.step(val["f1_macro"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "train_supervised_loss": sup_total / max(steps, 1),
            "train_unsupervised_loss": unsup_total / max(steps, 1),
            "pseudo_label_acceptance": accepted_total / max(steps, 1),
            "val_loss": val["loss"],
            "val_f1_macro": val["f1_macro"],
            "val_accuracy": val["accuracy"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"[{method}] budget={budget} seed={seed} epoch={epoch:03d} "
              f"val_F1={val['f1_macro']:.4f} val_acc={val['accuracy']:.4f} "
              f"aceitação={row['pseudo_label_acceptance']:.3f}")

        # O checkpoint é escolhido exclusivamente pela validação.
        if val["f1_macro"] > best_f1 + 1e-8:
            best_f1, best_val_accuracy, best_val_loss, stale = (
                val["f1_macro"], val["accuracy"], val["loss"], 0)
            torch.save({
                "state_dict": model.state_dict(), "class_names": list(class_names),
                "method": method, "budget": budget, "seed": seed,
                "epoch": epoch, "best_val_f1_macro": best_f1,
                "best_val_accuracy": best_val_accuracy, "best_val_loss": best_val_loss,
                "image_size": args.image_size, "num_parameters": nparams,
                "threshold": args.threshold if method != "supervised" else None,
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping: {args.patience} épocas sem melhora na validação.")
            break

    plot_history(pd.DataFrame(history), output_dir / "training_curves.png", method, budget, seed)
    return {
        "method": method, "budget": budget, "seed": seed,
        "best_val_f1_macro": best_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "checkpoint": str(checkpoint_path),
    }


def plot_history(history: pd.DataFrame, path: Path, method: str, budget: int, seed: int):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_supervised_loss, label="Loss supervisionada")
    if method != "supervised":
        axes[0].plot(history.epoch, history.train_unsupervised_loss, label="Loss não supervisionada")
    axes[0].set_title("Loss de treinamento")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history.epoch, history.val_f1_macro, label="F1 Macro validação")
    axes[1].set_title("Validação")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{method} | {budget} rótulos/classe | seed={seed}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def final_test_report(args, experiments, class_names, aliases):
    """Avalia test_public somente depois de encerrados todos os treinamentos."""
    test_rows, class_rows = [], []
    image_index = ImagePathIndex(args.data_dir / "images")
    _, eval_tf = make_transforms(args.image_size)
    test_ds = TrapDataset(args.data_dir / "labels" / "test_public.csv", image_index,
                          aliases, "test_public", eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.device.type == "cuda")
    for exp in experiments:
        checkpoint = torch.load(exp["checkpoint"], map_location=args.device)
        if list(checkpoint["class_names"]) != list(class_names):
            raise ValueError(f"Ordem de classes incompatível em {exp['checkpoint']}")
        model = build_model("resnet18", num_classes=len(class_names), pretrained=False).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate(model, test_loader, args.device, len(class_names), per_class=True)
        row = {"method": exp["method"], "budget": exp["budget"], "seed": exp["seed"],
               "test_f1_macro": metrics["f1_macro"], "test_accuracy": metrics["accuracy"],
               "checkpoint_epoch": checkpoint["epoch"],
               "best_val_f1_macro": checkpoint["best_val_f1_macro"]}
        test_rows.append(row)
        for name, score in zip(class_names, metrics["f1_by_class"]):
            class_rows.append({"method": exp["method"], "budget": exp["budget"],
                               "seed": exp["seed"], "class": name, "test_f1": score})
        del model
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(args.output_dir / "test_public_final_por_execucao.csv", index=False)
    pd.DataFrame(class_rows).to_csv(args.output_dir / "test_public_f1_por_classe.csv", index=False)
    return test_df


def save_summary(experiments, args):
    val_df = pd.DataFrame(experiments)
    val_df.to_csv(args.output_dir / "resultados_validacao_por_execucao.csv", index=False)
    # Mantém uma linha por execução, como no script original do grupo.
    val_df.rename(columns={"method": "metodo", "budget": "orcamento",
                           "best_val_f1_macro": "f1_macro_validacao",
                           "best_val_accuracy": "acuracia_validacao"}) \
        .to_csv(args.output_dir / "curva_orcamento.csv", index=False)
    summary = val_df.groupby(["method", "budget"], as_index=False).agg(
        f1_val_mean=("best_val_f1_macro", "mean"),
        f1_val_std=("best_val_f1_macro", "std"),
        accuracy_val_mean=("best_val_accuracy", "mean"),
        accuracy_val_std=("best_val_accuracy", "std"),
    )
    summary.to_csv(args.output_dir / "curva_orcamento_validacao_resumo.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, label in [("supervised", "Supervisionado (piso)"),
                          ("fixmatch_aligned", "FixMatch aligned")]:
        d = summary[summary.method == method].sort_values("budget")
        if len(d):
            ax.errorbar(d.budget, d.f1_val_mean, yerr=d.f1_val_std.fillna(0),
                        marker="o", capsize=4, label=label)
    ax.set_xscale("log", base=10)
    ax.set_xticks([1, 5, 20], labels=["1", "5", "20"])
    ax.set_xlabel("Orçamento (rótulos por classe)")
    ax.set_ylabel("F1 Macro de validação")
    ax.set_title("Curva de orçamento — seleção pela validação")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "curva_orcamento_validacao.png", dpi=180)
    plt.close(fig)

    budget5 = val_df[val_df.budget == 5]
    if len(budget5):
        # F1 por classe de validação no checkpoint selecionado, por seed e método.
        class_rows = []
        image_index = ImagePathIndex(args.data_dir / "images")
        _, eval_tf = make_transforms(args.image_size)
        val_ds = TrapDataset(args.data_dir / "labels" / "val.csv", image_index,
                             load_class_mapping(args.data_dir / "classes.csv", [
                                 args.data_dir / "labels" / "val.csv"])[1], "val", eval_tf)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.num_workers)
        for exp in budget5.to_dict("records"):
            ckpt = torch.load(exp["checkpoint"], map_location=args.device)
            model = build_model("resnet18", num_classes=len(ckpt["class_names"]), pretrained=False).to(args.device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval(); ys, ps = [], []
            with torch.no_grad():
                for x, y, _ in val_loader:
                    pred = model(x.to(args.device)).argmax(1).cpu().tolist()
                    ys.extend(y.tolist()); ps.extend(pred)
            scores = f1_score(ys, ps, labels=list(range(len(class_names))), average=None, zero_division=0)
            for name, score in zip(class_names, scores):
                class_rows.append({"method": exp["method"], "seed": exp["seed"],
                                   "class": name, "val_f1": float(score)})
            del model
        class_df = pd.DataFrame(class_rows)
        class_df.to_csv(args.output_dir / "f1_por_classe_orcamento_5_validacao.csv", index=False)
        mean_df = class_df.groupby(["method", "class"], as_index=False).agg(
            val_f1_mean=("val_f1", "mean"), val_f1_std=("val_f1", "std"))
        mean_df.to_csv(args.output_dir / "f1_por_classe_orcamento_5_resumo.csv", index=False)
        pivot = mean_df.pivot(index="class", columns="method", values="val_f1_mean").reset_index()
        if "supervised" in pivot.columns and "fixmatch_aligned" in pivot.columns:
            pivot["ganho_fixmatch_menos_piso"] = pivot["fixmatch_aligned"] - pivot["supervised"]
        pivot.to_csv(args.output_dir / "ganhos_por_classe_orcamento_5_validacao.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="PA3 Parte 2: curva de orçamento com piso e FixMatch aligned")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pa3_dados"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/parte2"))
    parser.add_argument("--source-labels-csv", type=Path, default=None,
                        help="CSV com >=20 rótulos por classe para gerar subconjuntos aninhados por seed. "
                             "Se omitido, usa labels/rotulos_1.csv, rotulos_5.csv e rotulos_20.csv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds-5", type=int, nargs="+", default=[42, 100, 200])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--skip-test", action="store_true",
                        help="Não avalia test_public; use apenas se quiser rodar/depurar os treinos.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = choose_device(args.device)
    if args.batch_size < 1 or args.mu < 1:
        raise ValueError("batch-size e mu precisam ser positivos.")
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold deve estar no intervalo (0, 1].")
    args.evaluate_test = not args.skip_test
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = args.data_dir / "labels"
    annotation_paths = [labels_dir / "val.csv"]
    if args.source_labels_csv is not None:
        annotation_paths.append(args.source_labels_csv)
    else:
        annotation_paths.extend(labels_dir / f"rotulos_{b}.csv" for b in (1, 5, 20))
    class_names, aliases = load_class_mapping(args.data_dir / "classes.csv", annotation_paths)
    if len(class_names) != 21:
        raise ValueError(f"Esperadas 21 classes em classes.csv; encontradas {len(class_names)}.")

    subsets = make_budget_csvs(args, class_names, aliases, args.output_dir)
    print(f"Dispositivo: {args.device}")
    print(f"Classes: {len(class_names)} | Experimentos: {len(subsets) * 2}")
    print("Seleção de checkpoint: exclusivamente F1 Macro de val.csv.")
    print("test_public: avaliação final após o treinamento, sem seleção de modelos.")

    experiments = []
    for (budget, seed), csv_path in subsets.items():
        for method in ("supervised", "fixmatch_aligned"):
            output_dir = args.output_dir / method / f"budget_{budget}" / f"seed_{seed}"
            result = train_experiment(method, args, budget, seed, csv_path,
                                      class_names, aliases, output_dir)
            experiments.append(result)
            pd.DataFrame(experiments).to_csv(
                args.output_dir / "resultados_validacao_parcial.csv", index=False)

    save_summary(experiments, args)
    if args.evaluate_test:
        test_df = final_test_report(args, experiments, class_names, aliases)
        test_summary = test_df.groupby(["method", "budget"], as_index=False).agg(
            test_f1_mean=("test_f1_macro", "mean"), test_f1_std=("test_f1_macro", "std"),
            test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"))
        test_summary.to_csv(args.output_dir / "curva_orcamento_test_public_final.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for method, label in [("supervised", "Supervisionado (piso)"),
                              ("fixmatch_aligned", "FixMatch aligned")]:
            d = test_summary[test_summary.method == method].sort_values("budget")
            if len(d):
                ax.errorbar(d.budget, d.test_f1_mean, yerr=d.test_f1_std.fillna(0),
                            marker="o", capsize=4, label=label)
        ax.set_xscale("log", base=10)
        ax.set_xticks([1, 5, 20], labels=["1", "5", "20"])
        ax.set_xlabel("Orçamento (rótulos por classe)")
        ax.set_ylabel("F1 Macro no test_public")
        ax.set_title("Curva final de orçamento — test_public")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "curva_orcamento_test_public_final.png", dpi=180)
        plt.close(fig)
        print("\nResumo final test_public (somente relatório):")
        print(test_summary.to_string(index=False))

    print("\nExperimentos concluídos. Resultados em:", args.output_dir)
    print(pd.DataFrame(experiments).to_string(index=False))


if __name__ == "__main__":
    main()
