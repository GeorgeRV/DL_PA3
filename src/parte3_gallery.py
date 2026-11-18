from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.dataset import (
    ImagePathIndex,
    SquarePad,
    TrapDataset,
    load_class_mapping,
    make_transforms,
)
from src.model import build_model


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: Iterable[Tensor], output: Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module: nn.Module, _grad_input: Iterable[Tensor], grad_output: Iterable[Tensor]) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, logits: Tensor, class_indices: Tensor) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM não conseguiu registrar ativações e gradientes.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cams = []
        for row in range(cam.shape[0]):
            score = cam[row]
            score -= score.min()
            denom = score.max()
            if denom > 0:
                score = score / denom
            cams.append(score.cpu().numpy())
        return np.asarray(cams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parte 3: galeria de erros e acertos com mapas Grad-CAM."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pa3_dados"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/parte1/fixmatch_aligned/best_model.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/parte3"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-errors", type=int, default=3)
    parser.add_argument("--num-correct", type=int, default=5)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def saliency_diagnostics(cam: np.ndarray) -> Dict[str, float]:
    height, width = cam.shape
    yy, xx = np.mgrid[0:height, 0:width]
    total = float(cam.sum())
    if total <= 0:
        cx = cy = 0.0
        concentration = 0.0
    else:
        cx = float((cam * xx).sum() / total)
        cy = float((cam * yy).sum() / total)
        concentration = float(cam[cam >= np.quantile(cam, 0.80)].sum() / total)
    return {
        "saliency_center_x": cx / max(width - 1, 1),
        "saliency_center_y": cy / max(height - 1, 1),
        "top20_fraction": concentration,
    }


def get_bbox(frame: pd.DataFrame, image_id: str) -> Optional[tuple[int, int, int, int]]:
    row = frame.loc[frame["image_id"].astype(str).str.strip() == str(image_id).strip()]
    if row.empty:
        return None

    column_sets = [
        ("x", "y", "w", "h"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]
    for columns in column_sets:
        if all(column in frame.columns for column in columns):
            values = [row.iloc[0][column] for column in columns]
            if all(pd.notna(value) for value in values):
                x1, y1, x2, y2 = [int(float(value)) for value in values]
                if columns in {column_sets[0], column_sets[2]}:
                    x2 = x1 + x2
                    y2 = y1 + y2
                return max(0, x1), max(0, y1), max(0, x2), max(0, y2)
    return None


def save_gallery(
    title: str,
    records: list[dict],
    output_path: Path,
    class_names: list[str],
) -> None:
    if not records:
        return
    columns = 5 if len(records) >= 5 else len(records)
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)

    for index, record in enumerate(records):
        ax = axes[index // columns, index % columns]
        ax.imshow(record["cam_image"])
        ax.axis("off")
        if record["is_error"]:
            caption = (
                f'{record["image_id"]}\n'
                f'True: {class_names[record["true"]]}\n'
                f'Pred: {class_names[record["pred"]]} ({record["confidence"]:.1%})'
            )
        else:
            caption = (
                f'{record["image_id"]}\n'
                f'Acerto: {class_names[record["pred"]]} ({record["confidence"]:.1%})'
            )
        ax.set_title(caption, fontsize=9)

    for index in range(len(records), rows * columns):
        axes[index // columns, index % columns].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_plain_gallery(
    title: str,
    records: list[dict],
    output_path: Path,
    class_names: list[str],
) -> None:
    if not records:
        return

    columns = 5 if len(records) >= 5 else len(records)
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)

    for index, record in enumerate(records):
        ax = axes[index // columns, index % columns]
        with Image.open(record["original_path"]) as source:
            ax.imshow(source.convert("RGB"))
        ax.axis("off")
        if record["is_error"]:
            caption = (
                f'{record["image_id"]}\n'
                f'True: {class_names[record["true"]]}\n'
                f'Pred: {class_names[record["pred"]]} ({record["confidence"]:.1%})'
            )
        else:
            caption = (
                f'{record["image_id"]}\n'
                f'Acerto: {class_names[record["pred"]]} ({record["confidence"]:.1%})'
            )
        ax.set_title(caption, fontsize=9)

    for index in range(len(records), rows * columns):
        axes[index // columns, index % columns].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_individual_originals(records: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        with Image.open(record["original_path"]) as source:
            rgb = source.convert("RGB")
            rgb.save(directory / f'{record["image_id"]}_original.png')


def write_diagnosis(
    output_path: Path,
    errors: list[dict],
    correct: list[dict],
    class_names: list[str],
) -> None:
    lines = [
        "# Diagnóstico completo da Parte 3",
        "",
        "## Metodologia",
        "",
        "- Modelo analisado: ResNet-18 treinado pelo FixMatch aligned da Parte 1.",
        "- Checkpoint: `outputs/parte1/fixmatch_aligned/best_model.pth`.",
        "- Explicabilidade: Grad-CAM aplicado à última unidade residual (`layer4[-1]`).",
        "- A previsão foi usada como classe-alvo do mapa.",
        "- Os erros exibidos são os três erros de maior confiança no `test_public`.",
        "- Os acertos são os cinco exemplos de maior confiança, diversificados por classe.",
        "- Todas as imagens foram redimensionadas pelo pipeline para 224 x 224.",
        "- Bounding boxes do animal não estão disponíveis no CSV fornecido; portanto, "
        "a análise visual compara diretamente o mapa com o animal visível na imagem.",
        "",
        "## Galerias",
        "",
        "- [Erros com Grad-CAM](galeria_erros_gradcam.png)",
        "- [Erros sem Grad-CAM](galeria_erros_sem_gradcam.png)",
        "- [Acertos com Grad-CAM](galeria_acertos_gradcam.png)",
        "- [Acertos sem Grad-CAM](galeria_acertos_sem_gradcam.png)",
        "- [Tabela detalhada](galeria_detalhes.csv)",
        "",
        "## Erros",
        "",
    ]

    error_notes = {
        "ece8e099eb242e5d": (
            "A imagem mostra um cachorro próximo de uma construção iluminada. "
            "O Grad-CAM destaca principalmente o fundo à direita e a vegetação iluminada, "
            "com menos evidência sobre o corpo do animal."
        ),
        "2a9136fc4f93cea5": (
            "A cena é dominada por vegetação e solo florestal. "
            "A saliência concentra-se no ambiente inferior e central, sugerindo que o "
            "modelo pode estar usando textura e contexto de floresta em vez de uma região "
            "clara sobre o urso."
        ),
        "ef959725a770d810": (
            "A imagem tem vegetação densa e pouca distinção visual entre o pequeno animal "
            "e o fundo. O mapa distribui a ativação por troncos e folhas, sem uma região "
            "isolada que confirme atenção consistente ao chipmunk."
        ),
    }

    for record in errors:
        note = error_notes.get(
            record["image_id"],
            "O mapa se concentra em uma região ampla da cena; o corpo do animal "
            "e o fundo devem ser comparados visualmente.",
        )
        lines.extend([
            f"### {record['image_id']}",
            "",
            f"- Verdade: **{class_names[record['true']]}**",
            f"- Predição: **{class_names[record['pred']]}**",
            f"- Confiança: **{record['confidence']:.2%}**",
            f"- Dimensão original: {record['original_width']} x {record['original_height']} pixels",
            f"- Entrada do modelo: {record['resized_width']} x {record['resized_height']} pixels",
            f"- Fração da saliência nos 20% mais ativos: {record['top20_fraction']:.2%}",
            f"- Região dominante aproximada: {record['saliency_region']}",
            f"- Diagnóstico: {note}",
            f"- Comparação sem mapa: [imagem original](erros_sem_saliencia/{record['image_id']}_original.png)",
            "",
        ])

    lines.extend([
        "## Acertos",
        "",
        "A pergunta inversa é observar quais acertos têm saliência sobre o animal "
        "e quais parecem apoiar-se no fundo.",
        "",
    ])

    correct_notes = {
        "e0f23fd97fb21b3e": (
            "O mapa está centrado na região do gambá, sendo o caso mais claro desta "
            "galeria de acerto sobre o animal."
        ),
        "2a0873bcaf6cbfbd": (
            "A ativação aparece sobre o cervo e seus contornos superiores. O fundo também "
            "recebe alguma ativação, mas o corpo do animal é uma referência central."
        ),
        "c80dcb0a6cacc16b": (
            "A saliência é ampla sobre vegetação e solo; não é possível afirmar que o "
            "modelo usou somente o esquilo, pois o contexto florestal recebe destaque."
        ),
        "784c50e9bcd996a3": (
            "O mapa cobre o tronco, o animal e a vegetação ao redor. O tronco parece "
            "particularmente relevante, indicando possível uso conjunto de animal e fundo."
        ),
        "7798c218eda5a8b1": (
            "A ativação fica sobre o gambá-listrado, mas também se espalha pelo fundo "
            "azulado. O resultado é positivo, porém não totalmente isolado no animal."
        ),
    }

    for record in correct:
        note = correct_notes.get(
            record["image_id"],
            "A imagem acerta, mas a distribuição da saliência deve ser comparada "
            "visualmente com o animal e o contexto.",
        )
        lines.extend([
            f"### {record['image_id']}",
            "",
            f"- Classe prevista e verdadeira: **{class_names[record['pred']]}**",
            f"- Confiança: **{record['confidence']:.2%}**",
            f"- Dimensão original: {record['original_width']} x {record['original_height']} pixels",
            f"- Entrada do modelo: {record['resized_width']} x {record['resized_height']} pixels",
            f"- Fração da saliência nos 20% mais ativos: {record['top20_fraction']:.2%}",
            f"- Região dominante aproximada: {record['saliency_region']}",
            f"- Diagnóstico: {note}",
            f"- Comparação sem mapa: [imagem original](acertos_sem_saliencia/{record['image_id']}_original.png)",
            "",
        ])

    lines.extend([
        "## Síntese",
        "",
        "- Os três erros são exemplos de alta confiança, portanto não parecem falhas "
        "próximas da fronteira de decisão.",
        "- Há evidência visual de possível dependência de fundo em pelo menos dois erros "
        "de ambiente florestal.",
        "- Nos acertos, a atenção é mais claramente localizada no animal em "
        "`virginia_opossum` e `white_tailed_deer`; os outros acertos mostram mistura "
        "entre animal e contexto.",
        "- A ausência de bounding boxes impede medir quantos pixels da saliência recaíram "
        "formalmente dentro do animal. A conclusão final deve priorizar a comparação "
        "visual das versões com e sem mapa.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {args.checkpoint}. "
            "Treine a Parte 2 ou corrija o caminho informado."
        )

    labels_dir = args.data_dir / "labels"
    test_csv = labels_dir / "test_public.csv"
    classes_csv = args.data_dir / "classes.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"CSV de test_public não encontrado: {test_csv}. "
            "Extraia os dados rotulados em data/pa3_dados."
        )

    class_names, aliases = load_class_mapping(classes_csv, [test_csv])
    image_index = ImagePathIndex(args.data_dir / "images")
    _, eval_transform = make_transforms(args.image_size)
    test_dataset = TrapDataset(test_csv, image_index, aliases, "test_public", eval_transform)
    test_frame = pd.read_csv(test_csv, dtype={"image_id": str})
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if list(checkpoint.get("class_names", class_names)) != list(class_names):
        raise ValueError("A ordem das classes do checkpoint difere de classes.csv.")

    model = build_model("resnet18", num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    grad_cam = GradCAM(model, model.layer4[-1])

    records: list[dict] = []
    with torch.enable_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            for row in range(images.shape[0]):
                model.zero_grad(set_to_none=True)
                single_image = images[row:row + 1]
                logits = model(single_image)
                probabilities = torch.softmax(logits, dim=1)
                confidence, prediction = probabilities.max(dim=1)
                logits[0, prediction[0]].backward()
                cam = grad_cam(logits, prediction)[0]
                cam_image = np.asarray(
                    Image.fromarray((cam * 255).astype(np.uint8)).resize(
                        (args.image_size, args.image_size),
                        Image.Resampling.BILINEAR,
                    )
                )
                cam_image = cam_image / 255.0

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                rgb = images[row].detach().cpu().numpy().transpose(1, 2, 0)
                rgb = np.clip(rgb * std + mean, 0, 1)

                heat = plt.get_cmap("jet")(cam_image)[..., :3]
                overlay = np.clip(0.55 * rgb + 0.45 * heat, 0, 1)

                original_id = str(image_ids[row])
                original_item = next(
                    (item for item in test_dataset.items if item[0] == original_id),
                    None,
                )
                if original_item is None:
                    raise RuntimeError(
                        f"Não foi possível localizar o caminho da imagem {original_id!r}."
                    )
                original_path = original_item[1]
                with Image.open(original_path) as source:
                    original_size = source.size

                diagnostics = saliency_diagnostics(cam)
                record = {
                    "image_id": original_id,
                    "true": int(labels[row].cpu()),
                    "pred": int(prediction[0].cpu()),
                    "confidence": float(confidence[0].cpu()),
                    "is_error": bool(labels[row].cpu() != prediction[0].cpu()),
                    "cam_image": overlay,
                    "original_path": original_path,
                    "original_width": original_size[0],
                    "original_height": original_size[1],
                    "resized_width": args.image_size,
                    "resized_height": args.image_size,
                    **diagnostics,
                }
                record["bbox"] = get_bbox(test_frame, original_id)
                dx = record["saliency_center_x"] - 0.5
                dy = record["saliency_center_y"] - 0.5
                horizontal = "direita" if dx > 0.12 else "esquerda" if dx < -0.12 else "centro"
                vertical = "inferior" if dy > 0.12 else "superior" if dy < -0.12 else "meio"
                record["saliency_region"] = f"{horizontal} / {vertical}"
                records.append(record)

    errors = sorted(
        (record for record in records if record["is_error"]),
        key=lambda record: record["confidence"],
        reverse=True,
    )[: args.num_errors]
    correct = sorted(
        (record for record in records if not record["is_error"]),
        key=lambda record: record["confidence"],
        reverse=True,
    )

    # Diversifica os acertos por classe antes de selecionar os cinco melhores.
    selected_correct = []
    seen_classes = set()
    for record in correct:
        if record["pred"] not in seen_classes:
            selected_correct.append(record)
            seen_classes.add(record["pred"])
        if len(selected_correct) == args.num_correct:
            break
    if len(selected_correct) < args.num_correct:
        selected_ids = {record["image_id"] for record in selected_correct}
        for record in correct:
            if record["image_id"] not in selected_ids:
                selected_correct.append(record)
            if len(selected_correct) == args.num_correct:
                break

    save_gallery(
        "Galeria de erros — Grad-CAM",
        errors,
        args.output_dir / "galeria_erros_gradcam.png",
        class_names,
    )
    save_gallery(
        "Galeria de acertos — Grad-CAM",
        selected_correct,
        args.output_dir / "galeria_acertos_gradcam.png",
        class_names,
    )
    save_plain_gallery(
        "Galeria de erros — sem Grad-CAM",
        errors,
        args.output_dir / "galeria_erros_sem_gradcam.png",
        class_names,
    )
    save_plain_gallery(
        "Galeria de acertos — sem Grad-CAM",
        selected_correct,
        args.output_dir / "galeria_acertos_sem_gradcam.png",
        class_names,
    )
    save_individual_originals(
        errors,
        args.output_dir / "erros_sem_saliencia",
    )
    save_individual_originals(
        selected_correct,
        args.output_dir / "acertos_sem_saliencia",
    )
    write_diagnosis(
        args.output_dir / "diagnostico_completo.md",
        errors,
        selected_correct,
        class_names,
    )

    gallery_records = []
    for record in errors + selected_correct:
        bbox = record["bbox"]
        if bbox is None:
            bbox_text = "não disponível"
        else:
            x1, y1, x2, y2 = bbox
            bbox_text = f"{max(1, x2 - x1)}x{max(1, y2 - y1)}"
        dx = record["saliency_center_x"] - 0.5
        dy = record["saliency_center_y"] - 0.5
        region = ("direita" if dx > 0.12 else "esquerda" if dx < -0.12 else "centro")
        region += " / "
        region += ("inferior" if dy > 0.12 else "superior" if dy < -0.12 else "meio")
        gallery_records.append(
            {
                "image_id": record["image_id"],
                "categoria": "erro" if record["is_error"] else "acerto",
                "classe_verdadeira": class_names[record["true"]],
                "classe_prevista": class_names[record["pred"]],
                "confianca": record["confidence"],
                "largura_original": record["original_width"],
                "altura_original": record["original_height"],
                "tamanho_redimensionado": f'{record["resized_width"]}x{record["resized_height"]}',
                "bbox_animal_original": bbox_text,
                "top20_saliencia_fracao": record["top20_fraction"],
                "regiao_central_saliencia": region,
                "diagnostico": (
                    f"Grad-CAM concentra {record['top20_fraction']:.1%} da ativação "
                    f"nos 20% mais salientes, na região {region}; "
                    f"bbox do animal: {bbox_text}. "
                    "Use o overlay para verificar se a região corresponde ao animal ou ao fundo."
                ),
            }
        )

    pd.DataFrame(gallery_records).to_csv(
        args.output_dir / "galeria_detalhes.csv",
        index=False,
    )
    print(f"Erros selecionados: {len(errors)}")
    print(f"Acertos selecionados: {len(selected_correct)}")
    print(f"Galerias e diagnóstico: {args.output_dir}")


if __name__ == "__main__":
    main()
