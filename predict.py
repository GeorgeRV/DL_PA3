from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dataset import SquarePad, load_class_mapping
from src.model import build_model, count_parameters


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PredictionDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            SquarePad(fill=(0, 0, 0)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediz classes das imagens em uma pasta usando apenas CPU."
    )
    parser.add_argument(
        "--imagens",
        required=True,
        type=Path,
        help="Pasta contendo as imagens.",
    )
    parser.add_argument(
        "--saida",
        required=True,
        type=Path,
        help="Arquivo CSV de saída.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/parte1/fixmatch_aligned/best_model.pth"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pa3_dados"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.imagens.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {args.imagens}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")

    classes_csv = args.data_dir / "classes.csv"
    if not classes_csv.exists():
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {classes_csv}")

    class_names, _ = load_class_mapping(
        classes_csv,
        [args.data_dir / "labels" / "test_public.csv"]
        if (args.data_dir / "labels" / "test_public.csv").exists()
        else [],
    )
    image_paths = sorted(
        path
        for path in args.imagens.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {args.imagens}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_classes = checkpoint.get("class_names")
    if saved_classes is not None and list(saved_classes) != list(class_names):
        raise ValueError("O checkpoint e classes.csv usam ordens de classes diferentes.")

    model = build_model("resnet18", num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device("cpu"))
    model.eval()

    parameter_count = count_parameters(model)
    dataset = PredictionDataset(image_paths, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    predictions: list[str] = []
    image_ids: list[str] = []
    with torch.inference_mode():
        for images, batch_ids in loader:
            logits = model(images)
            indices = logits.argmax(dim=1).tolist()
            predictions.extend(class_names[index] for index in indices)
            image_ids.extend(batch_ids)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "image_id": image_ids,
        "label": predictions,
    }).to_csv(args.saida, index=False)

    print(f"Número de parâmetros do modelo: {parameter_count:,}")
    print(f"Predições: {len(predictions)}")
    print(f"CSV escrito: {args.saida}")


if __name__ == "__main__":
    main()
