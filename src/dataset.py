from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms


_ID_COLUMNS = {"class_id", "class_idx", "class_index", "index", "idx", "id", "label_id", "target"}
_NAME_COLUMNS = {"class_name", "classname", "species", "name", "class", "label_name"}


def _normalise_token(value: object) -> str:
    """Normaliza valores de IDs/rótulos, inclusive números lidos como 3.0."""
    if pd.isna(value):
        return ""
    token = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", token):
        token = token.split(".", 1)[0]
    return token.casefold()


def _natural_sort_key(value: str):
    return (0, int(value)) if re.fullmatch(r"\d+", value) else (1, value.casefold())


def load_class_mapping(
    classes_csv: Path,
    annotation_csvs: Sequence[Path],
) -> Tuple[List[str], Dict[str, int]]:
    """Lê classes.csv e cria aliases para rótulos por nome ou índice.

    O README informa que classes.csv contém as 21 classes e seus índices, mas
    não fixa os nomes das colunas. Esta função aceita formatos comuns como
    (index, class_name), (class_id, label), uma coluna única de nomes, etc.
    """
    table = pd.read_csv(classes_csv)
    if table.empty:
        raise ValueError(f"O arquivo de classes está vazio: {classes_csv}")

    original_cols = list(table.columns)
    normalised_cols = {str(c).strip().casefold(): c for c in original_cols}
    id_col = next((normalised_cols[c] for c in _ID_COLUMNS if c in normalised_cols), None)
    name_col = next((normalised_cols[c] for c in _NAME_COLUMNS if c in normalised_cols), None)

    # Se a coluna chamada "class" for na verdade um índice numérico, prefira
    # uma coluna textual alternativa (por exemplo, "label" ou "species").
    candidates = [c for c in original_cols if c != id_col]
    non_numeric = [
        c for c in candidates
        if not pd.to_numeric(table[c], errors="coerce").notna().all()
    ]
    if name_col is not None and pd.to_numeric(table[name_col], errors="coerce").notna().all() and non_numeric:
        name_col = next((c for c in non_numeric if str(c).strip().casefold() in {"label", "class_name", "species", "name"}), non_numeric[0])

    # Se as colunas não tiverem nomes convencionais, tenta identificar a coluna
    # textual como nome e a coluna numérica como índice.
    if name_col is None:
        if len(candidates) == 1:
            name_col = candidates[0]
        elif non_numeric:
            name_col = non_numeric[0]
        elif candidates:
            name_col = candidates[-1]

    if name_col is None:
        raise ValueError(
            "Não consegui identificar a coluna com os nomes das classes em "
            f"{classes_csv}. Colunas encontradas: {original_cols}"
        )

    if id_col is not None:
        try:
            table = table.sort_values(id_col, key=lambda s: pd.to_numeric(s, errors="coerce"), kind="stable")
        except Exception:
            table = table.sort_values(id_col, kind="stable")

    names = [str(v).strip() for v in table[name_col].tolist()]
    if any(not n or n.casefold() == "nan" for n in names):
        raise ValueError(f"Há nomes de classe vazios em {classes_csv}.")
    if len(names) != len(set(n.casefold() for n in names)):
        raise ValueError(f"Há nomes de classe duplicados em {classes_csv}.")

    aliases: Dict[str, int] = {}
    for index, (_, row) in enumerate(table.iterrows()):
        # O nome é sempre um alias válido.
        aliases[_normalise_token(row[name_col])] = index
        # O índice/ID também pode ser o valor armazenado em labels/*.csv.
        if id_col is not None:
            aliases[_normalise_token(row[id_col])] = index
        # Algumas versões usam uma coluna chamada "label" como ID, enquanto
        # outras usam "label" como nome. Incluímos colunas candidatas extras.
        for col in original_cols:
            if col == name_col:
                continue
            col_name = str(col).strip().casefold()
            if col_name in _ID_COLUMNS or col_name in {"label", "class_label"}:
                aliases[_normalise_token(row[col])] = index

    # Validação adicional: os rótulos do orçamento devem ser reconhecidos.
    unknown = []
    for csv_path in annotation_csvs:
        frame = pd.read_csv(csv_path)
        if "label" not in frame.columns:
            raise ValueError(f"Esperava a coluna 'label' em {csv_path}; encontrei {list(frame.columns)}")
        for value in frame["label"].dropna().unique():
            if _normalise_token(value) not in aliases:
                unknown.append((csv_path.name, value))
    if unknown:
        preview = ", ".join(f"{filename}: {value!r}" for filename, value in unknown[:8])
        raise ValueError(
            "Alguns rótulos dos CSVs não correspondem aos nomes/índices de "
            f"classes.csv: {preview}. Confira o formato de classes.csv."
        )

    return names, aliases


class SquarePad:
    """Centraliza a imagem em um quadrado sem cortar o conteúdo."""

    def __init__(self, fill: Tuple[int, int, int] = (0, 0, 0)):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


class ImagePathIndex:
    """Índice de imagens pelo nome do arquivo, sem depender da pasta exata.

    O índice global também tolera pequenas diferenças de organização, como uma
    pasta combinada chamada test_publictrainval. Se houver IDs duplicados, a
    pasta correspondente ao split solicitado recebe prioridade.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, images_root: Path):
        if not images_root.exists():
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_root}")
        self.images_root = images_root
        self.by_stem: Dict[str, List[Path]] = {}
        for path in images_root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in self.EXTENSIONS:
                self.by_stem.setdefault(path.stem, []).append(path)

    def resolve(self, image_id: object, split: str) -> Path:
        stem = str(image_id).strip()
        if stem.lower().endswith(tuple(self.EXTENSIONS)):
            stem = Path(stem).stem
        candidates = self.by_stem.get(stem, [])
        if not candidates:
            raise FileNotFoundError(
                f"Imagem {stem!r} (split={split}) não encontrada sob {self.images_root}. "
                "Confira se o ZIP foi extraído integralmente."
            )
        if len(candidates) == 1:
            return candidates[0]

        split_key = split.casefold()
        preferred = [p for p in candidates if split_key in str(p.parent).casefold()]
        if not preferred and split == "test_public":
            preferred = [p for p in candidates if "test" in str(p.parent).casefold()]
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(
            f"O image_id {stem!r} aparece em mais de um caminho e não foi possível "
            f"desambiguar para o split {split!r}: {candidates}"
        )


class TrapDataset(Dataset):
    def __init__(
        self,
        annotations_csv: Path,
        image_index: ImagePathIndex,
        label_to_index: Dict[str, int],
        split: str,
        transform=None,
    ):
        self.frame = pd.read_csv(annotations_csv, dtype={"image_id": str})
        required = {"image_id", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes em {annotations_csv}: {sorted(missing)}")
        self.image_index = image_index
        self.split = split
        self.transform = transform
        self.items = []
        for row in self.frame.itertuples(index=False):
            image_id = getattr(row, "image_id")
            raw_label = getattr(row, "label")
            key = _normalise_token(raw_label)
            if key not in label_to_index:
                raise ValueError(f"Rótulo desconhecido {raw_label!r} em {annotations_csv}")
            path = image_index.resolve(image_id, split)
            self.items.append((str(image_id), path, int(label_to_index[key])))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_id, path, label = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label, image_id


def make_transforms(image_size: int = 224):
    """Transformações conservadoras para preservar animais pequenos.

    Não usa RandomResizedCrop: um crop agressivo pode remover o animal quando
    ele ocupa uma fração muito pequena do quadro.
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        SquarePad(fill=(0, 0, 0)),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    eval_transform = transforms.Compose([
        SquarePad(fill=(0, 0, 0)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_transform, eval_transform


def load_datasets(data_dir: Path, image_size: int = 224, include_test: bool = True):
    data_dir = Path(data_dir)
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"
    required_paths = [
        labels_dir / "rotulos_5.csv",
        labels_dir / "val.csv",
        data_dir / "classes.csv",
        images_dir,
    ]
    if include_test:
        required_paths.append(labels_dir / "test_public.csv")
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Arquivos/pastas obrigatórios não encontrados:\n- " + "\n- ".join(missing))

    annotation_paths = [labels_dir / "rotulos_5.csv", labels_dir / "val.csv"]
    if include_test:
        annotation_paths.append(labels_dir / "test_public.csv")
    class_names, label_to_index = load_class_mapping(data_dir / "classes.csv", annotation_paths)
    if len(class_names) != 21:
        raise ValueError(f"Esperadas 21 classes, mas classes.csv contém {len(class_names)}.")

    image_index = ImagePathIndex(images_dir)
    train_aug, eval_transform = make_transforms(image_size)

    train_dataset = TrapDataset(
        labels_dir / "rotulos_5.csv", image_index, label_to_index, "train", train_aug
    )
    train_eval_dataset = TrapDataset(
        labels_dir / "rotulos_5.csv", image_index, label_to_index, "train", eval_transform
    )
    val_dataset = TrapDataset(
        labels_dir / "val.csv", image_index, label_to_index, "val", eval_transform
    )
    test_dataset = None
    if include_test:
        test_dataset = TrapDataset(
            labels_dir / "test_public.csv", image_index, label_to_index, "test_public", eval_transform
        )

    if len(train_dataset) != 105:
        raise ValueError(
            f"rotulos_5.csv deveria ter 105 imagens (5 por classe), mas tem {len(train_dataset)}."
        )
    return class_names, train_dataset, train_eval_dataset, val_dataset, test_dataset
