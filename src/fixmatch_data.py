from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from src.dataset import ImagePathIndex, load_class_mapping, SquarePad

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class UnlabeledFixMatchDataset(Dataset):
    """Todas as imagens de images/train, exceto as 105 de rotulos_5.csv."""
    def __init__(self, data_dir, image_size=224):
        self.data_dir = Path(data_dir)
        labels_dir = self.data_dir / "labels"
        train_csv = labels_dir / "rotulos_5.csv"
        labeled = set(pd.read_csv(train_csv, dtype={"image_id": str})["image_id"].astype(str))
        image_root = self.data_dir / "images" / "train"
        if not image_root.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {image_root}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        self.items = [(p.stem, p) for p in paths if p.stem not in labeled]
        if not self.items:
            raise RuntimeError("Nenhuma imagem não rotulada encontrada em images/train.")
        pad = SquarePad(fill=(0, 0, 0))
        base = [pad, transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
        self.weak = transforms.Compose(base + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.strong = transforms.Compose(base + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.3, 0.08)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, path = self.items[index]
        with Image.open(path) as im:
            im = im.convert("RGB")
            return self.weak(im), self.strong(im), image_id


def load_fixmatch_datasets(data_dir, image_size=224):
    from src.dataset import TrapDataset
    data_dir = Path(data_dir)
    labels_dir = data_dir / "labels"
    class_names, aliases = load_class_mapping(
        data_dir / "classes.csv", [labels_dir / "rotulos_5.csv", labels_dir / "val.csv"]
    )
    index = ImagePathIndex(data_dir / "images")
    eval_tf = transforms.Compose([
        SquarePad(fill=(0,0,0)), transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    labeled = TrapDataset(labels_dir/"rotulos_5.csv", index, aliases, "train", eval_tf)
    val = TrapDataset(labels_dir/"val.csv", index, aliases, "val", eval_tf)
    unlabeled = UnlabeledFixMatchDataset(data_dir, image_size)
    return class_names, labeled, val, unlabeled
