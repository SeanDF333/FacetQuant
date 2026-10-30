import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


DATASETS = {
    "vmxi": ("VMXi_Detector_Datset", "vmxi_detector"),
    "xchem": ("XChem_Detector_Dataset", "xchem_detector"),
}


def read_split(root: Path, source: str, split: str) -> list[Path]:
    folder, prefix = DATASETS[source]
    csv_path = root / folder / f"{prefix}_{split}.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        names = [row["filename"] for row in csv.DictReader(file)]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate names in {csv_path}")
    return [root / folder / "images" / name for name in names]


def split_paths(root: Path, sources: tuple[str, ...], split: str) -> list[Path]:
    paths = [path for source in sources for path in read_split(root, source, split)]
    missing = [str(path) for path in paths if not path.exists() or not path.parent.parent.joinpath("masks", path.stem + ".npz").exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images or masks; first: {missing[0]}")
    return paths


class CrystalDataset(Dataset):
    def __init__(self, image_paths: list[Path], max_size: int = 1024, augment: bool = False):
        self.image_paths = image_paths
        self.max_size = max_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = min(1.0, self.max_size / max(width, height))
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if new_size != image.size:
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        mask_path = path.parent.parent / "masks" / (path.stem + ".npz")
        with np.load(mask_path) as archive:
            labels = archive["class_labels"]
            raw_masks = archive["masks"]
            selected = raw_masks[labels == "crystal"] if raw_masks.ndim == 3 else np.empty((0, height, width), dtype=bool)
            masks = [np.asarray(Image.fromarray(mask).resize(new_size, Image.Resampling.NEAREST), dtype=bool) for mask in selected]
        if masks:
            masks = np.stack(masks)
            masks = masks[masks.any(axis=(1, 2))]
        else:
            masks = np.empty((0, new_size[1], new_size[0]), dtype=bool)

        if self.augment:
            if np.random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                masks = masks[:, :, ::-1]
            if np.random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                masks = masks[:, ::-1, :]
            if np.random.random() < 0.5:
                image = image.transpose(Image.Transpose.ROTATE_90)
                masks = np.rot90(masks, axes=(1, 2))
            image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(0.85, 1.15)))
            image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(0.85, 1.15)))

        masks = np.ascontiguousarray(masks)
        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.as_tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
            "labels": torch.ones((len(masks),), dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": torch.as_tensor(masks.sum(axis=(1, 2)).copy(), dtype=torch.float32),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        pixels = torch.from_numpy(np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))).float() / 255.0
        return pixels, target


def collate(batch):
    return tuple(zip(*batch))
