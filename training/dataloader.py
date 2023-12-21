from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import (
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)


class TargetNormBenMal(Enum):
    NORMAL = 0
    BENIGN = 1
    MALIGNANT = 2

    @classmethod
    def from_birads(cls, birads: int) -> Enum:
        birads_map = {
            1: cls.NORMAL,
            2: cls.BENIGN,
            3: cls.BENIGN,
            4: cls.MALIGNANT,
            5: cls.MALIGNANT,
            6: cls.MALIGNANT,
        }

        return birads_map[birads]


class MgDataset(Dataset):
    """
    Class to ...

    Parameters:
    - ...
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        images_path_col: str,
        target_map: Callable[[Any], Enum],
        transform: Optional[A.Compose] = None,
    ):
        self.samples = samples
        self.images_path_col = images_path_col

        self.num_samples = len(samples)

        self.transform = transform

        self.target_map = target_map

    def apply_transformations(self, img: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            return torch.from_numpy(img)

        # Image must be numpy array with shape HWC
        # After it must be converted to pytorch `CHW` tensor using ToTensorV2

        # print("Image before transform:", img.shape, img.dtype)

        transformed = self.transform(image=img)

        return transformed["image"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_info = self.samples.iloc[idx]

        img = cv2.imread(
            str(sample_info[self.images_path_col]),
            cv2.IMREAD_ANYDEPTH,
        )
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        if img.shape[-1] == 1:
            img = np.stack([img[:, :, 0]] * 3, axis=-1)

        img = self.apply_transformations(img)

        # print(f"{img.shape=} {img.dtype=}")
        # print(f"{self.transform=}")

        target = self.target_map(sample_info["birads_int"]).value
        return img, target


def create_dataloader(
    samples_csv_path: Path,
    is_train: bool,
    weighted_train_sampler: bool,
    batch_size: int,
    num_workers: int = 0,
    images_path_col: str = "segmented_file_path",
) -> DataLoader:
    samples = pd.read_csv(samples_csv_path)

    if is_train:
        transforms = A.Compose(
            [
                A.Normalize(),
                A.Resize(224, 224),
                A.RandomRotate90(p=1),
                A.Flip(),
                ToTensorV2(),
            ]
        )
    else:
        transforms = A.Compose([A.Normalize(), A.Resize(224, 224), ToTensorV2()])

    dataset = MgDataset(
        samples,
        images_path_col=images_path_col,
        transform=transforms,
        target_map=TargetNormBenMal.from_birads,
    )

    if is_train:
        if weighted_train_sampler:
            print("Use weighted train sampler")
            weights, min_class_ocurrences = compute_sample_weights(
                dataset=dataset, num_classes=len(TargetNormBenMal)
            )
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(TargetNormBenMal) * min_class_ocurrences,
                replacement=True,
            )
        else:
            print("Use unweighted random train sampler")
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )


def compute_sample_weights(dataset: Dataset, num_classes: int):
    label_counts = {i: 0 for i in range(num_classes)}

    for _, label, *_ in dataset:
        label_counts[label] += 1

    print("Dataset disribution:", label_counts)
    min_class_ocurrences = min(label_counts.values())
    print("Min class ocurrences:", min_class_ocurrences)

    class_weights = {k: 1 / v for k, v in label_counts.items()}
    print("Class weights:", class_weights)

    sample_weights = []
    for _, label, *_ in dataset:
        sample_weight = class_weights[label]
        sample_weights.append(sample_weight)

    return sample_weights, min_class_ocurrences
