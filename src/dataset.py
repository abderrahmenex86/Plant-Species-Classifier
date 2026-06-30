import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


def create_transforms(is_training):
    if is_training:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness=0.3, contrast=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_data_pipelines(train_directory_path, validation_directory_path, batch_size, **kwargs):
    train_transform_pipeline = create_transforms(is_training=True)
    validation_transform_pipeline = create_transforms(is_training=False)

    train_dataset = ImageFolder(root=train_directory_path, transform=train_transform_pipeline)
    validation_dataset = ImageFolder(root=validation_directory_path, transform=validation_transform_pipeline)

    cuda_is_available = torch.cuda.is_available()

    num_workers_train = int(kwargs.get("num_workers_train", 12 if cuda_is_available else 0))
    num_workers_validation = int(kwargs.get("num_workers_validation", 4 if cuda_is_available else 0))
    pin_memory_flag = bool(kwargs.get("pin_memory_flag", True if cuda_is_available else False))
    prefetch_factor_value = (
        int(kwargs.get("prefetch_factor_value", 4 if cuda_is_available else 4)) if cuda_is_available else None
    )
    persistent_workers_flag = bool(
        kwargs.get("persistent_workers_flag", True if cuda_is_available and num_workers_train > 0 else False)
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=pin_memory_flag,
        prefetch_factor=prefetch_factor_value,
        persistent_workers=persistent_workers_flag,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_validation,
        pin_memory=pin_memory_flag,
        prefetch_factor=prefetch_factor_value,
        persistent_workers=persistent_workers_flag,
    )

    return train_dataloader, validation_dataloader, train_dataset, validation_dataset
