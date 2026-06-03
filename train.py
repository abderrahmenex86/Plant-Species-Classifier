if __name__ == "__main__":
    import json
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import (
        CenterCrop,
        ColorJitter,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        Resize,
        ToTensor,
    )

    from helpers import train
    from model import PlantClassifier

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.benchmark = True

    n_epochs = 40
    batch_size = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_workers = 18
    val_workers = 12
    pin_memory = True
    prefetch_factor = 4

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    train_transforms = Compose(
        [
            Resize(256),
            CenterCrop(224),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(root="dataset/plantnet_300K/images/train", transform=train_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    val_transforms = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = ImageFolder(root="dataset/plantnet_300K/images/val", transform=val_transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor * 2,
    )

    with open("plant_class_mapping.json", "w") as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)

    n_classes = len(train_dataset.classes)
    model = PlantClassifier(num_classes=n_classes).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.model.features.parameters(), "lr": 1e-5},
            {"params": model.model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-1,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_epochs, n_classes)

    with open("training_history.json", "w") as f:
        json.dump(history, f)
