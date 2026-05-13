import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm.auto import tqdm


def train_epoch(model, loader, criterion, optimizer, device, acc, top5_acc, f1):

    model.train()

    acc.reset()
    top5_acc.reset()
    f1.reset()

    total_loss = 0.0
    total_samples = 0

    for inputs, outputs in tqdm(loader, desc="Training", leave=False, unit="batch"):
        inputs, outputs = inputs.to(device, non_blocking=True), outputs.to(device, non_blocking=True)

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(predictions, outputs)

        loss.backward()

        optimizer.step()

        batch_size = inputs.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        acc.update(predictions, outputs)
        top5_acc.update(predictions, outputs)
        f1.update(predictions, outputs)

    epoch_acc = acc.compute().item()
    epoch_top5_acc = top5_acc.compute().item()
    epoch_f1 = f1.compute().item()

    return {
        "loss": total_loss / total_samples,
        "acc": epoch_acc * 100,
        "top5_acc": epoch_top5_acc * 100,
        "f1": epoch_f1 * 100,
    }


def evaluate(model, loader, criterion, device, acc, top5_acc, f1):

    model.eval()

    acc.reset()
    top5_acc.reset()
    f1.reset()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, outputs in tqdm(loader, desc="Evaluating", leave=False, unit="batch"):
            inputs, outputs = inputs.to(device, non_blocking=True), outputs.to(device, non_blocking=True)

            predictions = model(inputs)

            loss = criterion(predictions, outputs)

            batch_size = inputs.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            acc.update(predictions, outputs)
            top5_acc.update(predictions, outputs)
            f1.update(predictions, outputs)

    epoch_acc = acc.compute().item()
    epoch_top5_acc = top5_acc.compute().item()
    epoch_f1 = f1.compute().item()

    return {
        "loss": total_loss / total_samples,
        "acc": epoch_acc * 100,
        "top5_acc": epoch_top5_acc * 100,
        "f1": epoch_f1 * 100,
    }


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_epochs):
    best_val_f1 = 0.0
    min_delta = 1e-4
    history = {
        "train": {"loss": [], "acc": [], "top5_acc": [], "f1": []},
        "val": {"loss": [], "acc": [], "top5_acc": [], "f1": []},
    }

    acc = MulticlassAccuracy(num_classes=1081).to(device)
    top5_acc = MulticlassAccuracy(num_classes=1081, top_k=5).to(device)
    f1 = MulticlassF1Score(num_classes=1081, average="macro").to(device)

    for epoch in tqdm(range(1, n_epochs + 1), unit="epoch", leave=True):

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, acc, top5_acc, f1)
        val_metrics = evaluate(model, val_loader, criterion, device, acc, top5_acc, f1)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(f"Current LR: {current_lr:.6f}")

        for k, v in train_metrics.items():
            history["train"][k].append(v)
        for k, v in val_metrics.items():
            history["val"][k].append(v)

        tqdm.write(
            f"Train Set | Loss -> {train_metrics['loss']:.4f} | Accuracy -> {train_metrics['acc']:.2f}% | Top 5 Accuracy -> {train_metrics['top5_acc']:.2f}% | F1 -> {train_metrics['f1']:.2f}%"
        )

        tqdm.write(
            f"Val Set   | Loss -> {val_metrics['loss']:.4f} | Accuracy -> {val_metrics['acc']:.2f}% | Top 5 Accuracy -> {val_metrics['top5_acc']:.2f}% | F1 -> {val_metrics['f1']:.2f}%"
        )

        current_val_f1 = val_metrics["f1"]

        if current_val_f1 > best_val_f1 + min_delta:

            best_val_f1 = current_val_f1

            torch.save(model.state_dict(), "plant_model.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                "checkpoint.pth",
            )
            tqdm.write("---> Best Checkpoint saved.")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            "last_checkpoint.pth",
        )
        tqdm.write("---> Last Checkpoint saved.")

    return history
