"""Train and evaluate mmWaveTryOn models with participant-held-out folds."""

import argparse
import csv
import json
import math
import os
import random
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import ACTION_IDS, PARTICIPANTS, Dataset, split
from models import build_model


MODALITIES = {
    "skeleton-cnn": "skeleton",
    "skeleton-lstm": "skeleton",
    "radar-cnn": "radar",
    "radar-lstm": "radar",
    "concat": "concat",
    "attention": "fusion",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODALITIES, required=True)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data_Processing directory containing labels.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Parent directory for a new, uniquely named run")
    parser.add_argument("--test-person", type=int, choices=PARTICIPANTS,
                        help="Run one held-out participant (default: all eight)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    radar = MODALITIES[args.model] == "radar"
    if args.batch_size is None:
        args.batch_size = 4 if radar else 64
    if args.epochs is None:
        args.epochs = 100 if radar else 150
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if radar else 3e-4
    if min(args.batch_size, args.epochs, args.patience, args.min_epochs) < 1:
        parser.error("batch size, epochs, patience, and min-epochs must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("num-workers must be non-negative and seed must be in [0, 2**32)")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def make_loader(args, people, training=False):
    dataset = Dataset(args.data_dir, people, MODALITIES[args.model])
    if not len(dataset):
        raise ValueError(f"No clips found for participants {people}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )


def forward_batch(model, batch, device):
    inputs = [batch[key].to(device, non_blocking=True)
              for key in ("images", "skeletons") if key in batch]
    return model(*inputs)


def make_optimizer(model, args):
    radar = MODALITIES[args.model] == "radar"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95 if radar else 0.999), weight_decay=0.01)
    if radar:
        # Schedule used in the image-only benchmark: warm up for up to 20 epochs.
        def multiplier(epoch):
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
            return min((epoch + 1) / 20, cosine)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    return optimizer, scheduler


def train_epoch(model, loader, device, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = forward_batch(model, batch, device)
            loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, keep_predictions=False):
    model.eval()
    correct = 0
    predictions = []
    for batch in loader:
        logits = forward_batch(model, batch, device)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        predicted = logits.argmax(dim=1).cpu()
        labels = batch["label"]
        correct += (predicted == labels).sum().item()
        if keep_predictions:
            predictions.extend(
                (clip, ACTION_IDS[actual], ACTION_IDS[guess])
                for clip, actual, guess in zip(batch["clip_id"], labels.tolist(), predicted.tolist())
            )
    return correct, predictions


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)


def train_fold(args, test_person, run_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_ids, val_ids, test_ids = split(test_person)
    train = make_loader(args, train_ids, training=True)
    validation = make_loader(args, val_ids)
    model = build_model(args.model, num_classes=len(ACTION_IDS)).to(device)
    optimizer, scheduler = make_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    fold_dir = run_dir / str(test_person)
    fold_dir.mkdir()
    best_correct = -1
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train, device, optimizer, scaler)
        scheduler.step()
        correct, _ = evaluate(model, validation, device)
        print(f"Participant {test_person}, epoch {epoch}: train loss {loss:.4f}, "
              f"validation {correct}/{len(validation.dataset)}", flush=True)
        if correct > best_correct:
            best_correct, best_epoch = correct, epoch
            # Serialize immediately: a shallow copy of state_dict shares tensors
            # with the model and would change as subsequent epochs are trained.
            checkpoint = {
                "model": args.model,
                "model_state": model.state_dict(),
                "action_ids": list(ACTION_IDS),
                "batch_size": args.batch_size,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "epoch": best_epoch,
                "validation_correct": best_correct,
                "validation_count": len(validation.dataset),
                "train_ids": list(train_ids),
                "validation_ids": list(val_ids),
                "test_ids": list(test_ids),
            }
            temporary = fold_dir / "best.tmp"
            torch.save(checkpoint, temporary)
            os.replace(temporary, fold_dir / "best.pt")
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break

    checkpoint = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = make_loader(args, test_ids)
    correct, predictions = evaluate(model, test, device, keep_predictions=True)
    write_csv(fold_dir / "predictions.csv", ("clip_id", "true_action_id", "predicted_action_id"),
              predictions)
    return {
        "test_person": test_person,
        "validation_person": val_ids[0],
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "validation_accuracy_pct": 100 * best_correct / len(validation.dataset),
        "test_correct": correct,
        "test_count": len(test.dataset),
        "test_accuracy_pct": 100 * correct / len(test.dataset),
    }


def main():
    args = parse_args()
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.model}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
    run_dir = (args.output_dir / (prefix + uuid.uuid4().hex[:12])).resolve()
    run_dir.mkdir()
    print(f"Saving this run to {run_dir}", flush=True)
    config = dict(vars(args), data_dir=str(args.data_dir.resolve()), output_dir=str(run_dir))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    people = (args.test_person,) if args.test_person is not None else PARTICIPANTS
    results = []
    for person in people:
        result = train_fold(args, person, run_dir)
        results.append(result)
        print(f"Test accuracy for {person}: {result['test_accuracy_pct']:.2f}%", flush=True)
        temporary = run_dir / "folds.tmp"
        write_csv(temporary, result.keys(), (r.values() for r in results))
        os.replace(temporary, run_dir / "folds.csv")
    accuracies = [r["test_accuracy_pct"] for r in results]
    summary = {
        "model": args.model,
        "fold_count": len(results),
        "participant_mean_accuracy_pct": statistics.mean(accuracies),
        "participant_population_sd_pct": statistics.pstdev(accuracies),
        "pooled_accuracy_pct": 100 * sum(r["test_correct"] for r in results)
                               / sum(r["test_count"] for r in results),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
