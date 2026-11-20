"""Evaluate a saved mmWaveTryOn classifier on its held-out participant."""

import argparse
import json
from pathlib import Path

import torch

from dataset import ACTION_IDS, PARTICIPANTS, split
from models import build_model
from train import MODALITIES, evaluate, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model", choices=MODALITIES,
                        help="Required for a plain state_dict")
    parser.add_argument("--test-person", type=int, choices=PARTICIPANTS,
                        help="Held-out participant, required for a plain state_dict")
    parser.add_argument("--batch-size", type=int,
                        help="Override the saved batch size (a plain state_dict defaults to 4)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint:
        parser.error("Expected a best.pt checkpoint or a plain state_dict")

    if "model_state" in checkpoint:
        model_name = checkpoint.get("model")
        if model_name not in MODALITIES or (args.model and args.model != model_name):
            parser.error("Checkpoint model is missing, unknown, or differs from --model")
        if checkpoint.get("action_ids") != list(ACTION_IDS):
            parser.error("Checkpoint action ID mapping differs from the dataset mapping")
        test_ids = checkpoint.get("test_ids")
        if not isinstance(test_ids, (tuple, list)) or len(test_ids) != 1 or test_ids[0] not in PARTICIPANTS:
            parser.error("Checkpoint must identify one valid held-out participant")
        subject = test_ids[0]
        if args.test_person is not None and args.test_person != subject:
            parser.error("--test-person differs from the checkpoint's held-out participant")
        train_ids, val_ids, _ = split(subject)
        if set(checkpoint.get("train_ids", [])) != set(train_ids) or checkpoint.get("validation_ids") != list(val_ids):
            parser.error("Checkpoint training or validation participants differ from this fold")
        args.model = model_name
        if args.batch_size is None:
            args.batch_size = checkpoint.get("batch_size")
        benchmark = checkpoint.get("cudnn_benchmark", False)
        deterministic = checkpoint.get("cudnn_deterministic", True)
        state = checkpoint["model_state"]
    else:
        if args.model is None or args.test_person is None:
            parser.error("A plain state_dict requires --model and --test-person")
        subject = args.test_person
        if args.batch_size is None:
            args.batch_size = 4
        benchmark = True
        deterministic = True
        state = checkpoint

    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        parser.error("The checkpoint's model state must be a dictionary of named tensors")
    if not isinstance(args.batch_size, int) or args.batch_size < 1 or args.num_workers < 0:
        parser.error("Supply a positive --batch-size and a non-negative --num-workers")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

    model = build_model(args.model, num_classes=len(ACTION_IDS))
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device)
    loader = make_loader(args, (subject,))
    correct, _ = evaluate(model, loader, device)
    count = len(loader.dataset)
    print(json.dumps({"model": args.model, "test_person": subject, "test_correct": correct,
                      "test_count": count, "test_accuracy_pct": 100 * correct / count}, indent=2))


if __name__ == "__main__":
    main()
