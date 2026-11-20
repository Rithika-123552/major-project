"""Train the hybrid ResNet50 + DeiT-Tiny road-damage classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from classification import (
    CLASS_NAMES,
    CORRECTED_MANIFEST,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_PARTITION_COUNTS,
    RoadDamageDataset,
    load_corrected_split_records,
    set_seed,
)
from hybrid_model import HybridRoadDamageModel


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, dict, torch.Tensor]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            expected.extend(labels.tolist())
    return (
        accuracy_score(expected, predicted),
        classification_report(
            expected,
            predicted,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        torch.tensor(
            confusion_matrix(
                expected, predicted, labels=list(range(len(CLASS_NAMES)))
            )
        ),
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    manifest_path = args.manifest.resolve()
    train_records, validation_records, test_records = load_corrected_split_records(
        manifest_path
    )
    if (
        len(train_records) != EXPECTED_PARTITION_COUNTS["train"]
        or len(validation_records) != EXPECTED_PARTITION_COUNTS["validation"]
        or len(test_records) != EXPECTED_PARTITION_COUNTS["test"]
    ):
        raise ValueError("The corrected manifest does not match verified partition counts.")

    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    weights = models.ResNet50_Weights.DEFAULT
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(weights.transforms().mean, weights.transforms().std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(weights.transforms().mean, weights.transforms().std),
        ]
    )
    loaders = {
        "train": DataLoader(
            RoadDamageDataset(train_records, class_to_index, train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "validation": DataLoader(
            RoadDamageDataset(
                validation_records, class_to_index, evaluation_transform
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            RoadDamageDataset(test_records, class_to_index, evaluation_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRoadDamageModel(
        pretrained=True,
        freeze_backbones=args.freeze_backbones,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        validation_accuracy, _, _ = evaluate(model, loaders["validation"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / len(loaders["train"].dataset),
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch} loss={history[-1]['train_loss']:.4f} "
            f"validation_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")
    model.load_state_dict(best_state)
    test_accuracy, report, matrix = evaluate(model, loaders["test"], device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "hybrid_road_damage.pt"
    torch.save(
        {
            "model_name": "ResNet50 + DeiT-Tiny hybrid",
            "class_names": CLASS_NAMES,
            "model_state_dict": model.state_dict(),
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "freeze_backbones": args.freeze_backbones,
            },
        },
        checkpoint_path,
    )
    experiment = {
        "model_name": "ResNet50 + DeiT-Tiny hybrid",
        "device": str(device),
        "split_manifest": str(manifest_path),
        "split_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "class_counts": EXPECTED_CLASS_COUNTS,
        "history": history,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "training_time_seconds": time.perf_counter() - started,
        "research_output_disclaimer": (
            "Predictions are research outputs and are not certified "
            "road-safety assessments."
        ),
    }
    (args.output / "hybrid_experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    print(f"Hybrid checkpoint: {checkpoint_path}")
    print(f"Test accuracy: {test_accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CORRECTED_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbones",
        action="store_true",
        help="Train only the fusion classifier while keeping pretrained features fixed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
