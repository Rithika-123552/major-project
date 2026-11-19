"""Reproducible road-damage classification baseline.

This phase intentionally implements the dataset audit and the ResNet50
classification baseline before localization or explainability features.
Original images are read only; audit and experiment artifacts are written to
the configured output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_TITLE = (
    "Hybrid Vision Transformer and convolutional Network for "
    "Automated Road Damage Detection"
)
CLASS_NAMES = ("Pothole", "Asphalt", "Crack")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CORRECTED_MANIFEST = Path(
    "artifacts"
) / "duplicate_aware_splits_corrected" / "corrected_split_manifest.csv"
EXPECTED_PARTITION_COUNTS = {
    "train": 1125,
    "validation": 241,
    "test": 241,
}
EXPECTED_CLASS_COUNTS = {
    "train": {"Asphalt": 357, "Crack": 410, "Pothole": 358},
    "validation": {"Asphalt": 76, "Crack": 88, "Pothole": 77},
    "test": {"Asphalt": 77, "Crack": 88, "Pothole": 76},
}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    readable: bool
    file_size_bytes: int
    width: int | None
    height: int | None
    mode: str | None
    sha256: str | None
    error: str | None = None


def canonicalize_class_name(name: str) -> str:
    """Map harmless folder-name variations to the project's class names."""
    normalized = "".join(character for character in name.casefold() if character.isalnum())
    aliases = {
        "pothole": "Pothole",
        "potholes": "Pothole",
        "asphalt": "Asphalt",
        "crack": "Crack",
        "cracks": "Crack",
    }
    return aliases.get(normalized, name)


def set_seed(seed: int) -> None:
    """Set all supported random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _image_files(dataset_dir: Path) -> Iterable[tuple[Path, str]]:
    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            label = canonicalize_class_name(class_dir.name)
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield path, label


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_dir: Path) -> list[ImageRecord]:
    """Audit image readability and metadata without modifying source files."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    records: list[ImageRecord] = []
    for path, label in _image_files(dataset_dir):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
            records.append(
                ImageRecord(
                    path=str(path.resolve()),
                    label=label,
                    readable=True,
                    file_size_bytes=path.stat().st_size,
                    width=width,
                    height=height,
                    mode=mode,
                    sha256=_sha256(path),
                )
            )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            records.append(
                ImageRecord(
                    path=str(path.resolve()),
                    label=label,
                    readable=False,
                    file_size_bytes=path.stat().st_size,
                    width=None,
                    height=None,
                    mode=None,
                    sha256=None,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return records


def write_audit_summary(
    records: list[ImageRecord], output_dir: Path, dataset_dir: Path
) -> Path:
    """Write a machine-readable audit summary before any split or training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.sha256:
            duplicate_groups[record.sha256].append(record.path)

    readable = [record for record in records if record.readable]
    summary = {
        "project_title": PROJECT_TITLE,
        "dataset_directory": str(dataset_dir.resolve()),
        "audit_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_images_modified": False,
        "total_files": len(records),
        "readable_files": len(readable),
        "invalid_files": len(records) - len(readable),
        "class_distribution_all_files": dict(Counter(r.label for r in records)),
        "class_distribution_readable_files": dict(Counter(r.label for r in readable)),
        "dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in sorted(
                Counter((r.width, r.height) for r in readable).items()
            )
        ],
        "duplicate_groups": [
            paths for paths in duplicate_groups.values() if len(paths) > 1
        ],
        "invalid_files_detail": [
            {"path": r.path, "label": r.label, "error": r.error}
            for r in records
            if not r.readable
        ],
        "records": [asdict(record) for record in records],
    }
    summary_path = output_dir / "dataset_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


class RoadDamageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, records: list[ImageRecord], class_to_index: dict[str, int], transform
    ) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        try:
            image = Image.open(record.path).convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Could not load audited image: {record.path}") from error
        return self.transform(image), self.class_to_index[record.label]


def build_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_corrected_split_records(
    manifest_path: Path,
) -> tuple[
    list[ImageRecord],
    list[ImageRecord],
    list[ImageRecord],
]:
    project_root = Path(__file__).resolve().parent
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Corrected split manifest does not exist: {manifest_path}"
        )

    records_by_partition: dict[str, list[ImageRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen_paths: set[str] = set()
    partition_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = {
        partition: Counter() for partition in records_by_partition
    }

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_fields = {"image_path", "class_label", "assigned_split"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                "Corrected split manifest is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        for line_number, row in enumerate(reader, start=2):
            image_path = row["image_path"].strip()
            class_label = row["class_label"].strip()
            partition = row["assigned_split"].strip()

            if not image_path:
                raise ValueError(
                    f"Missing image_path in manifest row {line_number}"
                )
            if image_path in seen_paths:
                raise ValueError(
                    f"Duplicate image_path in corrected manifest: {image_path}"
                )
            seen_paths.add(image_path)

            if partition not in records_by_partition:
                raise ValueError(
                    f"Unexpected partition name '{partition}' in manifest row "
                    f"{line_number}"
                )
            if class_label not in CLASS_NAMES:
                raise ValueError(
                    f"Unexpected class label '{class_label}' in manifest row "
                    f"{line_number}"
                )

            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Manifest image does not exist: {image_path}"
                )

            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                with path.open("rb") as image_file:
                    digest = hashlib.sha256(image_file.read()).hexdigest()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise ValueError(
                    f"Manifest image cannot be opened: {image_path} "
                    f"({type(error).__name__}: {error})"
                ) from error

            record = ImageRecord(
                path=str(path.resolve()),
                label=class_label,
                readable=True,
                file_size_bytes=path.stat().st_size,
                width=width,
                height=height,
                mode=mode,
                sha256=digest,
            )
            records_by_partition[partition].append(record)
            partition_counts[partition] += 1
            class_counts[partition][class_label] += 1

    if len(seen_paths) != 1607:
        raise ValueError(
            f"Corrected manifest must contain 1607 unique images; "
            f"found {len(seen_paths)}"
        )
    if dict(partition_counts) != EXPECTED_PARTITION_COUNTS:
        raise ValueError(
            f"Partition counts do not match verified manifest: "
            f"expected {EXPECTED_PARTITION_COUNTS}, found {dict(partition_counts)}"
        )
    actual_class_counts = {
        partition: dict(class_counts[partition])
        for partition in records_by_partition
    }
    if actual_class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            f"Class counts do not match verified manifest: "
            f"expected {EXPECTED_CLASS_COUNTS}, found {actual_class_counts}"
        )

    return (
        records_by_partition["train"],
        records_by_partition["validation"],
        records_by_partition["test"],
    )


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, dict, np.ndarray]:
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
        confusion_matrix(expected, predicted, labels=list(range(len(CLASS_NAMES))),
        ),
    )


def train_baseline(
    dataset_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
) -> None:
    set_seed(seed)
    project_root = Path(__file__).resolve().parent
    resolved_manifest_path = (
        manifest_path if manifest_path.is_absolute()
        else project_root / manifest_path
    ).resolve()
    train_records, validation_records, test_records = load_corrected_split_records(
        resolved_manifest_path,
    )
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    print(f"Manifest: {resolved_manifest_path}")
    print(f"Class mapping: {class_to_index}")
    print(
        "Partition counts: "
        f"train={len(train_records)}, "
        f"validation={len(validation_records)}, "
        f"test={len(test_records)}"
    )
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
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "validation": DataLoader(
            RoadDamageDataset(validation_records, class_to_index, evaluation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            RoadDamageDataset(test_records, class_to_index, evaluation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        validation_accuracy, _, _ = _evaluate(model, loaders["validation"], device)
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": running_loss / len(loaders["train"].dataset),
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_result)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint.")
    model.load_state_dict(best_state)
    test_accuracy, report, matrix = _evaluate(model, loaders["test"], device)
    training_time_seconds = time.perf_counter() - training_started
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "resnet50_baseline.pt"
    torch.save(
        {
            "project_title": PROJECT_TITLE,
            "model_name": "ResNet50 baseline",
            "class_names": CLASS_NAMES,
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        },
        model_path,
    )
    (output_dir / "resnet50_experiment.json").write_text(
        json.dumps(
            {
                "project_title": PROJECT_TITLE,
                "model_version": "resnet50-baseline-v1",
                "device": str(device),
                "split_manifest": str(resolved_manifest_path),
                "split_sizes": {
                    "train": len(train_records),
                    "validation": len(validation_records),
                    "test": len(test_records),
                },
                "history": history,
                "test_accuracy": test_accuracy,
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "training_time_seconds": training_time_seconds,
                "research_output_disclaimer": (
                    "Predictions are research outputs and are not certified "
                    "road-safety assessments."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Model checkpoint: {model_path}")
    print(f"Test accuracy: {test_accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROJECT_TITLE)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CORRECTED_MANIFEST,
        help="Corrected duplicate-aware split manifest.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_baseline(
        dataset_dir=arguments.dataset,
        output_dir=arguments.output,
        manifest_path=arguments.manifest,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
    )