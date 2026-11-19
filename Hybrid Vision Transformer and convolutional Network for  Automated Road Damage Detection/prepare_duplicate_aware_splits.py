"""Prepare duplicate-aware train, validation, and test assignments.

This script only reads the source dataset. It never copies, edits, or deletes
images and it does not train a model. Near-duplicate matches are reported as
review candidates and are not merged into exact duplicate groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_TITLE = (
    "Hybrid Vision Transformer and convolutional Network for "
    "Automated Road Damage Detection"
)
CLASS_NAMES = ("Pothole", "Asphalt", "Crack")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "validation", "test")


@dataclass
class ImageInfo:
    path: str
    label: str
    readable: bool
    sha256: str | None
    phash: int | None
    error: str | None = None


def canonicalize_class_name(name: str) -> str:
    normalized = "".join(character for character in name.casefold() if character.isalnum())
    aliases = {
        "pothole": "Pothole",
        "potholes": "Pothole",
        "asphalt": "Asphalt",
        "crack": "Crack",
        "cracks": "Crack",
    }
    return aliases.get(normalized, name)


def image_files(dataset_dir: Path):
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = canonicalize_class_name(class_dir.name)
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    """Return a compact perceptual hash based on adjacent grayscale pixels."""
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flat:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def collect_images(dataset_dir: Path) -> list[ImageInfo]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    images: list[ImageInfo] = []
    for path, label in image_files(dataset_dir):
        resolved_path = str(path.resolve())
        try:
            with Image.open(path) as image:
                image.verify()
            images.append(
                ImageInfo(
                    path=resolved_path,
                    label=label,
                    readable=True,
                    sha256=sha256_file(path),
                    phash=difference_hash(path),
                )
            )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            images.append(
                ImageInfo(
                    path=resolved_path,
                    label=label,
                    readable=False,
                    sha256=None,
                    phash=None,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return images


def exact_groups(images: list[ImageInfo]) -> dict[str, list[ImageInfo]]:
    grouped: dict[str, list[ImageInfo]] = defaultdict(list)
    for image in images:
        if image.sha256 is not None:
            grouped[image.sha256].append(image)
    return {
        f"exact-{index:06d}": grouped[sha]
        for index, sha in enumerate(sorted(grouped), start=1)
    }


def near_duplicate_candidates(
    images: list[ImageInfo], threshold: int
) -> list[dict[str, object]]:
    readable = [image for image in images if image.readable and image.phash is not None]
    candidates: list[dict[str, object]] = []
    for index, first in enumerate(readable):
        for second in readable[index + 1 :]:
            if first.sha256 == second.sha256:
                continue
            distance = hamming_distance(first.phash, second.phash)
            if distance <= threshold:
                candidates.append(
                    {
                        "image_path_a": first.path,
                        "image_path_b": second.path,
                        "class_label_a": first.label,
                        "class_label_b": second.label,
                        "hamming_distance": distance,
                        "review_status": "candidate_requires_manual_review",
                    }
                )
    return candidates


def target_counts(total: int) -> dict[str, int]:
    train = round(total * 0.70)
    validation = round(total * 0.15)
    return {"train": train, "validation": validation, "test": total - train - validation}


def assign_groups(
    groups: dict[str, list[ImageInfo]], seed: int
) -> dict[str, str]:
    """Assign whole exact groups while minimizing size and class deviations."""
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    total = sum(len(group) for _, group in group_items)
    total_targets = target_counts(total)
    class_totals = Counter(image.label for group in groups.values() for image in group)
    class_targets = {
        label: target_counts(count) for label, count in class_totals.items()
    }
    assigned_counts = Counter()
    assigned_class_counts = {
        split: Counter() for split in SPLITS
    }
    assignments: dict[str, str] = {}

    def score(split: str, group: list[ImageInfo]) -> float:
        projected_totals = assigned_counts.copy()
        projected_totals[split] += len(group)
        total_error = sum(
            (
                projected_totals[name] - total_targets[name]
            )
            ** 2
            / max(total_targets[name], 1)
            for name in SPLITS
        )
        projected_class = {
            name: assigned_class_counts[name].copy() for name in SPLITS
        }
        projected_class[split].update(Counter(image.label for image in group))
        class_error = 0.0
        for label, targets in class_targets.items():
            for name in SPLITS:
                difference = projected_class[name][label] - targets[name]
                class_error += difference * difference / max(targets[name], 1)
        return total_error + class_error

    for group_id, group in group_items:
        chosen_split = min(SPLITS, key=lambda split: (score(split, group), split))
        assignments[group_id] = chosen_split
        assigned_counts[chosen_split] += len(group)
        assigned_class_counts[chosen_split].update(
            image.label for image in group
        )
    return assignments


def write_outputs(
    images: list[ImageInfo],
    groups: dict[str, list[ImageInfo]],
    assignments: dict[str, str],
    candidates: list[dict[str, object]],
    output_dir: Path,
    dataset_dir: Path,
    seed: int,
    near_threshold: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_ids = {
        image.path: group_id
        for group_id, group in groups.items()
        for image in group
    }
    readable_splits = {
        image.path: assignments[group_ids[image.path]]
        for image in images
        if image.readable
    }
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("image_path", "class_label", "duplicate_group_id", "assigned_split"),
        )
        writer.writeheader()
        for image in images:
            writer.writerow(
                {
                    "image_path": image.path,
                    "class_label": image.label,
                    "duplicate_group_id": group_ids.get(image.path, "unhashed"),
                    "assigned_split": readable_splits.get(image.path, "unassigned"),
                }
            )

    near_path = output_dir / "near_duplicate_candidates.csv"
    with near_path.open("w", newline="", encoding="utf-8") as near_file:
        fieldnames = (
            "image_path_a",
            "image_path_b",
            "class_label_a",
            "class_label_b",
            "hamming_distance",
            "review_status",
        )
        writer = csv.DictWriter(near_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    verification_groups = []
    violations = []
    conflicting_label_groups = []
    for group_id, group in groups.items():
        splits = sorted({readable_splits[image.path] for image in group})
        labels = sorted({image.label for image in group})
        entry = {
            "duplicate_group_id": group_id,
            "image_count": len(group),
            "labels": labels,
            "assigned_split": splits[0] if len(splits) == 1 else None,
            "assigned_splits": splits,
        }
        verification_groups.append(entry)
        if len(splits) > 1:
            violations.append(entry)
        if len(labels) > 1:
            conflicting_label_groups.append(entry)

    split_counts = Counter(readable_splits.values())
    class_split_counts = {
        split: dict(
            Counter(
                image.label
                for image in images
                if image.readable and readable_splits[image.path] == split
            )
        )
        for split in SPLITS
    }
    total_readable = sum(split_counts.values())
    summary = {
        "project_title": PROJECT_TITLE,
        "dataset_directory": str(dataset_dir.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "near_duplicate_hamming_threshold": near_threshold,
        "original_dataset_modified": False,
        "existing_model_or_artifacts_modified": False,
        "total_files": len(images),
        "readable_files": total_readable,
        "unreadable_files": len(images) - total_readable,
        "exact_group_count": len(groups),
        "exact_duplicate_group_count": sum(len(group) > 1 for group in groups.values()),
        "exact_duplicate_paths": sum(
            len(group) for group in groups.values() if len(group) > 1
        ),
        "near_duplicate_candidate_count": len(candidates),
        "conflicting_label_group_count": len(conflicting_label_groups),
        "target_split_counts": target_counts(total_readable),
        "actual_split_counts": dict(split_counts),
        "class_distribution_by_split": class_split_counts,
        "split_ratio_deviation": {
            split: split_counts[split] / max(total_readable, 1)
            for split in SPLITS
        },
        "unreadable_files": [
            {"image_path": image.path, "class_label": image.label, "error": image.error}
            for image in images
            if not image.readable
        ],
    }
    (output_dir / "duplicate_aware_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "duplicate_verification.json").write_text(
        json.dumps(
            {
                "exact_duplicate_groups_cross_partitions": bool(violations),
                "verified_without_cross_partition_leakage": not violations,
                "violating_groups": violations,
                "conflicting_label_groups": conflicting_label_groups,
                "groups": verification_groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {output_dir / 'duplicate_aware_summary.json'}")
    print(f"Verification: {output_dir / 'duplicate_verification.json'}")
    print(f"Near-duplicate candidates: {near_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROJECT_TITLE)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "duplicate_aware_splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    images = collect_images(arguments.dataset)
    groups = exact_groups(images)
    assignments = assign_groups(
        {group_id: group for group_id, group in groups.items()},
        arguments.seed,
    )
    candidates = near_duplicate_candidates(images, arguments.near_threshold)
    write_outputs(
        images,
        groups,
        assignments,
        candidates,
        arguments.output,
        arguments.dataset,
        arguments.seed,
        arguments.near_threshold,
    )


if __name__ == "__main__":
    main()
