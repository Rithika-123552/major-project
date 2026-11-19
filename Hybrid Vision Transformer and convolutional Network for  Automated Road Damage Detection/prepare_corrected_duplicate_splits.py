"""Prepare a group-aware duplicate-corrected split manifest.

This script only reads the source manifest and review metadata. It writes only
to the requested corrected-artifact directory and never modifies source data or
model artifacts.
"""

from __future__ import annotations

import csv
import json
import random
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_TITLE = (
    "Hybrid Vision Transformer and convolutional Network for "
    "Automated Road Damage Detection"
)
SPLITS = ("train", "validation", "test")
VALID_DECISIONS = {"duplicate", "not_duplicate", "uncertain"}


def resolve_manual_review_path(project_root, split_dir, supplied_path, parser):
    if supplied_path is None:
        review_path = (
            split_dir
            / "near_duplicate_review"
            / "manual_review_decisions.csv"
        )
    else:
        review_path = Path(supplied_path)
        if not review_path.is_absolute():
            review_path = project_root / review_path

    review_path = review_path.resolve()
    if not review_path.is_file():
        if supplied_path is None:
            parser.error(
                f"The manual-review CSV does not exist: {review_path}"
            )
        parser.error(
            f"The supplied manual-review CSV does not exist: {review_path}"
        )
    return review_path


class DisjointSet:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def target_counts(total: int):
    train = round(total * 0.70)
    validation = round(total * 0.15)
    return {
        "train": train,
        "validation": validation,
        "test": total - train - validation,
    }


def assign_groups(groups, seed):
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]["image_paths"]), reverse=True)

    total = sum(len(group["image_paths"]) for group in groups.values())
    total_targets = target_counts(total)
    class_totals = Counter(
        label
        for group in groups.values()
        for label in group["labels"]
        for _ in range(len(group["image_paths"]))
    )
    # Labels are one-to-one per image, so replace with image-label counts.
    class_totals = Counter()
    for group in groups.values():
        if len(group["labels"]) == 1:
            class_totals[next(iter(group["labels"]))] += len(group["image_paths"])

    class_targets = {
        label: target_counts(count) for label, count in class_totals.items()
    }
    assigned_counts = Counter()
    assigned_class_counts = {split: Counter() for split in SPLITS}
    assignments = {}

    def score(split, group):
        projected_totals = assigned_counts.copy()
        projected_totals[split] += len(group["image_paths"])
        total_error = sum(
            (projected_totals[name] - total_targets[name]) ** 2
            / max(total_targets[name], 1)
            for name in SPLITS
        )

        projected_class = {
            name: assigned_class_counts[name].copy() for name in SPLITS
        }
        if len(group["labels"]) == 1:
            label = next(iter(group["labels"]))
            projected_class[split][label] += len(group["image_paths"])

        class_error = 0.0
        for label, targets in class_targets.items():
            for name in SPLITS:
                difference = projected_class[name][label] - targets[name]
                class_error += difference * difference / max(targets[name], 1)
        return total_error + class_error

    for group_id, group in items:
        chosen = min(SPLITS, key=lambda split: (score(split, group), split))
        assignments[group_id] = chosen
        assigned_counts[chosen] += len(group["image_paths"])
        if len(group["labels"]) == 1:
            label = next(iter(group["labels"]))
            assigned_class_counts[chosen][label] += len(group["image_paths"])
    return assignments


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prepare duplicate-corrected train, validation, and test assignments."
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=None,
        help="Optional corrected manual-review decisions CSV.",
    )
    arguments = parser.parse_args()

    split_dir = project_root / "artifacts" / "duplicate_aware_splits"
    output_dir = project_root / "artifacts" / "duplicate_aware_splits_corrected"

    manifest_path = split_dir / "split_manifest.csv"
    review_path = resolve_manual_review_path(
        project_root,
        split_dir,
        arguments.manual_review,
        parser,
    )
    verification_path = split_dir / "duplicate_verification.json"
    summary_path = split_dir / "duplicate_aware_summary.json"

    manifest = read_csv(manifest_path)
    review = read_csv(review_path)
    exact_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    exact_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_by_path = {}
    for row in manifest:
        path = row["image_path"].strip()
        if path in manifest_by_path:
            raise RuntimeError(f"Duplicate manifest path: {path}")
        manifest_by_path[path] = row

    dsu = DisjointSet(manifest_by_path)

    exact_ids = defaultdict(set)
    for path, row in manifest_by_path.items():
        exact_ids[row["duplicate_group_id"]].add(path)
    for paths in exact_ids.values():
        ordered = sorted(paths)
        for path in ordered[1:]:
            dsu.union(ordered[0], path)

    confirmed_edges = []
    invalid_decisions = []
    for row in review:
        decision = (row.get("manual_decision") or "").strip()
        if decision not in VALID_DECISIONS:
            invalid_decisions.append(row)
            continue
        if decision == "duplicate":
            first = row["image_path_a"].strip()
            second = row["image_path_b"].strip()
            if first not in manifest_by_path or second not in manifest_by_path:
                raise RuntimeError(
                    f"Confirmed duplicate path missing from manifest: "
                    f"{first if first not in manifest_by_path else second}"
                )
            dsu.union(first, second)
            confirmed_edges.append((first, second))

    components = defaultdict(list)
    for path in manifest_by_path:
        components[dsu.find(path)].append(path)

    component_details = []
    conflicts = []
    for index, (root, paths) in enumerate(
        sorted(components.items(), key=lambda item: min(item[1])), start=1
    ):
        ordered = sorted(paths)
        labels = sorted({
            manifest_by_path[path]["class_label"].strip() for path in ordered
        })
        exact_group_ids = sorted({
            manifest_by_path[path]["duplicate_group_id"].strip()
            for path in ordered
        })
        edge_count = sum(
            dsu.find(first) == root and dsu.find(second) == root
            for first, second in confirmed_edges
        )
        detail = {
            "duplicate_group_id": f"corrected-{index:06d}",
            "image_count": len(ordered),
            "labels": labels,
            "image_paths": ordered,
            "exact_duplicate_group_ids": exact_group_ids,
            "confirmed_duplicate_edge_count": edge_count,
            "conflicting_labels": len(labels) > 1,
            "assigned_split": "",
        }
        component_details.append(detail)
        if len(labels) > 1:
            conflicts.append(detail)

    if invalid_decisions:
        verification = {
            "status": "blocked",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_images_modified": False,
            "existing_split_manifest_modified": False,
            "model_training_performed": False,
            "blocking_issues": [
                {
                    "issue": "invalid_manual_decision",
                    "candidate_id": row.get("candidate_id", ""),
                    "value": row.get("manual_decision", ""),
                }
                for row in invalid_decisions
            ],
            "conflicting_label_groups": [],
            "confirmed_duplicate_groups_cross_partitions": None,
            "exact_duplicate_groups_cross_partitions":
                exact_verification.get("exact_duplicate_groups_cross_partitions"),
        }
        (output_dir / "corrected_split_verification.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"Blocked by {len(invalid_decisions)} invalid manual decision(s). "
            f"No corrected split manifest was generated."
        )

    if conflicts:
        with (output_dir / "duplicate_group_report.csv").open(
            "w", newline="", encoding="utf-8"
        ) as report_file:
            writer = csv.DictWriter(
                report_file,
                fieldnames=(
                    "duplicate_group_id",
                    "image_count",
                    "labels",
                    "exact_duplicate_group_ids",
                    "confirmed_duplicate_edge_count",
                    "conflicting_labels",
                    "image_paths",
                ),
            )
            writer.writeheader()
            for detail in component_details:
                writer.writerow({
                    "duplicate_group_id": detail["duplicate_group_id"],
                    "image_count": detail["image_count"],
                    "labels": "|".join(detail["labels"]),
                    "exact_duplicate_group_ids":
                        "|".join(detail["exact_duplicate_group_ids"]),
                    "confirmed_duplicate_edge_count":
                        detail["confirmed_duplicate_edge_count"],
                    "conflicting_labels": str(detail["conflicting_labels"]).lower(),
                    "image_paths": "|".join(detail["image_paths"]),
                })

        verification = {
            "status": "blocked",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_images_modified": False,
            "existing_split_manifest_modified": False,
            "model_training_performed": False,
            "blocking_issues": [
                {
                    "issue": "conflicting_class_labels_in_transitive_duplicate_group",
                    "duplicate_group_id": detail["duplicate_group_id"],
                    "labels": detail["labels"],
                    "image_paths": detail["image_paths"],
                }
                for detail in conflicts
            ],
            "conflicting_label_groups": conflicts,
            "confirmed_duplicate_groups_cross_partitions": None,
            "exact_duplicate_groups_cross_partitions":
                exact_verification.get("exact_duplicate_groups_cross_partitions"),
        }
        (output_dir / "corrected_split_verification.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"Blocked by {len(conflicts)} conflicting duplicate group(s). "
            f"No corrected split manifest was generated."
        )

    groups = {
        detail["duplicate_group_id"]: {
            "image_paths": detail["image_paths"],
            "labels": set(detail["labels"]),
        }
        for detail in component_details
    }
    assignments = assign_groups(groups, 42)

    for detail in component_details:
        detail["assigned_split"] = assignments[detail["duplicate_group_id"]]

    manifest_out = output_dir / "corrected_split_manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "image_path",
                "class_label",
                "duplicate_group_id",
                "assigned_split",
            ),
        )
        writer.writeheader()
        for path in sorted(manifest_by_path):
            row = manifest_by_path[path]
            root = dsu.find(path)
            detail = next(
                detail
                for detail in component_details
                if dsu.find(detail["image_paths"][0]) == root
            )
            writer.writerow({
                "image_path": path,
                "class_label": row["class_label"],
                "duplicate_group_id": detail["duplicate_group_id"],
                "assigned_split": detail["assigned_split"],
            })

    with (output_dir / "duplicate_group_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=(
                "duplicate_group_id",
                "image_count",
                "labels",
                "exact_duplicate_group_ids",
                "confirmed_duplicate_edge_count",
                "conflicting_labels",
                "assigned_split",
                "image_paths",
            ),
        )
        writer.writeheader()
        for detail in component_details:
            writer.writerow({
                "duplicate_group_id": detail["duplicate_group_id"],
                "image_count": detail["image_count"],
                "labels": "|".join(detail["labels"]),
                "exact_duplicate_group_ids":
                    "|".join(detail["exact_duplicate_group_ids"]),
                "confirmed_duplicate_edge_count":
                    detail["confirmed_duplicate_edge_count"],
                "conflicting_labels": "false",
                "assigned_split": detail["assigned_split"],
                "image_paths": "|".join(detail["image_paths"]),
            })

    split_counts = Counter()
    class_counts = {split: Counter() for split in SPLITS}
    for detail in component_details:
        split = detail["assigned_split"]
        split_counts[split] += detail["image_count"]
        for path in detail["image_paths"]:
            class_counts[split][manifest_by_path[path]["class_label"]] += 1

    cross_confirmed = []
    for detail in component_details:
        memberships = {
            detail["assigned_split"] for _ in detail["image_paths"]
        }
        if len(memberships) > 1:
            cross_confirmed.append(detail["duplicate_group_id"])

    violations_exact = []
    for exact_id, paths in exact_ids.items():
        memberships = {
            next(
                detail["assigned_split"]
                for detail in component_details
                if path in detail["image_paths"]
            )
            for path in paths
        }
        if len(memberships) > 1:
            violations_exact.append(exact_id)

    summary = {
        "project_title": PROJECT_TITLE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "status": "completed",
        "source_manifest": str(manifest_path),
        "source_manual_review": str(review_path),
        "original_dataset_modified": False,
        "existing_split_manifest_modified": False,
        "model_training_performed": False,
        "total_original_images": len(manifest),
        "total_manifest_images": len(component_details),
        "exact_duplicate_group_count": len(exact_ids),
        "combined_duplicate_group_count": len(component_details),
        "combined_duplicate_groups_with_multiple_images":
            sum(detail["image_count"] > 1 for detail in component_details),
        "confirmed_duplicate_decisions": len(confirmed_edges),
        "confirmed_duplicate_edges": len(confirmed_edges),
        "transitive_duplicate_group_count": len(component_details),
        "conflicting_label_group_count": 0,
        "target_split_counts": target_counts(len(manifest)),
        "actual_split_counts": dict(split_counts),
        "class_distribution_all_files":
            dict(Counter(row["class_label"] for row in manifest)),
        "class_distribution_by_split": {
            split: dict(class_counts[split]) for split in SPLITS
        },
        "split_ratio": {
            split: split_counts[split] / len(manifest) for split in SPLITS
        },
        "exact_verification_metadata":
            exact_summary.get("target_split_counts", {}),
        "limitations": [],
    }
    (output_dir / "corrected_split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    verification = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_images_modified": False,
        "existing_split_manifest_modified": False,
        "model_training_performed": False,
        "all_original_images_accounted_once":
            len(manifest) == len(set(manifest_by_path))
            and len([row["image_path"] for row in manifest]) == len(set(
                row["image_path"] for row in manifest
            )),
        "confirmed_duplicate_groups_cross_partitions": cross_confirmed,
        "exact_duplicate_groups_cross_partitions": violations_exact,
        "conflicting_label_groups": [],
        "violating_confirmed_duplicate_groups": cross_confirmed,
        "violating_exact_duplicate_groups": violations_exact,
        "unresolved_grouping_problems": [],
        "split_sizes": dict(split_counts),
        "class_distribution_by_split": {
            split: dict(class_counts[split]) for split in SPLITS
        },
        "exact_verification_source": str(verification_path),
    }
    (output_dir / "corrected_split_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    print(f"Manifest: {manifest_out}")
    print(f"Summary: {output_dir / 'corrected_split_summary.json'}")
    print(f"Group report: {output_dir / 'duplicate_group_report.csv'}")
    print(f"Verification: {output_dir / 'corrected_split_verification.json'}")


if __name__ == "__main__":
    main()
