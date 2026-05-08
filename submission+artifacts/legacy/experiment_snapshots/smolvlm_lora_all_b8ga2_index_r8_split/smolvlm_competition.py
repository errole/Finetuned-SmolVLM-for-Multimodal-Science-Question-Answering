from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoProcessor, Trainer, TrainingArguments

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # transformers v5 also exposes image-text-to-text auto classes
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_TEXT_SCOPES = ("text_model", "language_model")
METADATA_FIELDS = ("task", "grade", "subject", "topic", "category", "skill")
QA_SCHEMA_FIELDS = (
    "id",
    "image_path",
    "question",
    "choices",
    "num_choices",
    "answer",
    "hint",
    "lecture",
    "solution",
) + METADATA_FIELDS
STRICT_SPLIT_GROUP_KEYS = ("image_hash", "question", "question_choices")
STRICT_SPLIT_STRATIFY_KEYS = (
    "num_choices_answer",
    "num_choices",
    "answer",
    "subject",
    "topic",
    "category",
    "skill",
)
COMPETITION_CONFIG_FILE = "competition_config.json"
CHECKPOINT_CONFIG_FIELDS = (
    "image_longest_edge",
    "do_image_splitting",
    "max_length",
    "question_chars",
    "hint_chars",
    "lecture_chars",
    "include_metadata",
    "answer_format",
    "score_normalization",
)
CHECKPOINT_CONFIG_DEFAULTS = {
    "image_longest_edge": 512,
    "do_image_splitting": False,
    "max_length": 2048,
    "question_chars": 2000,
    "hint_chars": 2500,
    "lecture_chars": 2500,
    "include_metadata": False,
    "answer_format": "index",
    "score_normalization": "auto",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\r\n", "\n").strip()


def trim_text(text: str, char_limit: int) -> str:
    text = clean_text(text)
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    trimmed = text[:char_limit].rstrip()
    last_space = trimmed.rfind(" ")
    if last_space > char_limit * 0.8:
        trimmed = trimmed[:last_space].rstrip()
    return trimmed


def parse_path_list(value: Optional[str]) -> List[Path]:
    if value is None:
        return []
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_choices(value: Any) -> List[str]:
    if isinstance(value, list):
        return [clean_text(choice) for choice in value]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, list):
        raise ValueError(f"Could not parse choices JSON: {text[:120]}")
    return [clean_text(choice) for choice in parsed]


def format_answer_target(item: Dict[str, Any], answer: int, answer_format: str) -> str:
    answer_index = int(answer)
    choice_text = clean_text(item["choices"][answer_index])
    if answer_format == "index":
        return str(answer_index)
    if answer_format == "choice_text":
        return choice_text
    if answer_format == "index_choice":
        return f"{answer_index}. {choice_text}"
    raise ValueError(f"Unsupported answer_format: {answer_format}")


def resolve_score_normalization(answer_format: str, score_normalization: str) -> str:
    if score_normalization != "auto":
        return score_normalization
    return "sum" if answer_format == "index" else "mean"


def resolve_image_path(data_dir: Path, image_path: Any) -> Path:
    raw = Path(clean_text(image_path))
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            data_dir / raw,
            data_dir / "images" / raw,
        ]
    )
    if raw.parts and raw.parts[0].lower() == "images":
        tail = Path(*raw.parts[1:])
        candidates.extend(
            [
                data_dir / tail,
                data_dir / "images" / tail,
            ]
        )
    candidates.append(data_dir / "images" / raw.name)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(f"Image not found for {image_path!r}. Searched:\n{searched}")


def load_rgb_image(path: Path, image_bytes: Optional[bytes] = None) -> Image.Image:
    source = BytesIO(image_bytes) if image_bytes is not None else path
    with Image.open(source) as image:
        return image.convert("RGB")


def compute_file_sha256(path: Path, cache: Optional[Dict[Path, str]] = None) -> str:
    cached = cache.get(path) if cache is not None else None
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache is not None:
        cache[path] = value
    return value


def resolve_eval_csv_path(args: argparse.Namespace, split: Optional[str] = None) -> Tuple[Path, str]:
    eval_csv = getattr(args, "eval_csv", None)
    if eval_csv:
        path = Path(eval_csv)
        return path, path.stem

    split_name = split or getattr(args, "split", "val")
    return Path(args.data_dir) / f"{split_name}.csv", split_name


def parse_group_key_list(value: str) -> List[str]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    if not keys:
        raise ValueError("Expected at least one --group-by key.")
    invalid = [key for key in keys if key not in STRICT_SPLIT_GROUP_KEYS]
    if invalid:
        raise ValueError(
            f"Unsupported group keys {invalid}. Expected values from {', '.join(STRICT_SPLIT_GROUP_KEYS)}."
        )
    return keys


def parse_stratify_key_list(value: str) -> List[str]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    if not keys:
        raise ValueError("Expected at least one --stratify-by key.")
    invalid = [key for key in keys if key not in STRICT_SPLIT_STRATIFY_KEYS]
    if invalid:
        raise ValueError(
            f"Unsupported stratify keys {invalid}. Expected values from {', '.join(STRICT_SPLIT_STRATIFY_KEYS)}."
        )
    return keys


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def load_records_for_analysis(
    csv_path: Path,
    data_dir: Path,
    split_name: str,
    require_answer: bool = True,
    image_hash_cache: Optional[Dict[Path, str]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    fieldnames, raw_rows = read_csv_rows(csv_path)
    records: List[Dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows, start=2):
        choices = parse_choices(row.get("choices", "[]"))
        num_choices = int(clean_text(row.get("num_choices", len(choices))))
        if num_choices != len(choices):
            raise ValueError(
                f"{csv_path}:{row_index} has num_choices={num_choices}, but parsed {len(choices)} choices."
            )

        record: Dict[str, Any] = {
            "row": {field: row.get(field, "") for field in QA_SCHEMA_FIELDS},
            "source_csv": str(csv_path),
            "split_name": split_name,
            "row_index": row_index,
            "id": clean_text(row.get("id")),
            "image_path": clean_text(row.get("image_path")),
            "image_file": resolve_image_path(data_dir, row.get("image_path")),
            "question": clean_text(row.get("question")),
            "choices": tuple(choices),
            "num_choices": num_choices,
            "hint": clean_text(row.get("hint")),
            "lecture": clean_text(row.get("lecture")),
            "solution": clean_text(row.get("solution")),
        }
        if require_answer:
            answer_text = clean_text(row.get("answer"))
            if answer_text == "":
                raise ValueError(f"{csv_path}:{row_index} is missing an answer.")
            record["answer"] = int(answer_text)
        else:
            record["answer"] = None

        for field in METADATA_FIELDS:
            record[field] = clean_text(row.get(field))

        record["image_hash"] = compute_file_sha256(record["image_file"], cache=image_hash_cache)
        records.append(record)

    return fieldnames, records


def summarize_records(fieldnames: List[str], records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    id_counts = Counter(record["id"] for record in records)
    image_counts = Counter(record["image_path"] for record in records)
    question_choice_counts = Counter((record["question"], record["choices"]) for record in records)
    num_choice_dist = Counter(int(record["num_choices"]) for record in records)
    answer_dist = Counter(int(record["answer"]) for record in records if record.get("answer") is not None)
    joint_dist = Counter(
        (int(record["num_choices"]), int(record["answer"]))
        for record in records
        if record.get("answer") is not None
    )
    metadata_missing = Counter()
    hint_missing = 0
    lecture_missing = 0
    solution_missing = 0
    question_lengths = []
    hint_lengths = []
    lecture_lengths = []

    for record in records:
        question_lengths.append(len(record["question"]))
        hint_lengths.append(len(record["hint"]))
        lecture_lengths.append(len(record["lecture"]))
        if not record["hint"]:
            hint_missing += 1
        if not record["lecture"]:
            lecture_missing += 1
        if not record["solution"]:
            solution_missing += 1
        for field in METADATA_FIELDS:
            if not record[field]:
                metadata_missing[field] += 1

    return {
        "rows": len(records),
        "schema_ok": fieldnames == list(QA_SCHEMA_FIELDS),
        "fieldnames": fieldnames,
        "duplicate_ids": sum(1 for count in id_counts.values() if count > 1),
        "duplicate_image_paths": sum(1 for count in image_counts.values() if count > 1),
        "duplicate_question_choice_pairs": sum(1 for count in question_choice_counts.values() if count > 1),
        "hint_missing": hint_missing,
        "lecture_missing": lecture_missing,
        "solution_missing": solution_missing,
        "metadata_missing": dict(metadata_missing),
        "num_choice_dist": dict(sorted(num_choice_dist.items())),
        "answer_dist": dict(sorted(answer_dist.items())),
        "joint_answer_dist": {f"{num_choices}:{answer}": count for (num_choices, answer), count in sorted(joint_dist.items())},
        "question_length": {
            "min": min(question_lengths) if question_lengths else 0,
            "max": max(question_lengths) if question_lengths else 0,
        },
        "hint_length": {
            "min": min(hint_lengths) if hint_lengths else 0,
            "max": max(hint_lengths) if hint_lengths else 0,
        },
        "lecture_length": {
            "min": min(lecture_lengths) if lecture_lengths else 0,
            "max": max(lecture_lengths) if lecture_lengths else 0,
        },
    }


def summarize_image_integrity(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dimensions = Counter()
    widths: List[int] = []
    heights: List[int] = []
    corrupt: List[Dict[str, Any]] = []

    for record in records:
        try:
            with Image.open(record["image_file"]) as image:
                image.verify()
            with Image.open(record["image_file"]) as image:
                width, height = image.size
        except Exception as exc:
            corrupt.append({"id": record["id"], "image": str(record["image_file"]), "error": repr(exc)})
            continue

        dimensions[(width, height)] += 1
        widths.append(width)
        heights.append(height)

    top_dimensions = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in dimensions.most_common(10)
    ]
    return {
        "corrupt_images": len(corrupt),
        "sample_corrupt_images": corrupt[:10],
        "unique_dimensions": len(dimensions),
        "top_dimensions": top_dimensions,
        "min_width": min(widths) if widths else 0,
        "max_width": max(widths) if widths else 0,
        "min_height": min(heights) if heights else 0,
        "max_height": max(heights) if heights else 0,
    }


def summarize_cross_split_overlap(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    train_questions = {record["question"] for record in train_records}
    train_question_choices = {(record["question"], record["choices"]) for record in train_records}
    train_image_hashes = {record["image_hash"] for record in train_records}
    train_image_question = {(record["image_hash"], record["question"]) for record in train_records}
    train_image_question_choices = {
        (record["image_hash"], record["question"], record["choices"]) for record in train_records
    }
    train_exact = {
        (record["image_hash"], record["question"], record["choices"], int(record["answer"]))
        for record in train_records
        if record.get("answer") is not None
    }

    val_overlap_counts = Counter()
    for record in val_records:
        if record["question"] in train_questions:
            val_overlap_counts["same_question"] += 1
        if (record["question"], record["choices"]) in train_question_choices:
            val_overlap_counts["same_question_choices"] += 1
        if record["image_hash"] in train_image_hashes:
            val_overlap_counts["same_image_hash"] += 1
        if (record["image_hash"], record["question"]) in train_image_question:
            val_overlap_counts["same_image_and_question"] += 1
        if (record["image_hash"], record["question"], record["choices"]) in train_image_question_choices:
            val_overlap_counts["same_image_question_choices"] += 1
        if record.get("answer") is not None and (
            record["image_hash"],
            record["question"],
            record["choices"],
            int(record["answer"]),
        ) in train_exact:
            val_overlap_counts["exact_duplicates"] += 1

    shared_questions = train_questions & {record["question"] for record in val_records}
    shared_image_hashes = train_image_hashes & {record["image_hash"] for record in val_records}
    return {
        "shared_unique_questions": len(shared_questions),
        "shared_unique_image_hashes": len(shared_image_hashes),
        "val_row_overlap_counts": dict(val_overlap_counts),
        "val_rows": len(val_records),
        "sample_shared_questions": sorted(shared_questions)[:10],
    }


def build_dataqa_report(
    data_dir: Path,
    train_csv: Path,
    val_csv: Path,
) -> Dict[str, Any]:
    image_hash_cache: Dict[Path, str] = {}
    train_fieldnames, train_records = load_records_for_analysis(
        train_csv,
        data_dir,
        split_name="train",
        image_hash_cache=image_hash_cache,
    )
    val_fieldnames, val_records = load_records_for_analysis(
        val_csv,
        data_dir,
        split_name="val",
        image_hash_cache=image_hash_cache,
    )

    conflicting_templates: List[Dict[str, Any]] = []
    template_groups: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for record in [*train_records, *val_records]:
        template_groups[(record["question"], record["choices"])].append(record)

    for (question, choices), records in template_groups.items():
        answers = sorted({int(record["answer"]) for record in records if record.get("answer") is not None})
        if len(records) > 1 and len(answers) > 1:
            conflicting_templates.append(
                {
                    "question": question[:160],
                    "num_choices": len(choices),
                    "answers": answers,
                    "example_ids": [record["id"] for record in records[:10]],
                    "splits": sorted({record["split_name"] for record in records}),
                }
            )

    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "split_reports": {
            "train": {
                **summarize_records(train_fieldnames, train_records),
                "image_integrity": summarize_image_integrity(train_records),
            },
            "val": {
                **summarize_records(val_fieldnames, val_records),
                "image_integrity": summarize_image_integrity(val_records),
            },
        },
        "cross_split": {
            **summarize_cross_split_overlap(train_records, val_records),
            "conflicting_question_choice_answers": {
                "count": len(conflicting_templates),
                "samples": conflicting_templates[:20],
            },
        },
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def group_records_for_strict_split(records: Sequence[Dict[str, Any]], group_keys: Sequence[str]) -> List[List[int]]:
    union_find = UnionFind(len(records))
    seen: Dict[Tuple[str, Any], int] = {}
    for index, record in enumerate(records):
        key_values: Dict[str, Any] = {
            "image_hash": record["image_hash"],
            "question": record["question"],
            "question_choices": (record["question"], record["choices"]),
        }
        for group_key in group_keys:
            value = key_values[group_key]
            seen_key = (group_key, value)
            if seen_key in seen:
                union_find.union(index, seen[seen_key])
            else:
                seen[seen_key] = index

    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union_find.find(index)].append(index)
    return list(groups.values())


def split_stratify_value(record: Dict[str, Any], stratify_key: str) -> Any:
    if stratify_key == "num_choices_answer":
        return (int(record["num_choices"]), int(record["answer"]))
    if stratify_key == "num_choices":
        return int(record["num_choices"])
    if stratify_key == "answer":
        return int(record["answer"])
    if stratify_key in METADATA_FIELDS:
        return clean_text(record.get(stratify_key))
    raise ValueError(f"Unsupported stratify key: {stratify_key}")


def build_group_split(
    records: Sequence[Dict[str, Any]],
    grouped_indices: Sequence[Sequence[int]],
    stratify_keys: Sequence[str],
    val_ratio: float,
    seed: int,
    search_trials: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    target_size = max(1, round(len(records) * val_ratio))
    target_counts_by_key = {
        key: Counter(split_stratify_value(record, key) for record in records) for key in stratify_keys
    }
    target_float_by_key = {
        key: {value: count * val_ratio for value, count in counts.items()}
        for key, counts in target_counts_by_key.items()
    }

    groups = []
    for group_id, indices in enumerate(grouped_indices):
        group_records = [records[index] for index in indices]
        stratify_counts = {
            key: Counter(split_stratify_value(record, key) for record in group_records) for key in stratify_keys
        }
        groups.append(
            {
                "group_id": group_id,
                "indices": list(indices),
                "records": group_records,
                "size": len(indices),
                "stratify_counts": stratify_counts,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(groups)

    def objective(candidate_size: int, candidate_counts_by_key: Dict[str, Counter]) -> float:
        size_term = abs(candidate_size - target_size) / max(target_size, 1)
        stratify_term = 0.0
        for stratify_key in stratify_keys:
            value_targets = target_float_by_key[stratify_key]
            value_term = 0.0
            for value, target in value_targets.items():
                value_term += abs(candidate_counts_by_key[stratify_key].get(value, 0) - target) / max(target, 1.0)
            stratify_term += value_term / max(len(value_targets), 1)
        return size_term + stratify_term / max(len(stratify_keys), 1)

    def summarize_candidate(candidate_val_groups: Sequence[Dict[str, Any]]) -> Tuple[int, Dict[str, Counter]]:
        candidate_size = sum(group["size"] for group in candidate_val_groups)
        candidate_counts_by_key: Dict[str, Counter] = {key: Counter() for key in stratify_keys}
        for group in candidate_val_groups:
            for key in stratify_keys:
                candidate_counts_by_key[key].update(group["stratify_counts"][key])
        return candidate_size, candidate_counts_by_key

    def random_candidate() -> List[Dict[str, Any]]:
        ordered_groups = groups[:]
        rng.shuffle(ordered_groups)
        candidate_groups: List[Dict[str, Any]] = []
        candidate_size = 0
        for group in ordered_groups:
            if candidate_size >= target_size:
                break
            candidate_groups.append(group)
            candidate_size += group["size"]
        return candidate_groups

    def greedy_candidate() -> List[Dict[str, Any]]:
        remaining_groups = groups[:]
        candidate_groups: List[Dict[str, Any]] = []
        candidate_size = 0
        candidate_counts_by_key: Dict[str, Counter] = {key: Counter() for key in stratify_keys}
        while remaining_groups and candidate_size < target_size:
            best_index = 0
            best_score: Optional[float] = None
            for index, group in enumerate(remaining_groups):
                next_counts_by_key = {
                    key: candidate_counts_by_key[key] + group["stratify_counts"][key] for key in stratify_keys
                }
                next_size = candidate_size + group["size"]
                score = objective(next_size, next_counts_by_key)
                if best_score is None or score < best_score:
                    best_index = index
                    best_score = score
            chosen_group = remaining_groups.pop(best_index)
            candidate_groups.append(chosen_group)
            candidate_size += chosen_group["size"]
            for key in stratify_keys:
                candidate_counts_by_key[key].update(chosen_group["stratify_counts"][key])
        return candidate_groups

    best_val_groups = greedy_candidate()
    best_val_size, best_val_counts_by_key = summarize_candidate(best_val_groups)
    best_score = objective(best_val_size, best_val_counts_by_key)
    trials = max(search_trials, 0)
    for _ in range(trials):
        candidate_val_groups = random_candidate()
        candidate_size, candidate_counts_by_key = summarize_candidate(candidate_val_groups)
        score = objective(candidate_size, candidate_counts_by_key)
        if score < best_score:
            best_score = score
            best_val_groups = candidate_val_groups

    val_group_ids = {group["group_id"] for group in best_val_groups}
    val_groups = [group for group in groups if group["group_id"] in val_group_ids]
    train_groups = [group for group in groups if group["group_id"] not in val_group_ids]
    train_records = [record for group in train_groups for record in group["records"]]
    val_records = [record for group in val_groups for record in group["records"]]
    train_records.sort(key=lambda record: record["id"])
    val_records.sort(key=lambda record: record["id"])

    summary = {
        "group_count": len(groups),
        "largest_group": max((group["size"] for group in groups), default=0),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "target_val_rows": target_size,
        "val_ratio": val_ratio,
        "stratify_by": list(stratify_keys),
        "search_trials": trials,
        "split_score": best_score,
    }
    return train_records, val_records, summary


def write_records_to_csv(csv_path: Path, records: Sequence[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QA_SCHEMA_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow({field: record["row"].get(field, "") for field in QA_SCHEMA_FIELDS})


@dataclass
class PromptBuilder:
    question_chars: int = 2000
    hint_chars: int = 2500
    lecture_chars: int = 2500
    include_metadata: bool = False
    answer_format: str = "index"

    def answer_instruction(self, num_choices: int) -> str:
        if self.answer_format == "index":
            return f"Answer with one integer from 0 to {num_choices - 1}."
        if self.answer_format == "choice_text":
            return "Answer with only the exact choice text."
        if self.answer_format == "index_choice":
            return "Answer with the full choice in the format '<index>. <choice text>'."
        raise ValueError(f"Unsupported answer_format: {self.answer_format}")

    def build_user_text(self, item: Dict[str, Any]) -> str:
        choices = item["choices"]
        num_choices = int(item["num_choices"])
        choice_lines = [f"{idx}. {choice}" for idx, choice in enumerate(choices)]

        answer_line = self.answer_instruction(num_choices)
        if self.answer_format == "index":
            response_rule = "Use only the provided image and text. Return only the 0-indexed answer integer."
        elif self.answer_format == "choice_text":
            response_rule = "Use only the provided image and text. Return only the exact choice text."
        else:
            response_rule = "Use only the provided image and text. Return only the matching '<index>. <choice text>' line."

        lines = [
            "You are solving a science multiple-choice question.",
            response_rule,
            "",
            "Question:",
            trim_text(item.get("question", ""), self.question_chars),
            "",
            "Choices:",
            *choice_lines,
        ]

        hint = trim_text(item.get("hint", ""), self.hint_chars)
        if hint:
            lines.extend(["", "Hint or passage:", hint])

        lecture = trim_text(item.get("lecture", ""), self.lecture_chars)
        if lecture:
            lines.extend(["", "Lesson context:", lecture])

        if self.include_metadata:
            metadata = [
                f"{field}: {clean_text(item.get(field, ''))}"
                for field in METADATA_FIELDS
                if clean_text(item.get(field, ""))
            ]
            if metadata:
                lines.extend(["", "Metadata:", *metadata])

        lines.extend(["", answer_line])
        return "\n".join(lines)


def build_messages(user_text: str, answer: Optional[str] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer)}],
            }
        )
    return messages


def apply_chat_template(processor: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
        )


def call_processor(
    processor: Any,
    texts: Sequence[str],
    images: Sequence[Image.Image],
    max_length: int,
    processor_kwargs: Dict[str, Any],
) -> Any:
    kwargs: Dict[str, Any] = {
        "text": list(texts),
        "images": list(images),
        "return_tensors": "pt",
        "padding": True,
    }
    if max_length > 0:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length

    try:
        return processor(**kwargs, **processor_kwargs)
    except TypeError as exc:
        message = str(exc).lower()
        if processor_kwargs and ("unexpected keyword" in message or "got an unexpected" in message):
            return processor(**kwargs)
        raise


class ScienceQADataset(Dataset):
    def __init__(
        self,
        csv_path: Path | Sequence[Path],
        data_dir: Path,
        prompt_builder: PromptBuilder,
        require_answer: bool,
        cache_images: str = "none",
    ) -> None:
        self.csv_paths = [Path(part) for part in csv_path] if isinstance(csv_path, (list, tuple)) else [Path(csv_path)]
        self.data_dir = data_dir
        self.prompt_builder = prompt_builder
        self.cache_images = cache_images
        frames = [pd.read_csv(path) for path in self.csv_paths]
        frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        self.items = [self._row_to_item(row, require_answer=require_answer) for _, row in frame.iterrows()]
        if self.cache_images == "bytes":
            cached_mb = sum(len(item["image_bytes"]) for item in self.items) / 1024**2
            dataset_label = "+".join(path.stem for path in self.csv_paths)
            print(f"Cached {len(self.items)} {dataset_label} images in RAM as compressed bytes ({cached_mb:.1f} MB).")

    def _row_to_item(self, row: pd.Series, require_answer: bool) -> Dict[str, Any]:
        choices = parse_choices(row.get("choices", "[]"))
        num_choices = int(row.get("num_choices", len(choices)))
        if num_choices != len(choices):
            raise ValueError(f"{row.get('id')} has num_choices={num_choices}, but {len(choices)} choices were parsed")

        item: Dict[str, Any] = {
            "id": clean_text(row.get("id")),
            "image_path": resolve_image_path(self.data_dir, row.get("image_path")),
            "question": clean_text(row.get("question")),
            "choices": choices,
            "num_choices": num_choices,
            "hint": clean_text(row.get("hint")),
            "lecture": clean_text(row.get("lecture")),
        }
        if self.cache_images == "bytes":
            item["image_bytes"] = item["image_path"].read_bytes()
        for field in METADATA_FIELDS:
            item[field] = clean_text(row.get(field))

        if "answer" in row and clean_text(row.get("answer")) != "":
            item["answer"] = int(row.get("answer"))
        elif require_answer:
            raise ValueError(f"{row.get('id')} is missing an answer")

        return item

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


@dataclass
class VlmDataCollator:
    processor: Any
    prompt_builder: PromptBuilder
    max_length: int
    processor_kwargs: Dict[str, Any]
    answer_format: str

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [load_rgb_image(feature["image_path"], feature.get("image_bytes")) for feature in features]
        prompt_texts: List[str] = []
        full_texts: List[str] = []

        for feature in features:
            user_text = self.prompt_builder.build_user_text(feature)
            prompt_texts.append(
                apply_chat_template(self.processor, build_messages(user_text), add_generation_prompt=True)
            )
            full_texts.append(
                apply_chat_template(
                    self.processor,
                    build_messages(
                        user_text,
                        answer=format_answer_target(feature, int(feature["answer"]), self.answer_format),
                    ),
                    add_generation_prompt=False,
                )
            )

        full_inputs = call_processor(
            self.processor,
            full_texts,
            images,
            max_length=self.max_length,
            processor_kwargs=self.processor_kwargs,
        )
        prompt_inputs = call_processor(
            self.processor,
            prompt_texts,
            images,
            max_length=self.max_length,
            processor_kwargs=self.processor_kwargs,
        )

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100

        active_labels = labels.ne(-100).sum(dim=1)
        if int(active_labels.min().item()) == 0:
            raise ValueError(
                "At least one example has no answer tokens after truncation. "
                "Increase --max-length or lower the prompt character limits."
            )

        full_inputs["labels"] = labels
        return dict(full_inputs)


def select_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32" or not torch.cuda.is_available():
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def require_cuda(args: argparse.Namespace, command_name: str) -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
        print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
        return

    if getattr(args, "allow_cpu", False):
        print(
            "WARNING: CUDA is not available. Running on CPU because --allow-cpu was set. "
            "This will be very slow for SmolVLM."
        )
        return

    raise RuntimeError(
        f"CUDA is not available for `{command_name}`.\n"
        f"Python executable: {sys.executable}\n"
        f"PyTorch version: {torch.__version__}\n"
        f"PyTorch CUDA runtime: {torch.version.cuda}\n\n"
        "You are probably running the system Python instead of the project venv. "
        "Stop this run and use one of these commands from the project directory:\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python smolvlm_competition.py train --gradient-checkpointing --batch-size 2 --grad-accum-steps 4\n\n"
        "Or bypass activation explicitly:\n"
        "  .\\.venv\\Scripts\\python.exe smolvlm_competition.py train --gradient-checkpointing --batch-size 2 --grad-accum-steps 4"
    )


def configure_cuda_performance(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = True
    if args.tf32:
        torch.set_float32_matmul_precision("high")


def load_processor(args: argparse.Namespace, source: Optional[Path] = None) -> Any:
    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SmolVLM image processing requires torchvision. Install/update dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or, in this virtual environment:\n"
            "  python -m pip install torchvision"
        ) from exc

    processor_source = str(source) if source else args.model_name
    kwargs: Dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.image_longest_edge > 0:
        kwargs["size"] = {"longest_edge": args.image_longest_edge}

    try:
        processor = AutoProcessor.from_pretrained(processor_source, **kwargs)
    except TypeError:
        kwargs.pop("size", None)
        processor = AutoProcessor.from_pretrained(processor_source, **kwargs)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    return processor


def from_pretrained_model(model_name: str, dtype: torch.dtype, args: argparse.Namespace) -> Any:
    kwargs: Dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.attn_implementation:
        kwargs["_attn_implementation"] = args.attn_implementation

    if dtype == torch.float32:
        return AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)

    try:
        return AutoModelForVision2Seq.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def load_base_model(args: argparse.Namespace, dtype: torch.dtype) -> Any:
    model = from_pretrained_model(args.model_name, dtype=dtype, args=args)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def save_competition_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {field: getattr(args, field) for field in CHECKPOINT_CONFIG_FIELDS if hasattr(args, field)}
    (output_dir / COMPETITION_CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_checkpoint_config(args: argparse.Namespace, checkpoint: Optional[Path]) -> None:
    if checkpoint is None:
        return
    config_path = checkpoint / COMPETITION_CONFIG_FILE
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    inherited: List[str] = []
    for field in CHECKPOINT_CONFIG_FIELDS:
        if field not in config or not hasattr(args, field):
            continue
        current_value = getattr(args, field)
        default_value = CHECKPOINT_CONFIG_DEFAULTS[field]
        if current_value == default_value:
            setattr(args, field, config[field])
            inherited.append(field)

    if inherited:
        print(f"Loaded checkpoint competition config from {config_path} ({', '.join(inherited)}).")


def load_model_for_inference(args: argparse.Namespace, dtype: torch.dtype) -> Tuple[Any, Any, torch.device]:
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    load_checkpoint_config(args, checkpoint)
    processor_source = checkpoint if checkpoint and (checkpoint / "processor_config.json").exists() else None
    processor = load_processor(args, source=processor_source)
    model = load_base_model(args, dtype=dtype)
    if checkpoint:
        try:
            model = PeftModel.from_pretrained(model, str(checkpoint), local_files_only=args.local_files_only)
        except TypeError:
            model = PeftModel.from_pretrained(model, str(checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device


class BalancedTrainer(Trainer):
    def __init__(self, *args: Any, sample_weights: Optional[Sequence[float]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sample_weights = None
        if sample_weights is not None:
            self.sample_weights = torch.tensor(list(sample_weights), dtype=torch.double)

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if self.sample_weights is None:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if len(dataset) != len(self.sample_weights):
            raise ValueError(
                f"Weighted sampler length mismatch: dataset has {len(dataset)} items but "
                f"{len(self.sample_weights)} sample weights were provided."
            )
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )


def find_lora_target_modules(
    model: Any,
    target_suffixes: Sequence[str],
    scope_keywords: Sequence[str],
) -> List[str]:
    suffixes = tuple(suffix.strip() for suffix in target_suffixes if suffix.strip())
    scopes = tuple(scope.strip() for scope in scope_keywords if scope.strip())
    exact_targets: List[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if suffixes and not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if scopes and not any(scope in name for scope in scopes):
            continue
        exact_targets.append(name)

    if exact_targets:
        return exact_targets

    return list(suffixes)


def count_trainable_parameters(model: Any) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, total


def add_lora_adapters(model: Any, args: argparse.Namespace) -> Any:
    target_suffixes = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
    scope_keywords = [part.strip() for part in args.lora_scope_keywords.split(",") if part.strip()]
    target_modules = find_lora_target_modules(model, target_suffixes, scope_keywords)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    trainable, total = count_trainable_parameters(model)
    print(f"LoRA target modules: {len(target_modules)}")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    if trainable > args.max_trainable_params:
        raise ValueError(
            f"Trainable parameter count {trainable:,} exceeds the limit of "
            f"{args.max_trainable_params:,}. Lower --lora-r or narrow --lora-scope-keywords."
        )
    return model


def make_processor_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {"do_image_splitting": args.do_image_splitting}


def make_prompt_builder(args: argparse.Namespace) -> PromptBuilder:
    return PromptBuilder(
        question_chars=args.question_chars,
        hint_chars=args.hint_chars,
        lecture_chars=args.lecture_chars,
        include_metadata=args.include_metadata,
        answer_format=args.answer_format,
    )


def resolve_train_csvs(args: argparse.Namespace, data_dir: Path) -> List[Path]:
    train_csvs = parse_path_list(args.train_csv) if args.train_csv else [data_dir / "train.csv"]
    if args.train_on_val:
        train_csvs.append(data_dir / "val.csv")
    return train_csvs


def build_sample_weights(items: Sequence[Dict[str, Any]], balance_sampler: str) -> Optional[List[float]]:
    if balance_sampler == "none":
        return None

    def group_key(item: Dict[str, Any]) -> Any:
        if balance_sampler == "answer":
            return int(item["answer"])
        if balance_sampler == "num_choices":
            return int(item["num_choices"])
        if balance_sampler == "joint":
            return (int(item["num_choices"]), int(item["answer"]))
        raise ValueError(f"Unsupported balance_sampler: {balance_sampler}")

    counts = Counter(group_key(item) for item in items)
    weights = [1.0 / counts[group_key(item)] for item in items]
    total = sum(weights)
    if total <= 0:
        return None
    scale = len(weights) / total
    return [weight * scale for weight in weights]


def dataqa(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    train_csv = Path(args.train_csv) if args.train_csv else data_dir / "train.csv"
    val_csv = Path(args.val_csv) if args.val_csv else data_dir / "val.csv"
    report = build_dataqa_report(data_dir=data_dir, train_csv=train_csv, val_csv=val_csv)

    train_summary = report["split_reports"]["train"]
    val_summary = report["split_reports"]["val"]
    overlap = report["cross_split"]
    val_overlap_counts = overlap["val_row_overlap_counts"]

    print(f"Train rows: {train_summary['rows']}; Val rows: {val_summary['rows']}")
    print(
        "Val overlap with train: "
        f"same_question={val_overlap_counts.get('same_question', 0)}, "
        f"same_question_choices={val_overlap_counts.get('same_question_choices', 0)}, "
        f"same_image_hash={val_overlap_counts.get('same_image_hash', 0)}, "
        f"same_image_and_question={val_overlap_counts.get('same_image_and_question', 0)}, "
        f"exact_duplicates={val_overlap_counts.get('exact_duplicates', 0)}"
    )
    print(
        "Rarest joint labels in train: "
        f"{sorted(train_summary['joint_answer_dist'].items(), key=lambda part: part[1])[:5]}"
    )

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote data QA report to {output_path}")


def strictsplit(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    source_csvs = parse_path_list(args.source_csvs) if args.source_csvs else [data_dir / "train.csv", data_dir / "val.csv"]
    if len(source_csvs) < 2:
        raise ValueError("Strict split generation expects at least two labeled source CSVs.")

    image_hash_cache: Dict[Path, str] = {}
    combined_records: List[Dict[str, Any]] = []
    for csv_path in source_csvs:
        _, records = load_records_for_analysis(
            Path(csv_path),
            data_dir,
            split_name=Path(csv_path).stem,
            image_hash_cache=image_hash_cache,
        )
        combined_records.extend(records)

    group_keys = parse_group_key_list(args.group_by)
    stratify_keys = parse_stratify_key_list(args.stratify_by)
    grouped_indices = group_records_for_strict_split(combined_records, group_keys)
    train_records, val_records, split_summary = build_group_split(
        combined_records,
        grouped_indices,
        stratify_keys=stratify_keys,
        val_ratio=args.val_ratio,
        seed=args.seed,
        search_trials=args.search_trials,
    )

    output_dir = Path(args.output_dir)
    train_output = output_dir / "train.csv"
    val_output = output_dir / "val.csv"
    write_records_to_csv(train_output, train_records)
    write_records_to_csv(val_output, val_records)

    qa_report = build_dataqa_report(data_dir=data_dir, train_csv=train_output, val_csv=val_output)
    qa_report["strict_split"] = {
        **split_summary,
        "source_csvs": [str(path) for path in source_csvs],
        "group_by": group_keys,
        "stratify_by": stratify_keys,
        "seed": args.seed,
    }

    qa_output_path = Path(args.qa_output_path) if args.qa_output_path else output_dir / "qa_report.json"
    qa_output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_output_path.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")

    overlap = qa_report["cross_split"]["val_row_overlap_counts"]
    print(f"Wrote strict split train CSV to {train_output}")
    print(f"Wrote strict split val CSV to {val_output}")
    print(
        f"Strict split rows: train={len(train_records)}, val={len(val_records)} "
        f"(target val={split_summary['target_val_rows']})"
    )
    print(
        "Leakage check on strict val: "
        f"same_question={overlap.get('same_question', 0)}, "
        f"same_question_choices={overlap.get('same_question_choices', 0)}, "
        f"same_image_hash={overlap.get('same_image_hash', 0)}, "
        f"same_image_and_question={overlap.get('same_image_and_question', 0)}"
    )
    print(f"Wrote strict split QA report to {qa_output_path}")


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.select_best_checkpoint and not args.save_final:
        raise ValueError("Checkpoint selection requires --save-final so the chosen adapter can be kept.")
    require_cuda(args, "train")
    configure_cuda_performance(args)
    dtype = select_dtype(args.dtype)
    processor = load_processor(args)
    model = load_base_model(args, dtype=dtype)
    model = add_lora_adapters(model, args)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_dir = Path(args.data_dir)
    prompt_builder = make_prompt_builder(args)
    train_csvs = resolve_train_csvs(args, data_dir)
    train_dataset = ScienceQADataset(
        train_csvs,
        data_dir,
        prompt_builder,
        require_answer=True,
        cache_images=args.cache_images,
    )
    collator = VlmDataCollator(
        processor=processor,
        prompt_builder=prompt_builder,
        max_length=args.max_length,
        processor_kwargs=make_processor_kwargs(args),
        answer_format=args.answer_format,
    )
    sample_weights = build_sample_weights(train_dataset.items, args.balance_sampler)
    if sample_weights is not None:
        print(f"Using weighted sampling with mode={args.balance_sampler}.")

    output_dir = Path(args.output_dir)
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "bf16": dtype == torch.bfloat16 and torch.cuda.is_available(),
        "fp16": dtype == torch.float16 and torch.cuda.is_available(),
        "tf32": bool(args.tf32),
        "optim": "adamw_torch",
        "remove_unused_columns": False,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_persistent_workers": args.num_workers > 0 and args.persistent_workers,
        "report_to": "none",
    }
    if args.num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = args.prefetch_factor
    training_args = TrainingArguments(**training_kwargs)
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        sample_weights=sample_weights,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA memory allocated: {peak_gb:.2f} GB")

    final_dir = output_dir / "final_adapter"
    if args.save_final:
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        save_competition_config(args, final_dir)
        print(f"Saved final adapter and processor to {final_dir}")

    needs_reload = args.select_best_checkpoint or not args.skip_eval
    if needs_reload:
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_final and args.select_best_checkpoint:
        checkpoint_dirs = sorted(
            [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        candidate_dirs = checkpoint_dirs + [final_dir]
        if not candidate_dirs:
            raise ValueError("No candidate checkpoints were found for validation-based selection.")

        best_dir = final_dir
        best_accuracy = -1.0
        for candidate_dir in candidate_dirs:
            accuracy, _, split_name = evaluate_checkpoint(args, checkpoint=str(candidate_dir), split="val")
            print(f"{split_name} accuracy for {candidate_dir.name}: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_dir = candidate_dir

        if best_dir != final_dir:
            shutil.rmtree(final_dir, ignore_errors=True)
            shutil.copytree(best_dir, final_dir)
            processor.save_pretrained(final_dir)
            save_competition_config(args, final_dir)
            print(f"Selected best checkpoint {best_dir} and copied it to {final_dir}")
        (output_dir / "best_checkpoint.txt").write_text(f"{best_dir}\n{best_accuracy:.6f}\n", encoding="utf-8")

    if not args.skip_eval:
        if not args.save_final:
            raise ValueError("Evaluation after training requires --save-final so the adapter can be reloaded.")

        eval_args = argparse.Namespace(**vars(args))
        eval_args.checkpoint = str(final_dir)
        eval_csv_path, eval_split_name = resolve_eval_csv_path(eval_args, split="val")
        eval_args.eval_csv = str(eval_csv_path)
        eval_args.split = eval_split_name
        eval_args.predictions_path = str(output_dir / f"{eval_split_name}_predictions.csv")
        evaluate(eval_args)


def score_item(
    model: Any,
    processor: Any,
    device: torch.device,
    item: Dict[str, Any],
    prompt_builder: PromptBuilder,
    max_length: int,
    processor_kwargs: Dict[str, Any],
    answer_format: str,
    score_normalization: str,
) -> List[float]:
    image = load_rgb_image(item["image_path"], item.get("image_bytes"))
    user_text = prompt_builder.build_user_text(item)
    prompt_text = apply_chat_template(processor, build_messages(user_text), add_generation_prompt=True)

    full_texts = [
        apply_chat_template(
            processor,
            build_messages(user_text, answer=format_answer_target(item, candidate, answer_format)),
            add_generation_prompt=False,
        )
        for candidate in range(int(item["num_choices"]))
    ]
    images = [image] * len(full_texts)
    prompt_texts = [prompt_text] * len(full_texts)

    full_inputs = call_processor(
        processor,
        full_texts,
        images,
        max_length=max_length,
        processor_kwargs=processor_kwargs,
    )
    prompt_inputs = call_processor(
        processor,
        prompt_texts,
        images,
        max_length=max_length,
        processor_kwargs=processor_kwargs,
    )

    labels = full_inputs["input_ids"].clone()
    labels[full_inputs["attention_mask"] == 0] = -100
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    for row_index, prompt_length in enumerate(prompt_lengths):
        labels[row_index, : int(prompt_length)] = -100

    full_inputs = full_inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(**full_inputs)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_scores = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    scores = (token_scores * mask).sum(dim=1)
    normalization_mode = resolve_score_normalization(answer_format, score_normalization)
    if normalization_mode == "mean":
        token_counts = mask.sum(dim=1).clamp(min=1)
        scores = scores / token_counts
    score_list = [float(score.detach().cpu()) for score in scores]
    return score_list


def load_score_bias(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    bias_path = Path(path)
    config = json.loads(bias_path.read_text(encoding="utf-8"))
    mode = config.get("mode")
    if mode not in {"answer", "joint"}:
        raise ValueError(f"Unsupported score bias mode {mode!r} in {bias_path}")
    return config


def apply_score_biases(
    scores: Sequence[float],
    item: Dict[str, Any],
    score_bias: Optional[Dict[str, Any]],
) -> List[float]:
    adjusted = [float(score) for score in scores]
    if not score_bias:
        return adjusted

    mode = score_bias["mode"]
    biases = score_bias.get("biases", {})
    num_choices = str(int(item["num_choices"]))
    for candidate in range(len(adjusted)):
        candidate_key = str(candidate)
        if mode == "answer":
            bias = biases.get(candidate_key, 0.0)
        else:
            bias = biases.get(num_choices, {}).get(candidate_key, 0.0)
        adjusted[candidate] += float(bias)
    return adjusted


def predict_from_scores(
    scores: Sequence[float],
    item: Dict[str, Any],
    score_bias: Optional[Dict[str, Any]],
) -> Tuple[int, List[float]]:
    adjusted_scores = apply_score_biases(scores, item, score_bias)
    return int(np.argmax(adjusted_scores)), adjusted_scores


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint: Optional[str],
    split: str,
) -> Tuple[float, List[Dict[str, Any]], str]:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.checkpoint = checkpoint
    eval_args.split = split
    dtype = select_dtype(eval_args.dtype)
    model, processor, device = load_model_for_inference(eval_args, dtype=dtype)
    data_dir = Path(eval_args.data_dir)
    prompt_builder = make_prompt_builder(eval_args)
    csv_path, split_name = resolve_eval_csv_path(eval_args, split=split)
    dataset = ScienceQADataset(
        csv_path,
        data_dir,
        prompt_builder,
        require_answer=True,
        cache_images=eval_args.cache_images,
    )
    items = dataset.items[: eval_args.limit] if eval_args.limit and eval_args.limit > 0 else dataset.items

    rows = []
    correct = 0
    processor_kwargs = make_processor_kwargs(eval_args)
    score_bias = load_score_bias(getattr(eval_args, "score_bias_path", None))
    for item in tqdm(items, desc=f"Scoring {split_name} ({Path(checkpoint).name if checkpoint else 'base'})"):
        scores = score_item(
            model=model,
            processor=processor,
            device=device,
            item=item,
            prompt_builder=prompt_builder,
            max_length=eval_args.max_length,
            processor_kwargs=processor_kwargs,
            answer_format=eval_args.answer_format,
            score_normalization=eval_args.score_normalization,
        )
        prediction, adjusted_scores = predict_from_scores(scores, item, score_bias)
        answer = int(item["answer"])
        correct += int(prediction == answer)
        row = {
            "id": item["id"],
            "answer": answer,
            "prediction": prediction,
            "correct": int(prediction == answer),
            "scores": json.dumps(scores),
        }
        if score_bias:
            row["adjusted_scores"] = json.dumps(adjusted_scores)
        rows.append(row)

    accuracy = correct / max(len(items), 1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accuracy, rows, split_name


def evaluate(args: argparse.Namespace) -> None:
    require_cuda(args, "eval")
    configure_cuda_performance(args)
    accuracy, rows, split_name = evaluate_checkpoint(args, checkpoint=args.checkpoint, split=args.split)
    correct = sum(int(row["correct"]) for row in rows)
    print(f"{split_name} accuracy: {accuracy:.4f} ({correct}/{len(rows)})")
    if args.predictions_path:
        output_path = Path(args.predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Wrote predictions to {output_path}")


def load_prediction_score_rows(predictions_path: Path) -> List[Dict[str, Any]]:
    frame = pd.read_csv(predictions_path)
    required_columns = {"id", "answer", "scores"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{predictions_path} is missing required columns: {sorted(missing_columns)}")

    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        scores = [float(score) for score in json.loads(clean_text(row["scores"]))]
        if not scores:
            continue
        answer = int(row["answer"])
        if answer < 0 or answer >= len(scores):
            raise ValueError(f"{row['id']} answer {answer} is outside the scored range 0..{len(scores) - 1}")
        rows.append(
            {
                "id": clean_text(row["id"]),
                "answer": answer,
                "num_choices": len(scores),
                "scores": scores,
            }
        )
    if not rows:
        raise ValueError(f"No scored prediction rows were loaded from {predictions_path}")
    return rows


def build_empty_score_bias(mode: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if mode == "answer":
        max_choice = max(len(row["scores"]) for row in rows)
        return {"mode": mode, "biases": {str(candidate): 0.0 for candidate in range(max_choice)}}
    if mode == "joint":
        biases: Dict[str, Dict[str, float]] = {}
        for row in rows:
            num_choices = str(int(row["num_choices"]))
            if num_choices not in biases:
                biases[num_choices] = {str(candidate): 0.0 for candidate in range(int(row["num_choices"]))}
        return {"mode": mode, "biases": dict(sorted(biases.items(), key=lambda part: int(part[0])))}
    raise ValueError(f"Unsupported score bias mode: {mode}")


def score_bias_keys(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    mode = config["mode"]
    if mode == "answer":
        return [("answer", key) for key in sorted(config["biases"], key=int) if key != "0"]
    keys: List[Tuple[str, str]] = []
    for num_choices, biases in sorted(config["biases"].items(), key=lambda part: int(part[0])):
        keys.extend((num_choices, key) for key in sorted(biases, key=int) if key != "0")
    return keys


def get_score_bias_value(config: Dict[str, Any], key: Tuple[str, str]) -> float:
    group, candidate = key
    if config["mode"] == "answer":
        return float(config["biases"].get(candidate, 0.0))
    return float(config["biases"].get(group, {}).get(candidate, 0.0))


def set_score_bias_value(config: Dict[str, Any], key: Tuple[str, str], value: float) -> None:
    group, candidate = key
    if config["mode"] == "answer":
        config["biases"][candidate] = float(value)
    else:
        config["biases"][group][candidate] = float(value)


def score_bias_accuracy(rows: Sequence[Dict[str, Any]], config: Optional[Dict[str, Any]]) -> Tuple[float, int]:
    correct = 0
    for row in rows:
        item = {"num_choices": row["num_choices"]}
        prediction, _ = predict_from_scores(row["scores"], item, config)
        correct += int(prediction == int(row["answer"]))
    return correct / max(len(rows), 1), correct


def calibrate_score_bias(args: argparse.Namespace) -> None:
    rows = load_prediction_score_rows(Path(args.predictions_path))
    values = [round(float(value), 6) for value in np.arange(args.bias_min, args.bias_max + args.bias_step * 0.5, args.bias_step)]
    if not values:
        raise ValueError("Calibration grid is empty; check --bias-min, --bias-max, and --bias-step.")

    config = build_empty_score_bias(args.mode, rows)
    before_accuracy, before_correct = score_bias_accuracy(rows, None)
    best_accuracy, best_correct = score_bias_accuracy(rows, config)
    keys = score_bias_keys(config)

    for _ in range(args.max_iterations):
        improved = False
        for key in keys:
            current_value = get_score_bias_value(config, key)
            local_best_value = current_value
            local_best_accuracy = best_accuracy
            local_best_correct = best_correct

            for value in values:
                set_score_bias_value(config, key, value)
                accuracy, correct = score_bias_accuracy(rows, config)
                if accuracy > local_best_accuracy or (
                    accuracy == local_best_accuracy and abs(value) < abs(local_best_value)
                ):
                    local_best_value = value
                    local_best_accuracy = accuracy
                    local_best_correct = correct

            set_score_bias_value(config, key, local_best_value)
            if local_best_accuracy > best_accuracy:
                improved = True
                best_accuracy = local_best_accuracy
                best_correct = local_best_correct

        if not improved:
            break

    config.update(
        {
            "source_predictions": str(args.predictions_path),
            "rows": len(rows),
            "accuracy_before": before_accuracy,
            "correct_before": before_correct,
            "accuracy_after": best_accuracy,
            "correct_after": best_correct,
            "grid": {
                "min": args.bias_min,
                "max": args.bias_max,
                "step": args.bias_step,
                "max_iterations": args.max_iterations,
            },
        }
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"Calibration accuracy: {before_accuracy:.4f} ({before_correct}/{len(rows)}) -> "
        f"{best_accuracy:.4f} ({best_correct}/{len(rows)})"
    )
    print(f"Wrote score bias config to {output_path}")


def validate_submission_file(submission_path: Path, test_dataset: ScienceQADataset) -> None:
    submission = pd.read_csv(submission_path)
    expected_columns = ["id", "answer"]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns}, got {list(submission.columns)}")

    expected_ids = [item["id"] for item in test_dataset.items]
    actual_ids = submission["id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError("Submission ids or order do not match data/test.csv")

    answers = submission["answer"].tolist()
    for item, answer in zip(test_dataset.items, answers):
        if not isinstance(answer, (int, np.integer)) and not str(answer).isdigit():
            raise ValueError(f"{item['id']} has non-integer answer {answer!r}")
        answer_int = int(answer)
        if answer_int < 0 or answer_int >= int(item["num_choices"]):
            raise ValueError(
                f"{item['id']} answer {answer_int} is outside valid range 0..{int(item['num_choices']) - 1}"
            )


def predict(args: argparse.Namespace) -> None:
    require_cuda(args, "predict")
    configure_cuda_performance(args)
    dtype = select_dtype(args.dtype)
    model, processor, device = load_model_for_inference(args, dtype=dtype)
    data_dir = Path(args.data_dir)
    prompt_builder = make_prompt_builder(args)
    test_dataset = ScienceQADataset(
        data_dir / "test.csv",
        data_dir,
        prompt_builder,
        require_answer=False,
        cache_images=args.cache_images,
    )
    items = test_dataset.items[: args.limit] if args.limit and args.limit > 0 else test_dataset.items

    rows = []
    detail_rows = []
    processor_kwargs = make_processor_kwargs(args)
    score_bias = load_score_bias(getattr(args, "score_bias_path", None))
    for item in tqdm(items, desc="Scoring test"):
        scores = score_item(
            model=model,
            processor=processor,
            device=device,
            item=item,
            prompt_builder=prompt_builder,
            max_length=args.max_length,
            processor_kwargs=processor_kwargs,
            answer_format=args.answer_format,
            score_normalization=args.score_normalization,
        )
        prediction, adjusted_scores = predict_from_scores(scores, item, score_bias)
        rows.append({"id": item["id"], "answer": prediction})
        if getattr(args, "score_details_path", None):
            detail_row = {
                "id": item["id"],
                "num_choices": int(item["num_choices"]),
                "prediction": prediction,
                "scores": json.dumps(scores),
            }
            if score_bias:
                detail_row["adjusted_scores"] = json.dumps(adjusted_scores)
            detail_rows.append(detail_row)

    submission_path = Path(args.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "answer"]).to_csv(submission_path, index=False)
    if not args.limit:
        validate_submission_file(submission_path, test_dataset)
    print(f"Wrote submission to {submission_path}")
    if getattr(args, "score_details_path", None):
        details_path = Path(args.score_details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detail_rows).to_csv(details_path, index=False)
        print(f"Wrote score details to {details_path}")


def download(args: argparse.Namespace) -> None:
    dtype = select_dtype(args.dtype)
    _ = load_processor(args)
    model = load_base_model(args, dtype=dtype)
    trainable, total = count_trainable_parameters(model)
    print(f"Cached {args.model_name}")
    print(f"Base model parameters: {total:,}; trainable before LoRA: {trainable:,}")


def parse_int_list(value: str) -> List[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"Expected a comma-separated integer list, got {value!r}")
    return parsed


def powershell_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def add_if(command: List[str], condition: bool, flag: str) -> None:
    if condition:
        command.append(flag)


def build_train_command_for_autotune(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script_path),
        "train",
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(args.max_steps),
        "--skip-eval",
        "--no-save-final",
        "--batch-size",
        str(candidate["batch_size"]),
        "--grad-accum-steps",
        str(candidate["grad_accum_steps"]),
        "--num-workers",
        str(candidate["num_workers"]),
        "--prefetch-factor",
        str(candidate["prefetch_factor"]),
        "--logging-steps",
        str(args.logging_steps),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-target-modules",
        args.lora_target_modules,
        "--lora-scope-keywords",
        args.lora_scope_keywords,
        "--max-trainable-params",
        str(args.max_trainable_params),
        "--model-name",
        args.model_name,
        "--dtype",
        args.dtype,
        "--image-longest-edge",
        str(args.image_longest_edge),
        "--max-length",
        str(args.max_length),
        "--question-chars",
        str(args.question_chars),
        "--hint-chars",
        str(args.hint_chars),
        "--lecture-chars",
        str(args.lecture_chars),
        "--answer-format",
        args.answer_format,
        "--score-normalization",
        args.score_normalization,
        "--cache-images",
        args.cache_images,
        "--data-dir",
        args.data_dir,
    ]
    if getattr(args, "train_csv", None):
        command.extend(["--train-csv", args.train_csv])
    add_if(command, args.train_on_val, "--train-on-val")
    if args.balance_sampler != "none":
        command.extend(["--balance-sampler", args.balance_sampler])

    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    add_if(command, args.local_files_only, "--local-files-only")
    add_if(command, args.include_metadata, "--include-metadata")
    add_if(command, args.allow_cpu, "--allow-cpu")
    add_if(command, args.tf32, "--tf32")
    add_if(command, not args.tf32, "--no-tf32")
    add_if(command, args.persistent_workers, "--persistent-workers")
    add_if(command, not args.persistent_workers, "--no-persistent-workers")
    add_if(command, candidate["gradient_checkpointing"], "--gradient-checkpointing")
    add_if(command, candidate["do_image_splitting"], "--do-image-splitting")
    return command


def parse_autotune_metrics(output: str) -> Dict[str, Optional[float]]:
    patterns = {
        "train_runtime": r"train_runtime['\"]?\s*:\s*'?([0-9.]+)",
        "train_samples_per_second": r"train_samples_per_second['\"]?\s*:\s*'?([0-9.]+)",
        "train_steps_per_second": r"train_steps_per_second['\"]?\s*:\s*'?([0-9.]+)",
        "train_loss": r"train_loss['\"]?\s*:\s*'?([0-9.]+)",
        "peak_cuda_gb": r"Peak CUDA memory allocated:\s*([0-9.]+)\s*GB",
    }
    metrics: Dict[str, Optional[float]] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        metrics[key] = float(matches[-1]) if matches else None
    return metrics


def run_autotune_candidate(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    run_index: int,
    output_root: Path,
) -> Dict[str, Any]:
    run_name = (
        f"run_{run_index:03d}_b{candidate['batch_size']}_ga{candidate['grad_accum_steps']}"
        f"_w{candidate['num_workers']}_p{candidate['prefetch_factor']}"
        f"_split{int(candidate['do_image_splitting'])}_gc{int(candidate['gradient_checkpointing'])}"
    )
    run_dir = output_root / "runs" / run_name
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command_for_autotune(args, candidate, run_dir)

    if args.dry_run:
        print(powershell_command(command))
        return {
            **candidate,
            "run": run_name,
            "status": "dry_run",
            "returncode": 0,
            "command": powershell_command(command),
        }

    print(f"\n[{run_index}] Benchmarking {run_name}")
    print(powershell_command(command))

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_lines: List[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        if args.verbose:
            print(line, end="")
    returncode = process.wait()

    output = "".join(output_lines)
    log_path = log_dir / f"{run_name}.log"
    log_path.write_text(output, encoding="utf-8")
    metrics = parse_autotune_metrics(output)
    status = "ok" if returncode == 0 and metrics["train_samples_per_second"] is not None else "failed"
    if "out of memory" in output.lower() or "cuda error: out of memory" in output.lower():
        status = "oom"

    if not args.keep_runs and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    result = {
        **candidate,
        **metrics,
        "run": run_name,
        "status": status,
        "returncode": returncode,
        "log_path": str(log_path),
        "command": powershell_command(command),
    }
    metric = result.get("train_samples_per_second")
    peak = result.get("peak_cuda_gb")
    metric_text = f"{metric:.3f} samples/s" if metric is not None else "no metric"
    peak_text = f", peak {peak:.2f} GB" if peak is not None else ""
    print(f"[{run_index}] {status}: {metric_text}{peak_text}")
    return result


def build_autotune_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    batch_sizes = parse_int_list(args.batch_sizes)
    grad_accum_steps = parse_int_list(args.grad_accum_steps_list)
    num_workers = parse_int_list(args.num_workers_list)
    prefetch_factors = parse_int_list(args.prefetch_factors)
    image_splitting_values = [False, True] if args.test_image_splitting else [args.do_image_splitting]
    checkpointing_values = [False, True] if args.test_gradient_checkpointing else [args.gradient_checkpointing]

    candidates = []
    for batch_size in batch_sizes:
        for grad_accum_step in grad_accum_steps:
            for worker_count in num_workers:
                for prefetch_factor in prefetch_factors:
                    for do_image_splitting in image_splitting_values:
                        for gradient_checkpointing in checkpointing_values:
                            candidates.append(
                                {
                                    "batch_size": batch_size,
                                    "grad_accum_steps": grad_accum_step,
                                    "num_workers": worker_count,
                                    "prefetch_factor": prefetch_factor,
                                    "do_image_splitting": do_image_splitting,
                                    "gradient_checkpointing": gradient_checkpointing,
                                }
                            )
    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]
    return candidates


def recommended_train_command(args: argparse.Namespace, best: Dict[str, Any]) -> List[str]:
    output_dir = args.recommended_output_dir or "outputs/smolvlm_lora_autotuned"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "train",
        "--output-dir",
        output_dir,
        "--batch-size",
        str(best["batch_size"]),
        "--grad-accum-steps",
        str(best["grad_accum_steps"]),
        "--num-workers",
        str(best["num_workers"]),
        "--prefetch-factor",
        str(best["prefetch_factor"]),
        "--cache-images",
        args.cache_images,
        "--model-name",
        args.model_name,
        "--data-dir",
        args.data_dir,
        "--image-longest-edge",
        str(args.image_longest_edge),
        "--max-length",
        str(args.max_length),
        "--question-chars",
        str(args.question_chars),
        "--hint-chars",
        str(args.hint_chars),
        "--lecture-chars",
        str(args.lecture_chars),
        "--answer-format",
        args.answer_format,
        "--score-normalization",
        args.score_normalization,
    ]
    add_if(command, best["do_image_splitting"], "--do-image-splitting")
    add_if(command, best["gradient_checkpointing"], "--gradient-checkpointing")
    add_if(command, args.include_metadata, "--include-metadata")
    add_if(command, args.local_files_only, "--local-files-only")
    add_if(command, args.tf32, "--tf32")
    add_if(command, not args.tf32, "--no-tf32")
    add_if(command, args.persistent_workers, "--persistent-workers")
    add_if(command, not args.persistent_workers, "--no-persistent-workers")
    return command


def autotune(args: argparse.Namespace) -> None:
    require_cuda(args, "autotune")
    candidates = build_autotune_candidates(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Autotuning {len(candidates)} candidate(s) for {args.max_steps} steps each.")
    if not args.keep_runs:
        print("Candidate model directories will be removed after each run; logs and results are kept.")

    results = []
    for index, candidate in enumerate(candidates, start=1):
        result = run_autotune_candidate(args, candidate, index, output_root)
        results.append(result)
        pd.DataFrame(results).to_csv(output_root / "results.csv", index=False)

    if args.dry_run:
        print(f"Dry run listed {len(results)} command(s).")
        return

    ok_results = [
        result
        for result in results
        if result["status"] == "ok" and result.get("train_samples_per_second") is not None
    ]
    if not ok_results:
        raise RuntimeError(f"No successful autotune candidates. Check logs in {output_root / 'logs'}.")

    best = max(ok_results, key=lambda result: float(result["train_samples_per_second"]))
    pd.DataFrame(results).sort_values(
        by=["train_samples_per_second"],
        ascending=False,
        na_position="last",
    ).to_csv(output_root / "results_ranked.csv", index=False)

    print("\nBest throughput candidate:")
    print(
        f"  batch_size={best['batch_size']}, grad_accum_steps={best['grad_accum_steps']}, "
        f"num_workers={best['num_workers']}, prefetch_factor={best['prefetch_factor']}, "
        f"image_splitting={best['do_image_splitting']}, "
        f"gradient_checkpointing={best['gradient_checkpointing']}"
    )
    print(
        f"  train_samples_per_second={best['train_samples_per_second']:.3f}, "
        f"train_steps_per_second={best['train_steps_per_second']:.3f}, "
        f"peak_cuda_gb={best['peak_cuda_gb']}"
    )
    print(f"\nResults: {output_root / 'results_ranked.csv'}")
    print("\nRecommended full training command:")
    print(powershell_command(recommended_train_command(args, best)))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Allowed model or a local mirror of it.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only locally cached model files.")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention backend.")
    parser.add_argument("--image-longest-edge", type=int, default=512, help="Lower values reduce memory use.")
    parser.add_argument("--do-image-splitting", action="store_true", help="Enable SmolVLM image splitting.")
    parser.add_argument("--max-length", type=int, default=2048, help="Token max length; 0 disables token truncation.")
    parser.add_argument("--question-chars", type=int, default=2000)
    parser.add_argument("--hint-chars", type=int, default=2500)
    parser.add_argument("--lecture-chars", type=int, default=2500)
    parser.add_argument(
        "--answer-format",
        choices=["index", "choice_text", "index_choice"],
        default="index",
        help="Target format used for training and candidate scoring.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=["auto", "sum", "mean"],
        default="auto",
        help="How to compare candidate target scores during eval/predict.",
    )
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--cache-images",
        choices=["none", "bytes"],
        default="none",
        help="Cache split images in RAM as compressed bytes to reduce disk/OneDrive reads.",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="Allow train/eval/predict to run without CUDA.")
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TF32 matmul on NVIDIA GPUs for faster float32 operations.",
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data")


def add_score_bias_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--score-bias-path",
        default=None,
        help="Optional JSON score-bias config from calibrate-bias to add to candidate scores.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM-500M-Instruct with LoRA and create the course submission CSV."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Cache the allowed model and processor.")
    add_common_args(download_parser)

    dataqa_parser = subparsers.add_parser("dataqa", help="Audit labeled CSVs for quality, imbalance, and split overlap.")
    add_data_args(dataqa_parser)
    dataqa_parser.add_argument("--train-csv", default=None, help="Training CSV to audit; defaults to data/train.csv.")
    dataqa_parser.add_argument("--val-csv", default=None, help="Validation CSV to audit; defaults to data/val.csv.")
    dataqa_parser.add_argument(
        "--output-path",
        default="outputs/dataqa/report.json",
        help="Where to write the full JSON QA report.",
    )

    strictsplit_parser = subparsers.add_parser(
        "strictsplit",
        help="Create a leakage-aware local train/val split from the labeled data.",
    )
    add_data_args(strictsplit_parser)
    strictsplit_parser.add_argument(
        "--source-csvs",
        default=None,
        help="Comma-separated labeled CSVs to pool before splitting; defaults to data/train.csv,data/val.csv.",
    )
    strictsplit_parser.add_argument("--output-dir", default="data/strict_split")
    strictsplit_parser.add_argument("--val-ratio", type=float, default=0.25)
    strictsplit_parser.add_argument(
        "--group-by",
        default="image_hash,question",
        help=f"Comma-separated grouping keys from: {', '.join(STRICT_SPLIT_GROUP_KEYS)}.",
    )
    strictsplit_parser.add_argument(
        "--stratify-by",
        default="num_choices_answer,topic,category",
        help=f"Comma-separated stratification keys from: {', '.join(STRICT_SPLIT_STRATIFY_KEYS)}.",
    )
    strictsplit_parser.add_argument(
        "--search-trials",
        type=int,
        default=500,
        help="Random candidate splits to try after the greedy split; higher is slower but usually more balanced.",
    )
    strictsplit_parser.add_argument("--seed", type=int, default=42)
    strictsplit_parser.add_argument(
        "--qa-output-path",
        default=None,
        help="Optional JSON report path; defaults to <output-dir>/qa_report.json.",
    )

    train_parser = subparsers.add_parser("train", help="Train LoRA adapters.")
    add_common_args(train_parser)
    add_data_args(train_parser)
    train_parser.add_argument(
        "--train-csv",
        default=None,
        help="Optional training CSV path, or a comma-separated list of CSV paths.",
    )
    train_parser.add_argument(
        "--eval-csv",
        default=None,
        help="Optional labeled CSV for post-train evaluation and best-checkpoint selection; defaults to data/val.csv.",
    )
    train_parser.add_argument(
        "--train-on-val",
        action="store_true",
        help="Append data/val.csv to the training data after hyperparameter tuning.",
    )
    train_parser.add_argument("--output-dir", default="outputs/smolvlm_lora")
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs for short benchmark runs.")
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--grad-accum-steps", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--logging-steps", type=int, default=10)
    train_parser.add_argument("--save-total-limit", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--prefetch-factor", type=int, default=4)
    train_parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep dataloader workers alive between epochs when --num-workers > 0.",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--resume-from-checkpoint", default=None)
    train_parser.add_argument("--skip-eval", action="store_true")
    train_parser.add_argument(
        "--select-best-checkpoint",
        action="store_true",
        help="Score saved checkpoints on val and keep the best one as final_adapter.",
    )
    train_parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save final_adapter at the end of training. Disable for benchmark/autotune runs.",
    )
    train_parser.add_argument("--limit", type=int, default=0, help="Evaluation limit after training; 0 means full val.")
    train_parser.add_argument("--split", default="val")
    train_parser.add_argument("--predictions-path", default=None)
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--lora-r", type=int, default=4)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument(
        "--balance-sampler",
        choices=["none", "answer", "num_choices", "joint"],
        default="none",
        help="Reweight training samples to focus more on underrepresented answer or choice-count groups.",
    )
    train_parser.add_argument("--lora-target-modules", default=",".join(DEFAULT_TARGET_SUFFIXES))
    train_parser.add_argument("--lora-scope-keywords", default=",".join(DEFAULT_TEXT_SCOPES))
    train_parser.add_argument("--max-trainable-params", type=int, default=5_000_000)

    eval_parser = subparsers.add_parser("eval", help="Evaluate an adapter on train or val by answer scoring.")
    add_common_args(eval_parser)
    add_data_args(eval_parser)
    eval_parser.add_argument("--checkpoint", default="outputs/smolvlm_lora/final_adapter")
    eval_parser.add_argument("--split", choices=["train", "val"], default="val")
    eval_parser.add_argument(
        "--eval-csv",
        default=None,
        help="Optional labeled CSV to score instead of data/<split>.csv.",
    )
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--predictions-path", default="outputs/smolvlm_lora/val_predictions.csv")
    add_score_bias_args(eval_parser)

    calibrate_parser = subparsers.add_parser(
        "calibrate-bias",
        help="Fit additive answer-index score biases from an eval predictions CSV.",
    )
    calibrate_parser.add_argument("--predictions-path", required=True)
    calibrate_parser.add_argument("--output-path", required=True)
    calibrate_parser.add_argument(
        "--mode",
        choices=["answer", "joint"],
        default="joint",
        help="Fit one bias per answer index, or per (num_choices, answer index).",
    )
    calibrate_parser.add_argument("--bias-min", type=float, default=-1.0)
    calibrate_parser.add_argument("--bias-max", type=float, default=1.0)
    calibrate_parser.add_argument("--bias-step", type=float, default=0.05)
    calibrate_parser.add_argument("--max-iterations", type=int, default=3)

    predict_parser = subparsers.add_parser("predict", help="Create submission.csv for test.csv.")
    add_common_args(predict_parser)
    add_data_args(predict_parser)
    predict_parser.add_argument("--checkpoint", default="outputs/smolvlm_lora/final_adapter")
    predict_parser.add_argument("--submission-path", default="submission/submission.csv")
    predict_parser.add_argument("--score-details-path", default=None)
    predict_parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means full test set.")
    add_score_bias_args(predict_parser)

    autotune_parser = subparsers.add_parser("autotune", help="Benchmark hardware-facing training settings.")
    add_common_args(autotune_parser)
    add_data_args(autotune_parser)
    autotune_parser.set_defaults(cache_images="bytes")
    autotune_parser.add_argument("--output-dir", default="outputs/autotune")
    autotune_parser.add_argument("--recommended-output-dir", default="outputs/smolvlm_lora_autotuned")
    autotune_parser.add_argument("--max-steps", type=int, default=30)
    autotune_parser.add_argument("--batch-sizes", default="8,12,16")
    autotune_parser.add_argument("--grad-accum-steps-list", default="1")
    autotune_parser.add_argument("--num-workers-list", default="8,16")
    autotune_parser.add_argument("--prefetch-factors", default="4")
    autotune_parser.add_argument("--test-image-splitting", action="store_true")
    autotune_parser.add_argument("--test-gradient-checkpointing", action="store_true")
    autotune_parser.add_argument("--gradient-checkpointing", action="store_true")
    autotune_parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    autotune_parser.add_argument("--epochs", type=float, default=3.0)
    autotune_parser.add_argument("--learning-rate", type=float, default=2e-4)
    autotune_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    autotune_parser.add_argument("--weight-decay", type=float, default=0.0)
    autotune_parser.add_argument("--logging-steps", type=int, default=10)
    autotune_parser.add_argument("--lora-r", type=int, default=4)
    autotune_parser.add_argument("--lora-alpha", type=int, default=16)
    autotune_parser.add_argument("--lora-dropout", type=float, default=0.05)
    autotune_parser.add_argument("--lora-target-modules", default=",".join(DEFAULT_TARGET_SUFFIXES))
    autotune_parser.add_argument("--lora-scope-keywords", default=",".join(DEFAULT_TEXT_SCOPES))
    autotune_parser.add_argument("--max-trainable-params", type=int, default=5_000_000)
    autotune_parser.add_argument("--max-runs", type=int, default=0, help="Limit candidate count after expansion.")
    autotune_parser.add_argument("--keep-runs", action="store_true", help="Keep candidate model output directories.")
    autotune_parser.add_argument("--dry-run", action="store_true", help="Print candidate commands without running them.")
    autotune_parser.add_argument("--verbose", action="store_true", help="Stream full child training logs.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "download":
        download(args)
    elif args.command == "dataqa":
        dataqa(args)
    elif args.command == "strictsplit":
        strictsplit(args)
    elif args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "calibrate-bias":
        calibrate_score_bias(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "autotune":
        autotune(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
