from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
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
from torch.utils.data import Dataset
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


@dataclass
class PromptBuilder:
    question_chars: int = 2000
    hint_chars: int = 2500
    lecture_chars: int = 2500
    include_metadata: bool = False

    def build_user_text(self, item: Dict[str, Any]) -> str:
        choices = item["choices"]
        num_choices = int(item["num_choices"])
        choice_lines = [f"{idx}. {choice}" for idx, choice in enumerate(choices)]

        lines = [
            "You are solving a science multiple-choice question.",
            "Use only the provided image and text. Return only the 0-indexed answer integer.",
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

        lines.extend(["", f"Answer with one integer from 0 to {num_choices - 1}."])
        return "\n".join(lines)


def build_messages(user_text: str, answer: Optional[int] = None) -> List[Dict[str, Any]]:
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
                "content": [{"type": "text", "text": str(int(answer))}],
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
        csv_path: Path,
        data_dir: Path,
        prompt_builder: PromptBuilder,
        require_answer: bool,
        cache_images: str = "none",
    ) -> None:
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.prompt_builder = prompt_builder
        self.cache_images = cache_images
        frame = pd.read_csv(csv_path)
        self.items = [self._row_to_item(row, require_answer=require_answer) for _, row in frame.iterrows()]
        if self.cache_images == "bytes":
            cached_mb = sum(len(item["image_bytes"]) for item in self.items) / 1024**2
            print(f"Cached {len(self.items)} {csv_path.stem} images in RAM as compressed bytes ({cached_mb:.1f} MB).")

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
                    build_messages(user_text, answer=feature["answer"]),
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


def load_model_for_inference(args: argparse.Namespace, dtype: torch.dtype) -> Tuple[Any, Any, torch.device]:
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    processor = load_processor(args, source=checkpoint if checkpoint and checkpoint.exists() else None)
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
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
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
    train_dataset = ScienceQADataset(
        data_dir / "train.csv",
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
    )

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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
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
        print(f"Saved final adapter and processor to {final_dir}")

    if not args.skip_eval:
        if not args.save_final:
            raise ValueError("Evaluation after training requires --save-final so the adapter can be reloaded.")
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_args = argparse.Namespace(**vars(args))
        eval_args.checkpoint = str(final_dir)
        eval_args.split = "val"
        eval_args.predictions_path = str(output_dir / "val_predictions.csv")
        evaluate(eval_args)


def score_item(
    model: Any,
    processor: Any,
    device: torch.device,
    item: Dict[str, Any],
    prompt_builder: PromptBuilder,
    max_length: int,
    processor_kwargs: Dict[str, Any],
) -> Tuple[int, List[float]]:
    image = load_rgb_image(item["image_path"], item.get("image_bytes"))
    user_text = prompt_builder.build_user_text(item)
    prompt_text = apply_chat_template(processor, build_messages(user_text), add_generation_prompt=True)

    full_texts = [
        apply_chat_template(processor, build_messages(user_text, answer=candidate), add_generation_prompt=False)
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
    score_list = [float(score.detach().cpu()) for score in scores]
    prediction = int(np.argmax(score_list))
    return prediction, score_list


def evaluate(args: argparse.Namespace) -> None:
    require_cuda(args, "eval")
    configure_cuda_performance(args)
    dtype = select_dtype(args.dtype)
    model, processor, device = load_model_for_inference(args, dtype=dtype)
    data_dir = Path(args.data_dir)
    prompt_builder = make_prompt_builder(args)
    csv_path = data_dir / f"{args.split}.csv"
    dataset = ScienceQADataset(
        csv_path,
        data_dir,
        prompt_builder,
        require_answer=True,
        cache_images=args.cache_images,
    )
    items = dataset.items[: args.limit] if args.limit and args.limit > 0 else dataset.items

    rows = []
    correct = 0
    processor_kwargs = make_processor_kwargs(args)
    for item in tqdm(items, desc=f"Scoring {args.split}"):
        prediction, scores = score_item(
            model=model,
            processor=processor,
            device=device,
            item=item,
            prompt_builder=prompt_builder,
            max_length=args.max_length,
            processor_kwargs=processor_kwargs,
        )
        answer = int(item["answer"])
        correct += int(prediction == answer)
        rows.append(
            {
                "id": item["id"],
                "answer": answer,
                "prediction": prediction,
                "correct": int(prediction == answer),
                "scores": json.dumps(scores),
            }
        )

    accuracy = correct / max(len(items), 1)
    print(f"{args.split} accuracy: {accuracy:.4f} ({correct}/{len(items)})")
    if args.predictions_path:
        output_path = Path(args.predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Wrote predictions to {output_path}")


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
    processor_kwargs = make_processor_kwargs(args)
    for item in tqdm(items, desc="Scoring test"):
        prediction, _ = score_item(
            model=model,
            processor=processor,
            device=device,
            item=item,
            prompt_builder=prompt_builder,
            max_length=args.max_length,
            processor_kwargs=processor_kwargs,
        )
        rows.append({"id": item["id"], "answer": prediction})

    submission_path = Path(args.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "answer"]).to_csv(submission_path, index=False)
    if not args.limit:
        validate_submission_file(submission_path, test_dataset)
    print(f"Wrote submission to {submission_path}")


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
        "--cache-images",
        args.cache_images,
        "--data-dir",
        args.data_dir,
    ]

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM-500M-Instruct with LoRA and create the course submission CSV."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Cache the allowed model and processor.")
    add_common_args(download_parser)

    train_parser = subparsers.add_parser("train", help="Train LoRA adapters.")
    add_common_args(train_parser)
    add_data_args(train_parser)
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
    train_parser.add_argument("--lora-target-modules", default=",".join(DEFAULT_TARGET_SUFFIXES))
    train_parser.add_argument("--lora-scope-keywords", default=",".join(DEFAULT_TEXT_SCOPES))
    train_parser.add_argument("--max-trainable-params", type=int, default=5_000_000)

    eval_parser = subparsers.add_parser("eval", help="Evaluate an adapter on train or val by answer scoring.")
    add_common_args(eval_parser)
    add_data_args(eval_parser)
    eval_parser.add_argument("--checkpoint", default="outputs/smolvlm_lora/final_adapter")
    eval_parser.add_argument("--split", choices=["train", "val"], default="val")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--predictions-path", default="outputs/smolvlm_lora/val_predictions.csv")

    predict_parser = subparsers.add_parser("predict", help="Create submission.csv for test.csv.")
    add_common_args(predict_parser)
    add_data_args(predict_parser)
    predict_parser.add_argument("--checkpoint", default="outputs/smolvlm_lora/final_adapter")
    predict_parser.add_argument("--submission-path", default="submission/submission.csv")
    predict_parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means full test set.")

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
    elif args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "autotune":
        autotune(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
