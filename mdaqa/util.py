from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import AutoTokenizer

BOTTLE_DIM = 256
MASK_SCALE = 100
BATCH_SIZE = 8

# "SQuAD" "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web" "SearchQA"
SOURCE_DOMAIN = "SQuAD"
TARGET_DOMAIN = "NaturalQuestionsShort"

# "bert-base-uncased" "roberta-base" "distilroberta-base"
MODEL_CHECKPOINT = "roberta-base"
SEED = 8888
NUM_LOOPS = 5

MAXW_LENGTH = 384
STRIDE = 128
N_BEST = 20
MAX_ANSWER_LENGTH = 30

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoint"
SOURCE_CHECKPOINT_CANDIDATES = ("source_model", "BMW_source", "mask")

_TOKENIZER_CACHE: dict[str, Any] = {}
_METRIC = None


@dataclass(frozen=True)
class QaConfig:
    source_domain: str = SOURCE_DOMAIN
    target_domain: str = TARGET_DOMAIN
    model_checkpoint: str = MODEL_CHECKPOINT
    batch_size: int = BATCH_SIZE
    seed: int = SEED
    num_loops: int = NUM_LOOPS
    max_length: int = MAXW_LENGTH
    stride: int = STRIDE
    n_best: int = N_BEST
    max_answer_length: int = MAX_ANSWER_LENGTH
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR

    @property
    def experiment_name(self) -> str:
        return f"{self.source_domain}-{self.model_checkpoint}"

    @property
    def experiment_dir(self) -> Path:
        return self.checkpoint_dir / self.experiment_name

    @property
    def source_train_data(self) -> Path:
        return self.data_dir / f"{self.source_domain}_train"

    @property
    def target_train_data(self) -> Path:
        return self.data_dir / f"{self.target_domain}_train"

    @property
    def target_test_data(self) -> Path:
        return self.data_dir / f"{self.target_domain}_test"

    @property
    def source_model_dir(self) -> Path:
        return self.experiment_dir / "source_model"

    @property
    def adaptation_dir(self) -> Path:
        return self.experiment_dir / "self-train"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--source-domain", default=SOURCE_DOMAIN)
    parser.add_argument("--target-domain", default=TARGET_DOMAIN)
    parser.add_argument("--model-checkpoint", default=MODEL_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser


def build_config(args: argparse.Namespace) -> QaConfig:
    return QaConfig(
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        model_checkpoint=args.model_checkpoint,
        batch_size=args.batch_size,
        seed=args.seed,
        num_loops=getattr(args, "num_loops", NUM_LOOPS),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_tokenizer(model_checkpoint: str):
    tokenizer = _TOKENIZER_CACHE.get(model_checkpoint)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        _TOKENIZER_CACHE[model_checkpoint] = tokenizer
    return tokenizer


def get_metric():
    global _METRIC
    if _METRIC is None:
        _METRIC = load_metric("squad")
    return _METRIC


def resolve_source_model_dir(config: QaConfig, explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return path

    for candidate in SOURCE_CHECKPOINT_CANDIDATES:
        path = config.experiment_dir / candidate
        if path.exists():
            return path

    checked = ", ".join(str(config.experiment_dir / name) for name in SOURCE_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(f"Could not find a source checkpoint. Checked: {checked}")


def preprocess_training_examples(
    examples,
    tokenizer=None,
    max_length: int = MAXW_LENGTH,
    stride: int = STRIDE,
):
    tokenizer = tokenizer or get_tokenizer(MODEL_CHECKPOINT)
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(
    examples,
    tokenizer=None,
    max_length: int = MAXW_LENGTH,
    stride: int = STRIDE,
):
    tokenizer = tokenizer or get_tokenizer(MODEL_CHECKPOINT)
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            value if sequence_ids[k] == 1 else None for k, value in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(
    start_logits,
    end_logits,
    features,
    examples,
    n_best: int = N_BEST,
    max_answer_length: int = MAX_ANSWER_LENGTH,
):
    metric = get_metric()
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        offsets[start_index] is None
                        or offsets[end_index] is None
                        or len(offsets[start_index]) == 0
                        or len(offsets[end_index]) == 0
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answers.append(
                        {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

        if answers:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
