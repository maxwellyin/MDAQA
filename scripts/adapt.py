from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator, get_scheduler, pipeline

from mdaqa.modeling_roberta import RobertaBMW
from mdaqa.util import (
    BOTTLE_DIM,
    add_common_args,
    build_config,
    compute_metrics,
    ensure_dir,
    get_tokenizer,
    preprocess_training_examples,
    preprocess_validation_examples,
    resolve_source_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-free domain adaptation for MDAQA.")
    add_common_args(parser)
    parser.add_argument("--num-loops", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--lr-base", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=5e-3)
    parser.add_argument("--mask-distance", type=float, default=3.0)
    parser.add_argument("--train-sample-limit", type=int, default=2000)
    parser.add_argument("--train-split-ratio", type=float, default=0.5)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=None,
        help="Path to a pretrained source checkpoint. Defaults to checkpoint/<source>-<model>/source_model, with legacy fallbacks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for adapted checkpoints. Defaults to checkpoint/<source>-<model>/self-train.",
    )
    return parser.parse_args()


def prepare_validation_set(raw_dataset, tokenizer, config):
    validation_dataset = raw_dataset.map(
        partial(
            preprocess_validation_examples,
            tokenizer=tokenizer,
            max_length=config.max_length,
            stride=config.stride,
        ),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    return validation_dataset, validation_set


def build_question_answerer(model, tokenizer):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)


def apply_mask_constraints(model, masks_old, hidden_size: int, mask_distance: float):
    target_mask = model.getMask()

    for name, param in model.bottleMask.bottleneck.named_parameters():
        if param.grad is None:
            continue
        if "weight" in name:
            grad_mask = (1 - masks_old).view(-1, 1).expand(BOTTLE_DIM, hidden_size)
        else:
            grad_mask = (1 - masks_old).squeeze()
        param.grad.data *= grad_mask.to(param.grad.device)

    for name, param in model.qa_outputs.named_parameters():
        if param.grad is None or "weight_v" not in name:
            continue
        grad_mask = (1 - masks_old.view(1, -1).expand(2, BOTTLE_DIM)).to(param.grad.device)
        param.grad.data *= grad_mask

    if torch.abs(target_mask - masks_old).sum() > mask_distance and model.bottleMask.m.grad is not None:
        model.bottleMask.m.grad.data.zero_()


def train_one_round(train_dataset, model, masks_old, args, config):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.batch_size,
    )

    params_base = [
        value
        for name, value in model.named_parameters()
        if "bottleMask" not in name and "qa_outputs" not in name
    ]
    params_head = [
        value for name, value in model.named_parameters() if "bottleMask" in name or "qa_outputs" in name
    ]
    optimizer_base = AdamW(params_base, lr=args.lr_base)
    optimizer_head = AdamW(params_head, lr=args.lr_head)

    accelerator = Accelerator()
    model, optimizer_base, optimizer_head, train_dataloader = accelerator.prepare(
        model, optimizer_base, optimizer_head, train_dataloader
    )

    total_steps = args.epochs * len(train_dataloader)
    scheduler_base = get_scheduler(
        "linear",
        optimizer=optimizer_base,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    scheduler_head = get_scheduler(
        "linear",
        optimizer=optimizer_head,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    hidden_size = model.module.roberta.config.hidden_size if hasattr(model, "module") else model.roberta.config.hidden_size
    progress_bar = tqdm(range(total_steps), desc="adapt-train")
    model.train()
    masks_old = masks_old.to(accelerator.device)

    for _ in range(args.epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            accelerator.backward(outputs.loss)
            apply_mask_constraints(model, masks_old, hidden_size, args.mask_distance)

            optimizer_base.step()
            scheduler_base.step()
            optimizer_base.zero_grad()

            optimizer_head.step()
            scheduler_head.step()
            optimizer_head.zero_grad()
            progress_bar.update(1)

    accelerator.wait_for_everyone()
    return accelerator.unwrap_model(model)


def evaluate(model, validation_dataset, validation_set, raw_examples, batch_size):
    eval_dataloader = DataLoader(validation_set, collate_fn=default_data_collator, batch_size=batch_size)
    accelerator = Accelerator()
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    model.eval()
    start_logits = []
    end_logits = []
    for batch in tqdm(eval_dataloader, desc="eval"):
        with torch.no_grad():
            outputs = model(**batch)
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)[: len(validation_dataset)]
    end_logits = np.concatenate(end_logits)[: len(validation_dataset)]
    return compute_metrics(start_logits, end_logits, validation_dataset, raw_examples)


def create_pseudo_labeled_train_set(raw_train_split, question_answerer, tokenizer, config, threshold):
    def create_pseudo_label(example):
        out = question_answerer(question=example["question"], context=example["context"])
        answers = {"text": [out["answer"]], "answer_start": [out["start"]]}
        return {"answers": answers, "score": out["score"]}

    pseudo_set = raw_train_split.map(create_pseudo_label, load_from_cache_file=False)
    filtered = pseudo_set.filter(lambda example: example["score"] > threshold)
    train_dataset = filtered.map(
        partial(
            preprocess_training_examples,
            tokenizer=tokenizer,
            max_length=config.max_length,
            stride=config.stride,
        ),
        batched=True,
        remove_columns=filtered.column_names,
    )
    train_dataset.set_format("torch")
    return train_dataset, len(filtered)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    tokenizer = get_tokenizer(config.model_checkpoint)

    source_model_dir = resolve_source_model_dir(config, args.source_model_dir)
    output_dir = ensure_dir(args.output_dir or config.adaptation_dir)

    model = RobertaBMW.from_pretrained(str(source_model_dir))
    source_mask = model.getMask().detach()

    train_dataset = load_from_disk(str(config.target_train_data)).shuffle(seed=config.seed)
    if args.train_sample_limit is not None:
        train_dataset = train_dataset.select(range(min(args.train_sample_limit, len(train_dataset))))
    train_splits = train_dataset.train_test_split(train_size=args.train_split_ratio, seed=config.seed)

    test_dataset = load_from_disk(str(config.target_test_data)).shuffle(seed=config.seed)
    raw_datasets = DatasetDict(
        {
            "train": train_splits["train"],
            "validation": train_splits["test"],
            "test": test_dataset,
        }
    )
    validation_dataset, validation_set = prepare_validation_set(raw_datasets["test"], tokenizer, config)

    outcomes = []
    for loop_idx in range(args.num_loops):
        question_answerer = build_question_answerer(model, tokenizer)
        pseudo_train_dataset, pseudo_count = create_pseudo_labeled_train_set(
            raw_datasets["train"], question_answerer, tokenizer, config, args.threshold
        )
        model = train_one_round(pseudo_train_dataset, model, source_mask, args, config)
        outcome = evaluate(
            model=model,
            validation_dataset=validation_dataset,
            validation_set=validation_set,
            raw_examples=raw_datasets["test"],
            batch_size=config.batch_size,
        )
        target_mask = model.getMask().detach()
        outcome["mask_diff"] = torch.abs(target_mask - source_mask).sum().item()
        outcome["pseudo_examples"] = pseudo_count
        outcomes.append(outcome)

        checkpoint_path = ensure_dir(output_dir / str(loop_idx))
        model.save_pretrained(str(checkpoint_path))
        print(f"loop={loop_idx} metrics={outcome} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
