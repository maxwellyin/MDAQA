from __future__ import annotations

import argparse
from functools import partial

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator, get_scheduler

from mdaqa.modeling_roberta import RobertaBMW
from mdaqa.util import (
    add_common_args,
    build_config,
    ensure_dir,
    get_tokenizer,
    preprocess_training_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the source-domain MDAQA model.")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-base", type=float, default=2e-5)
    parser.add_argument("--lr-mask", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit checkpoint directory. Defaults to checkpoint/<source>-<model>/source_model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = ensure_dir(config.source_model_dir if args.output_dir is None else Path(args.output_dir))

    raw_dataset = load_from_disk(str(config.source_train_data))
    tokenizer = get_tokenizer(config.model_checkpoint)
    train_dataset = raw_dataset.map(
        partial(
            preprocess_training_examples,
            tokenizer=tokenizer,
            max_length=config.max_length,
            stride=config.stride,
        ),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    train_dataset.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.batch_size,
    )

    model = RobertaBMW.from_pretrained(config.model_checkpoint)
    params_base = [v for k, v in model.named_parameters() if "bottleMask" not in k]
    params_mask = [v for k, v in model.named_parameters() if "bottleMask" in k]

    optimizer_base = AdamW(params_base, lr=args.lr_base)
    optimizer_mask = Adam(params_mask, lr=args.lr_mask)

    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
    model, optimizer_base, optimizer_mask, train_dataloader = accelerator.prepare(
        model, optimizer_base, optimizer_mask, train_dataloader
    )

    total_steps = args.epochs * len(train_dataloader)
    lr_scheduler_base = get_scheduler(
        "linear",
        optimizer=optimizer_base,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    progress_bar = tqdm(range(total_steps), desc="source-train")
    model.train()
    for _ in range(args.epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            accelerator.backward(outputs.loss)

            optimizer_base.step()
            lr_scheduler_base.step()
            optimizer_base.zero_grad()

            optimizer_mask.step()
            optimizer_mask.zero_grad()
            progress_bar.update(1)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(str(output_dir), save_function=accelerator.save)
    accelerator.print(f"Saved source model to {output_dir}")
if __name__ == "__main__":
    main()
