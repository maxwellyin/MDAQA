from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator

from mdaqa.modeling_roberta import RobertaBMW
from mdaqa.util import (
    add_common_args,
    build_config,
    compute_metrics,
    get_tokenizer,
    preprocess_validation_examples,
    resolve_source_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MDAQA checkpoint on the target test set.")
    add_common_args(parser)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Checkpoint to evaluate. Defaults to checkpoint/<source>-<model>/source_model, with legacy fallbacks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    tokenizer = get_tokenizer(config.model_checkpoint)

    raw_dataset = load_from_disk(str(config.target_test_data)).shuffle(seed=config.seed)
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

    model_dir = resolve_source_model_dir(config, args.model_dir)
    model = RobertaBMW.from_pretrained(str(model_dir))

    eval_dataloader = DataLoader(
        validation_set,
        collate_fn=default_data_collator,
        batch_size=config.batch_size,
    )
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

    metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_dataset)
    accelerator.print(metrics)


if __name__ == "__main__":
    main()
