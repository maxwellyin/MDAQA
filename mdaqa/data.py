from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset

DEFAULT_RAW_DATA_DIR = Path("datasets/QVE/data")
DEFAULT_OUTPUT_DIR = Path("data")


def build_preprocess_paths(
    domain: str,
    split: str,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    resolved_input = input_path or (DEFAULT_RAW_DATA_DIR / f"{domain}.{split}.json")
    resolved_output = output_path or (DEFAULT_OUTPUT_DIR / f"{domain}_{split}")
    return resolved_input, resolved_output


def convert_mrqa_to_hf_dataset(input_path: Path) -> Dataset:
    with input_path.open() as handle:
        data = json.load(handle)

    frame = pd.DataFrame(data["data"])
    paragraphs = frame["paragraphs"].apply(lambda items: items[0])
    paragraph_frame = pd.DataFrame(paragraphs.tolist())
    qa_frame = pd.DataFrame(sum(paragraph_frame["qas"], []))

    counts = [len(items) for items in paragraph_frame["qas"]]
    contexts = []
    for idx, count in enumerate(counts):
        contexts.extend([paragraph_frame.loc[idx, "context"]] * count)
    qa_frame["context"] = contexts

    for idx in range(len(qa_frame)):
        qa_frame.at[idx, "answers"] = qa_frame.at[idx, "answers"][0]
        qa_frame.at[idx, "answers"]["answer_start"] = [qa_frame.at[idx, "answers"]["answer_start"]]
        qa_frame.at[idx, "answers"]["text"] = [qa_frame.at[idx, "answers"]["text"]]

    return Dataset.from_pandas(qa_frame)
