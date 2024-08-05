from __future__ import annotations

import argparse
from pathlib import Path

from mdaqa.data import build_preprocess_paths, convert_mrqa_to_hf_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MRQA-style QA JSON into a Hugging Face dataset.")
    parser.add_argument("--domain", required=True, help='Example: "BioASQ" or "SQuAD".')
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path, output_path = build_preprocess_paths(
        domain=args.domain,
        split=args.split,
        input_path=args.input_path,
        output_path=args.output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = convert_mrqa_to_hf_dataset(input_path)
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
