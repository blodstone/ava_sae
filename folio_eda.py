"""
EDA and data preparation script for FOLIO dataset validation set.
Loads JSONL data, formats premises and conclusions into prompts with labels,
and saves the processed data to a new JSONL file for tokenization.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def build_prompt_from_row(row: dict) -> str:
    # try common keys for premises
    prompt = (
        "Premises " + row['premises-FOL'] + " ")
    prompt += f"Conclusion -> {row['conclusion-FOL']} -> Label: {row['label']}"
    return prompt


def load_jsonl(input_path: Path) -> list[dict]:
    with input_path.open() as f:
        return [json.loads(line) for line in f]


def prepare_dataframe(data: list[dict]) -> pd.DataFrame:
    data_pd = pd.DataFrame(data)
    if not data_pd.empty:
        data_pd["text"] = data_pd.apply(lambda r: build_prompt_from_row(r.to_dict()), axis=1)
    return data_pd


def save_text_jsonl(data_pd: pd.DataFrame, output_path: Path) -> None:
    with output_path.open('w') as f:
        for _, row in data_pd.iterrows():
            json.dump({"text": row["text"]}, f)
            f.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FOLIO JSONL prompts for tokenization.")
    parser.add_argument("input_file", type=Path, help="Path to the input FOLIO JSONL file")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = input_path.with_name(f"{input_path.stem}_tokenized{input_path.suffix}")

    data = load_jsonl(input_path)
    data_pd = prepare_dataframe(data)
    save_text_jsonl(data_pd, output_path)


if __name__ == "__main__":
    main()
