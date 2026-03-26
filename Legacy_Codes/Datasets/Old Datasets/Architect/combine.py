"""
combine.py
----------
Combines all batch_XX.jsonl files in the Training Data folder into a
single architect_dataset.jsonl file.

Usage:
    python combine.py
    python combine.py --input_dir "Training Data" --output "architect_dataset.jsonl"
"""

import argparse
import json
import os
from pathlib import Path


def combine_jsonl_batches(input_dir: str, output_file: str) -> None:
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path.resolve()}")

    # Collect all batch files, sorted numerically (batch_01, batch_02, ...)
    batch_files = sorted(input_path.glob("batch_*.jsonl"))

    if not batch_files:
        print(f"[!] No batch_*.jsonl files found in '{input_dir}'")
        return

    print(f"[+] Found {len(batch_files)} batch files in '{input_dir}'")

    total_records = 0
    skipped_lines = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for batch_file in batch_files:
            file_records = 0
            with open(batch_file, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f, start=1):
                    line = line.strip()
                    if not line:
                        continue  # skip empty lines

                    # Validate each line is valid JSON
                    try:
                        json.loads(line)
                        out_f.write(line + "\n")
                        file_records += 1
                        total_records += 1
                    except json.JSONDecodeError as e:
                        print(f"  [!] Skipping invalid JSON in {batch_file.name} "f"line {line_num}: {e}")
                        skipped_lines += 1

            print(f"  [✓] {batch_file.name}: {file_records} records")

    print()
    print(f"[✓] Combined dataset saved to: '{output_file}'")
    print(f"    Total records : {total_records}")
    if skipped_lines:
        print(f"    Skipped lines : {skipped_lines} (invalid JSON)")


def main():
    parser = argparse.ArgumentParser(
        description="Combine batch JSONL files into a single dataset file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Training Data",
        help="Directory containing batch_XX.jsonl files (default: 'Training Data')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="architect_dataset.jsonl",
        help="Output JSONL file path (default: 'architect_dataset.jsonl')",
    )
    args = parser.parse_args()

    combine_jsonl_batches(args.input_dir, args.output)


if __name__ == "__main__":
    main()