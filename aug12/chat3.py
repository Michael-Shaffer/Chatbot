#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def sample_chunks(chunks: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return random.sample(chunks, min(n, len(chunks)))


def pretty_print_chunk(chunk: Dict[str, Any]) -> None:
    meta = f"[{chunk['doc_id']}:{chunk['page']}] ({chunk['block_type']})"
    section = f"Section: {chunk['section_path']}" if chunk.get("section_path") else ""
    print("=" * 80)
    print(meta)
    if section:
        print(section)
    if chunk['block_type'] == "table":
        print("-- Table Markdown --")
        print(chunk['table_markdown'])
        print(f"(CSV path: {chunk['table_csv_path']})")
    else:
        print("-- Text --")
        print(chunk['text'])
    print("=" * 80)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample ingested chunks for manual QA.")
    parser.add_argument("--jsonl", required=True, help="Path to ingested JSONL file.")
    parser.add_argument("--n", type=int, default=5, help="Number of chunks to sample.")
    args = parser.parse_args()

    chunks = load_jsonl(Path(args.jsonl))
    samples = sample_chunks(chunks, args.n)

    for c in samples:
        pretty_print_chunk(c)


if __name__ == "__main__":
    main()