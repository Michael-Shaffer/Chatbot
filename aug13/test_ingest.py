#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def sample_chunks(chunks: List[Dict[str, Any]], n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    return random.sample(chunks, min(n, len(chunks)))


def pretty_print_chunk(c: Dict[str, Any]) -> None:
    hdr_meta = f"[{c.get('doc_id','?')}:{c.get('page','?')}] ({c.get('block_type','?')})"
    hdr_id = f"block_id: {c.get('block_id','')}"
    hdr_src = f"source: {c.get('source_path','')}  ts: {c.get('ts_extracted','')}"
    section_path = c.get("section_path", "") or ""
    section_label = c.get("section_label", "") or ""
    section_intro = c.get("section_intro", "") or ""

    print("=" * 100)
    print(hdr_meta)
    print(hdr_id)
    if section_path or section_label:
        print(f"section_path: {section_path}")
        if section_label:
            print(f"section_label: {section_label}")
    if section_intro:
        print("-- Section Intro --")
        print(section_intro)

    bt = c.get("block_type", "")
    if bt == "table":
        print("-- Table Markdown --")
        print(c.get("table_markdown", "") or c.get("markdown", ""))
        csvp = c.get("table_csv_path", "")
        if csvp:
            print(f"(CSV path: {csvp})")
    else:
        print("-- Text --")
        print(c.get("text", "") or c.get("markdown", ""))

    ctx_b = c.get("context_before", "")
    ctx_a = c.get("context_after", "")
    if ctx_b or ctx_a:
        print("-- Context --")
        if ctx_b:
            print(f"before: {ctx_b}")
        if ctx_a:
            print(f"after : {ctx_a}")

    span_start = c.get("span_start")
    span_end = c.get("span_end")
    if span_start is not None and span_end is not None:
        print(f"spans: [{span_start}, {span_end})")

    print(hdr_src)
    print("=" * 100)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample ingested blocks (with full context) for manual QA.")
    parser.add_argument("--jsonl", required=True, help="Path to ingested JSONL file.")
    parser.add_argument("--n", type=int, default=5, help="Number of blocks to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--block_type", choices=["paragraph", "list", "code", "table", "heading"], default=None,
                        help="Filter sampled blocks by type.")
    args = parser.parse_args()

    chunks = load_jsonl(Path(args.jsonl))
    if args.block_type:
        chunks = [c for c in chunks if c.get("block_type") == args.block_type]
        if not chunks:
            print(f"No blocks of type '{args.block_type}' found.")
            return

    samples = sample_chunks(chunks, args.n, seed=args.seed)

    for c in samples:
        pretty_print_chunk(c)


if __name__ == "__main__":
    main()
