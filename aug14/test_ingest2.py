#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def sample_items(items: List[Dict[str, Any]], n: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    return random.sample(items, min(n, len(items)))


def read_page_md(page_md_path: Optional[str], max_chars: int) -> Tuple[str, str]:
    """Returns (mode, preview_text) where mode ∈ {'file','missing'}."""
    if not page_md_path:
        return ("missing", "")
    p = Path(page_md_path)
    if not p.exists():
        return ("missing", f"(missing file) {page_md_path}")
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception as e:
        return ("missing", f"(error reading {page_md_path}: {e})")
    txt = txt.strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars] + " …"
    return ("file", txt)


def pretty_print_block(
    b: Dict[str, Any],
    page_md_mode: str,
    page_md_chars: int,
) -> None:
    # Header
    meta = f"[{b.get('doc_id','?')}:{b.get('page','?')}]  type={b.get('block_type','?')}"
    print("=" * 110)
    print(meta)
    print(f"block_id: {b.get('block_id','')}")

    # Section info
    sec_path = (b.get("section_path") or "").strip()
    sec_label = (b.get("section_label") or "").strip()
    if sec_path or sec_label:
        print(f"section_path: {sec_path}")
        if sec_label:
            print(f"section_label: {sec_label}")
    sec_intro = (b.get("section_intro") or "").strip()
    if sec_intro:
        print("-- Section Intro --")
        print(sec_intro)

    # Chunk text
    bt = b.get("block_type")
    if bt == "table":
        print("-- Table Markdown --")
        print((b.get("table_markdown") or b.get("markdown") or "").strip())
        csvp = b.get("table_csv_path")
        if csvp:
            print(f"(CSV: {csvp})")
    else:
        print("-- Chunk Text --")
        print((b.get("text") or b.get("markdown") or "").strip())

    # Context
    cb = (b.get("context_before") or "").strip()
    ca = (b.get("context_after") or "").strip()
    if cb or ca:
        print("-- Adjacent Context --")
        if cb:
            print(f"before: {cb}")
        if ca:
            print(f"after : {ca}")

    # Page-level Markdown
    ppath = b.get("page_md_path")
    pexcerpt = (b.get("page_md_excerpt") or "").strip()
    print("-- Page Markdown --")
    if page_md_mode == "off":
        print("(disabled)")
    elif page_md_mode == "excerpt":
        if pexcerpt:
            print(f"[excerpt] {pexcerpt}")
        else:
            # fallback to reading file if excerpt missing
            mode, preview = read_page_md(ppath, page_md_chars)
            print(f"[{mode}] {preview}" if preview else "(no page MD available)")
    elif page_md_mode == "file":
        mode, preview = read_page_md(ppath, page_md_chars)
        if preview:
            print(f"[{mode}] {preview}")
        else:
            print("(no page MD available)")
    else:
        print("(unknown mode)")

    # Source & spans
    span_start = b.get("span_start")
    span_end = b.get("span_end")
    if span_start is not None and span_end is not None:
        print(f"spans: [{span_start}, {span_end})")
    print(f"source: {b.get('source_path','')}    ts: {b.get('ts_extracted','')}")
    if ppath:
        print(f"page_md_path: {ppath}")
    print("=" * 110)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample ingested blocks and show page-level Markdown.")
    ap.add_argument("--jsonl", required=True, help="Path to ingested JSONL.")
    ap.add_argument("--n", type=int, default=5, help="Number of blocks to sample.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    ap.add_argument(
        "--block_type",
        choices=["paragraph", "list", "code", "table", "heading"],
        default=None,
        help="Filter by block type.",
    )
    ap.add_argument(
        "--page_md_mode",
        choices=["excerpt", "file", "off"],
        default="excerpt",
        help="Show page-level Markdown: stored excerpt (default), read from file, or off.",
    )
    ap.add_argument(
        "--page_md_chars",
        type=int,
        default=600,
        help="Max characters to display from page Markdown (for excerpt or file).",
    )
    args = ap.parse_args()

    chunks = load_jsonl(Path(args.jsonl))
    if args.block_type:
        chunks = [c for c in chunks if c.get("block_type") == args.block_type]
        if not chunks:
            print(f"No blocks of type '{args.block_type}' found.")
            return

    samples = sample_items(chunks, args.n, args.seed)
    for b in samples:
        pretty_print_block(b, page_md_mode=args.page_md_mode, page_md_chars=args.page_md_chars)


if __name__ == "__main__":
    main()
