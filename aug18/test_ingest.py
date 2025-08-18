# test_ingest.py
# Inspect JSONL chunks produced by ingest.py
# Usage examples:
#   python test_ingest.py --jsonl your.jsonl --show 6
#   python test_ingest.py --jsonl your.jsonl --stats
#   python test_ingest.py --jsonl your.jsonl --grep radar

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Iterable

# --------------------------
# IO
# --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items

# --------------------------
# Pretty printing
# --------------------------

def _short(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[:n].rstrip() + "…"

def show_samples(chunks: List[Dict[str, Any]], n: int = 6, block_type: str = "") -> None:
    total = len(chunks)
    if block_type:
        chunks = [c for c in chunks if c.get("block_type") == block_type]
    take = min(n, len(chunks))
    print(f"\n=== Sample Chunks ({take}/{total}{' filtered' if block_type else ''}) ===")
    for ch in chunks[:take]:
        print(f"- id={ch.get('chunk_id','')}  page={ch.get('page',0)}  type={ch.get('block_type','')}")
        print(f"  section: {ch.get('section_label') or ch.get('section_path')}")
        if ch.get("section_intro"):
            print(f"  intro  : {_short(ch['section_intro'])}")
        if ch.get("block_type") == "table" and ch.get("table_summary"):
            print(f"  table  : {_short(ch['table_summary'])}")
        if ch.get("text"):
            print(f"  text   : {_short(ch['text'])}")
        print()

def show_stats(chunks: List[Dict[str, Any]]) -> None:
    kinds = Counter(c.get("block_type","") for c in chunks)
    with_intro = sum(1 for c in chunks if (c.get("section_intro") or "").strip())
    tables = [c for c in chunks if c.get("block_type") == "table"]
    tables_with_summary = sum(1 for t in tables if (t.get("table_summary") or "").strip())
    avg_len = _avg_len(c.get("text","") for c in chunks if c.get("text"))
    print("\n=== Corpus Stats ===")
    print(f"chunks total             : {len(chunks)}")
    print(f"by block_type            : {dict(kinds)}")
    print(f"chunks with section_intro: {with_intro}")
    print(f"tables total             : {len(tables)}")
    print(f"tables with summary      : {tables_with_summary}")
    print(f"avg visible text length  : {avg_len:.1f} chars")

def _avg_len(texts: Iterable[str]) -> float:
    total, count = 0, 0
    for t in texts:
        total += len(t)
        count += 1
    return (total / count) if count else 0.0

def grep_chunks(chunks: List[Dict[str, Any]], pattern: str, n: int = 10) -> None:
    import re
    regex = re.compile(pattern, re.IGNORECASE)
    matches = [c for c in chunks if regex.search(c.get("text",""))]
    print(f"\n=== Grep '{pattern}' ({len(matches)} matches) ===")
    for ch in matches[:n]:
        print(f"- page={ch.get('page')} type={ch.get('block_type')} sec={ch.get('section_label')}")
        print(f"  {_short(ch.get('text',''))}\n")

# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Inspect JSONL chunks from ingest.py")
    ap.add_argument("--jsonl", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--show", type=int, default=0, help="Show N sample chunks")
    ap.add_argument("--type", default="", help="Filter samples by block_type (paragraph|table|list|heading)")
    ap.add_argument("--stats", action="store_true", help="Show corpus statistics")
    ap.add_argument("--grep", default="", help="Regex search in chunk text")
    return ap.parse_args()

def main():
    args = parse_args()
    chunks = read_jsonl(args.jsonl)
    if args.stats:
        show_stats(chunks)
    if args.show > 0:
        show_samples(chunks, n=args.show, block_type=args.type)
    if args.grep:
        grep_chunks(chunks, args.grep)
    if not (args.stats or args.show or args.grep):
        print("Nothing to do. Use --stats, --show N, or --grep pattern.")

if __name__ == "__main__":
    main()
