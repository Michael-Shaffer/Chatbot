# test_ingest.py
# Inspect prebuilt JSONL chunks and run local retrieval smoke tests.
# Usage:
#   python test_ingest.py --jsonl index/chunks.jsonl --show 6
#   python test_ingest.py --jsonl index/chunks.jsonl --q "track smoothing parameters" --k 5
#   python test_ingest.py --jsonl index/chunks.jsonl --stats

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter
from retriever import Retriever

# --------------------------
# IO
# --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
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

def _short(s: str, n: int = 180) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n].rstrip() + "…"

def show_samples(chunks: List[Dict[str, Any]], n: int = 6, block_type: str = "") -> None:
    total = len(chunks)
    if block_type:
        chunks = [c for c in chunks if c.get("block_type") == block_type]
    take = min(n, len(chunks))
    print(f"\n=== Sample Chunks ({take}/{total}{' filtered' if block_type else ''}) ===")
    for ch in chunks[:take]:
        cid = ch.get("chunk_id", "")
        page = ch.get("page", 0)
        btype = ch.get("block_type", "")
        spath = ch.get("section_path", "")
        slabel = ch.get("section_label", "")
        intro = ch.get("section_intro", "")
        summary = ch.get("table_summary", "")
        text = ch.get("text", "")
        print(f"- id={cid}  page={page}  type={btype}")
        print(f"  section: {slabel or spath}")
        if intro:
            print(f"  intro  : {_short(intro)}")
        if btype == "table" and summary:
            print(f"  table  : {_short(summary)}")
        print(f"  text   : {_short(text)}\n")

def show_stats(chunks: List[Dict[str, Any]]) -> None:
    kinds = Counter(c.get("block_type", "") for c in chunks)
    with_intro = sum(1 for c in chunks if (c.get("section_intro") or "").strip())
    tables = [c for c in chunks if c.get("block_type") == "table"]
    tables_with_summary = sum(1 for t in tables if (t.get("table_summary") or "").strip())
    avg_len = _avg_len(c.get("text", "") for c in chunks if c.get("text"))
    print("\n=== Corpus Stats ===")
    print(f"chunks total            : {len(chunks)}")
    print(f"by block_type           : {dict(kinds)}")
    print(f"chunks with section_intro: {with_intro}")
    print(f"tables total            : {len(tables)}")
    print(f"tables with summary     : {tables_with_summary}")
    print(f"avg visible text length : {avg_len:.1f} chars")

def _avg_len(texts: Iterable[str]) -> float:
    total, count = 0, 0
    for t in texts:
        total += len(t)
        count += 1
    return (total / count) if count else 0.0

# --------------------------
# Retrieval
# --------------------------

def run_search(chunks: List[Dict[str, Any]], query: str, k: int = 5) -> None:
    r = Retriever()
    r.fit(chunks)
    hits = r.search(query, top_k=k)
    print(f"\n=== Query: {query} ===")
    for h in hits:
        page = h.get("page", 0)
        btype = h.get("block_type", "")
        sec = h.get("section_label", "") or h.get("section_path", "")
        txt = h.get("text", "") or h.get("table_summary", "")
        print(f"* [{h['score']:.3f}] p{page} {btype} :: {sec}")
        print(f"  {_short(txt, 220)}\n")

# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect JSONL chunks and test retrieval.")
    ap.add_argument("--jsonl", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--show", type=int, default=0, help="Print N sample chunks")
    ap.add_argument("--type", default="", help="Filter samples by block_type (paragraph|table|list|heading)")
    ap.add_argument("--stats", action="store_true", help="Print corpus stats")
    ap.add_argument("--q", default="", help="Run a retrieval query")
    ap.add_argument("--k", type=int, default=5, help="Top-K results for retrieval")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.jsonl)
    if args.stats:
        show_stats(chunks)
    if args.show > 0:
        show_samples(chunks, n=args.show, block_type=args.type)
    if args.q:
        run_search(chunks, query=args.q, k=args.k)
    if not (args.stats or args.show or args.q):
        print("Nothing to do. Use --stats and/or --show N and/or --q 'query'.")

if __name__ == "__main__":
    main()
