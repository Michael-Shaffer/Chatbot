# build_index.py
# Build a local RAG index from PDFs: chunks.jsonl (+ optional retriever.pkl)
# Works with ingest.py and retriever.py from this project.
# deps: pdfplumber, rank-bm25, scikit-learn, joblib

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Any, Union
from dataclasses import asdict, is_dataclass
from joblib import dump, load

# Project-local imports
from ingest import make_chunks
from retriever import Retriever

# --------------------------
# Filesystem helpers
# --------------------------

def find_pdfs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == ".pdf":
            out.append(path)
            continue
        if path.is_dir():
            for f in path.rglob("*.pdf"):
                out.append(f)
    return sorted(set(out))

def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path:
        return items
    p = Path(path)
    if not p.exists():
        return items
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

def write_jsonl(items: Iterable[Union[Dict[str, Any], Any]], out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in items:
            rec = asdict(obj) if is_dataclass(obj) else obj
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def dedupe_by_key(items: List[Dict[str, Any]], key: str = "chunk_id") -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for it in items:
        cid = str(it.get(key, ""))
        if not cid:
            continue
        seen[cid] = it
    return list(seen.values())

# --------------------------
# Build / Merge / Store
# --------------------------

def build_index(
    inputs: List[str],
    outdir: Union[str, Path] = "index",
    merge_jsonl: Union[str, Path] = "",
    store_retriever: bool = True,
) -> Dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(inputs)
    if not pdfs:
        print("No PDFs found in provided paths.", file=sys.stderr)
        return {}

    all_chunks: List[Dict[str, Any]] = []
    for pdf in pdfs:
        print(f"[ingest] {pdf}")
        chunks = make_chunks(str(pdf))
        all_chunks.extend([c.__dict__ for c in chunks])

    prior = read_jsonl(merge_jsonl) if merge_jsonl else []
    if prior:
        print(f"[merge] prior chunks: {len(prior)}")

    merged = dedupe_by_key(prior + all_chunks, key="chunk_id")
    merged_path = outdir / "chunks.jsonl"
    write_jsonl(merged, merged_path)
    print(f"[write] {len(merged)} chunks -> {merged_path}")

    retr_path = outdir / "retriever.pkl"
    if store_retriever:
        r = Retriever()
        r.fit(merged)
        dump(r, retr_path)
        print(f"[store] retriever -> {retr_path}")

    return {"chunks": merged_path, "retriever": retr_path if store_retriever else Path("")}

# --------------------------
# Search (smoke test)
# --------------------------

def load_retriever(index_dir: Union[str, Path]) -> Retriever:
    p = Path(index_dir) / "retriever.pkl"
    if p.exists():
        return load(p)
    # Fallback: build ephemeral retriever from chunks.jsonl
    cj = Path(index_dir) / "chunks.jsonl"
    items = read_jsonl(cj)
    if not items:
        raise FileNotFoundError("No retriever.pkl or chunks.jsonl found.")
    r = Retriever()
    r.fit(items)
    return r

def search_cli(index_dir: Union[str, Path], query: str, k: int = 5) -> None:
    r = load_retriever(index_dir)
    hits = r.search(query, top_k=k)
    print(f"\n=== Query: {query} ===")
    for h in hits:
        p = h.get("page", 0)
        t = h.get("block_type", "")
        sec = h.get("section_label", "") or h.get("section_path", "")
        txt = h.get("text", "") or h.get("table_summary", "")
        print(f"* [{h['score']:.3f}] p{p} {t} :: {sec}")
        print(f"  {txt[:200]}\n")

# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a local RAG index (chunks.jsonl + retriever.pkl).")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="Ingest PDFs and build the index.")
    b.add_argument("inputs", nargs="+", help="PDF files and/or directories")
    b.add_argument("--outdir", default="index", help="Output directory")
    b.add_argument("--merge", default="", help="Existing chunks.jsonl to merge with")
    b.add_argument("--no-store", action="store_true", help="Do not serialize retriever.pkl")

    s = sub.add_parser("search", help="Run a quick search against an index dir.")
    s.add_argument("--index", default="index", help="Index directory (with retriever.pkl or chunks.jsonl)")
    s.add_argument("--q", required=True, help="Query text")
    s.add_argument("--k", type=int, default=5, help="Top K results")

    if len(sys.argv) == 1:
        ap.print_help(sys.stderr)
        sys.exit(1)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    if args.cmd == "build":
        build_index(
            inputs=args.inputs,
            outdir=args.outdir,
            merge_jsonl=args.merge,
            store_retriever=not args.no_store,
        )
        return
    if args.cmd == "search":
        search_cli(index_dir=args.index, query=args.q, k=args.k)
        return

if __name__ == "__main__":
    main()
