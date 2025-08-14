#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ------------------------ Tokenizer for BM25 ------------------------

TOK = re.compile(r"[A-Za-z0-9_./-]+")
def tok(s: str) -> List[str]:
    return TOK.findall((s or "").lower())


# ------------------------ JSONL Loading ------------------------

def load_blocks(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# ------------------------ Text Preparation ------------------------

def build_dense_text(
    b: Dict[str, Any],
    use_section_in_dense: bool = True,
    use_intro_in_dense: bool = True,
    use_page_excerpt_in_dense: bool = False,
    max_chars: int = 1200,
) -> str:
    """
    Keep dense texts focused (good semantic signal).
    Default: section label + intro + chunk text.
    Optionally include a short page excerpt (off by default).
    """
    parts: List[str] = []
    if use_section_in_dense:
        sec = (b.get("section_label") or "").strip()
        if sec:
            parts.append(sec)
    if use_intro_in_dense:
        intro = (b.get("section_intro") or "").strip()
        if intro:
            parts.append(intro)

    # prefer textual block; fallback to table_markdown
    text = (b.get("text") or b.get("table_markdown") or "").strip()
    parts.append(text)

    if use_page_excerpt_in_dense:
        pe = (b.get("page_md_excerpt") or "").strip()
        if pe:
            parts.append(pe)

    s = "\n".join(p for p in parts if p)
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def build_sparse_text(b: Dict[str, Any], include_context: bool = True) -> str:
    """
    Rich view for BM25 so acronyms and identifiers match:
    [section_path + section_label + intro] + text + optional (context_before/after) + table_id
    """
    bits: List[str] = []
    if b.get("section_path"):
        bits.append(str(b["section_path"]))
    if b.get("section_label"):
        bits.append(str(b["section_label"]))
    if b.get("section_intro"):
        bits.append(str(b["section_intro"]))
    # main body (text or table)
    body = (b.get("text") or b.get("table_markdown") or "").strip()
    bits.append(body)
    if include_context:
        cb = (b.get("context_before") or "").strip()
        ca = (b.get("context_after") or "").strip()
        if cb: bits.append(cb)
        if ca: bits.append(ca)
    # table id or csv path gives useful keywords sometimes
    if b.get("table_id"): bits.append(str(b["table_id"]))
    if b.get("table_csv_path"): bits.append(str(b["table_csv_path"]))
    return "\n".join(x for x in bits if x)


# ------------------------ Embedding (HF AutoModel) ------------------------

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.inference_mode()
def embed_texts_transformers(
    texts: List[str],
    model_dir: str,
    device: str,
    batch_size: int = 64,
    max_length: int = 512,
    prefix: str = "passage: ",
    use_fp16: bool = True,
    local_only: bool = True,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_only)
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else None
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=local_only, torch_dtype=dtype).to(device).eval()

    out_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        if prefix:
            batch = [f"{prefix}{t}" if t else "" for t in batch]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        rep = mdl(**enc).last_hidden_state
        pooled = mean_pool(rep, enc["attention_mask"])
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out_vecs.append(normed.detach().cpu().to(torch.float32).numpy())

    return np.vstack(out_vecs) if out_vecs else np.zeros((0, mdl.config.hidden_size), dtype=np.float32)


# ------------------------ FAISS ------------------------

def build_faiss_hnsw(vecs: np.ndarray, m: int = 64, ef_construction: int = 200, ef_search: int = 256) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vecs.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vecs.astype(np.float32))
    return index


# ------------------------ Main ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build hybrid (FAISS HNSW + BM25) index from ingest.py JSONL.")
    ap.add_argument("--jsonl", required=True, help="Path to corpus.blocks.jsonl produced by ingest.py")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--model_path", required=True, help="Local dir for HF encoder (e.g., /models/bge-base-en-v1.5)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--no_fp16", action="store_true")

    # Dense text composition knobs
    ap.add_argument("--dense_use_section", action="store_true", help="Include section_label in dense text")
    ap.add_argument("--dense_use_intro", action="store_true", help="Include section_intro in dense text")
    ap.add_argument("--dense_use_page_excerpt", action="store_true", help="Include page_md_excerpt in dense text")
    ap.add_argument("--dense_max_chars", type=int, default=1200)
    ap.add_argument("--corpus_prefix", default="passage: ", help='Prefix for corpus texts (e.g., "passage: " for BGE). Use "" to disable.')

    # Sparse text knobs
    ap.add_argument("--sparse_include_context", action="store_true", help="Include context_before/after in BM25 text")

    # FAISS params
    ap.add_argument("--faiss_m", type=int, default=64)
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--ef_search", type=int, default=256)

    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # Load rows
    blocks = load_blocks(Path(args.jsonl))

    # Prepare texts
    dense_texts: List[str] = []
    sparse_texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for b in tqdm(blocks, desc="Preparing texts"):
        dense = build_dense_text(
            b,
            use_section_in_dense=args.dense_use_section,
            use_intro_in_dense=args.dense_use_intro,
            use_page_excerpt_in_dense=args.dense_use_page_excerpt,
            max_chars=args.dense_max_chars,
        )
        # If no dense text (shouldn't happen), fallback to text field
        if not dense:
            dense = (b.get("text") or b.get("table_markdown") or "")

        sparse = build_sparse_text(b, include_context=args.sparse_include_context)

        dense_texts.append(dense)
        sparse_texts.append(sparse)

        # Carry useful metadata forward for serving time
        meta.append(
            {
                "doc_id": b.get("doc_id"),
                "page": b.get("page"),
                "block_id": b.get("block_id"),
                "block_type": b.get("block_type"),
                "section_path": b.get("section_path", ""),
                "section_label": b.get("section_label", ""),
                "section_intro": b.get("section_intro", ""),
                "table_id": b.get("table_id"),
                "table_csv_path": b.get("table_csv_path"),
                "source_path": b.get("source_path"),
                "page_md_path": b.get("page_md_path"),
                "page_md_hash": b.get("page_md_hash"),
            }
        )

    # Embed
    vecs = embed_texts_transformers(
        dense_texts,
        model_dir=args.model_path,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prefix=args.corpus_prefix,
        use_fp16=not args.no_fp16,
        local_only=True,
    )

    # Build FAISS HNSW
    index = build_faiss_hnsw(
        vecs,
        m=args.faiss_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    # Build BM25
    bm25 = BM25Okapi([tok(t) for t in tqdm(sparse_texts, desc="BM25 tokenization")])

    # Persist
    faiss.write_index(index, str(out / "dense_hnsw.faiss"))
    np.save(out / "dense.npy", vecs)
    with (out / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    with (out / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm": bm25, "texts": sparse_texts}, f)

    # Manifest
    manifest = {
        "encoder_dir": args.model_path,
        "device": device,
        "count": len(blocks),
        "dim": int(vecs.shape[1]) if vecs.size else 0,
        "faiss": {"type": "HNSWFlat", "M": args.faiss_m, "efSearch": args.ef_search},
        "corpus_prefix": args.corpus_prefix,
        "dense_policy": {
            "use_section": bool(args.dense_use_section),
            "use_intro": bool(args.dense_use_intro),
            "use_page_excerpt": bool(args.dense_use_page_excerpt),
            "max_chars": int(args.dense_max_chars),
        },
        "sparse_policy": {"include_context": bool(args.sparse_include_context)},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
