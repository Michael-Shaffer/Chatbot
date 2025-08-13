#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi
import json


TOK = re.compile(r"[A-Za-z0-9_./-]+")


def tok(s: str) -> List[str]:
    return TOK.findall((s or "").lower())


def load_blocks(jsonl_path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in jsonl_path.open("r", encoding="utf-8")]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def prepare_corpus_text(b: Dict[str, Any], corpus_prefix: str) -> str:
    # What we embed: prefer text; fallback to table_markdown
    t = (b.get("text") or b.get("table_markdown") or "").strip()
    return f"{corpus_prefix}{t}" if corpus_prefix else t


@torch.inference_mode()
def embed_texts_transformers(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 64,
    use_fp16: bool = True,
    max_length: int = 512,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if (use_fp16 and device.startswith("cuda")) else None)
    mdl.to(device)
    mdl.eval()

    out_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = mdl(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        normed = l2_normalize(pooled)
        out_vecs.append(normed.detach().cpu().to(torch.float32).numpy())
    return np.vstack(out_vecs)


def build_faiss_hnsw(vecs: np.ndarray, m: int = 64, ef_construction: int = 200, ef_search: int = 256) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vecs.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vecs.astype(np.float32))
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hybrid (FAISS HNSW + BM25) index with HF AutoModel embeddings.")
    ap.add_argument("--jsonl", required=True, help="Path to ingested blocks JSONL")
    ap.add_argument("--outdir", required=True, help="Output directory for index files")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="HF embedding model name")
    ap.add_argument("--device", default="auto", help='cuda|cpu|auto (default "auto")')
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--no_fp16", action="store_true", help="Disable fp16 even on CUDA")
    ap.add_argument("--faiss_m", type=int, default=64)
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--ef_search", type=int, default=256)
    ap.add_argument("--corpus_prefix", default="passage: ", help='Prefix for corpus texts (e.g., "passage: " for BGE). Use "" to disable.')
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    blocks = load_blocks(Path(args.jsonl))

    # Dense input: short semantic signal. Use passage-prefix for BGE-style models.
    dense_texts = [prepare_corpus_text(b, args.corpus_prefix) for b in blocks]

    # Sparse input: keep section path + text so acronyms/IDs are matched by BM25.
    sparse_texts = [
        (b.get("section_path") or "") + "\n" + (b.get("text") or "")
        for b in blocks
    ]

    meta = [
        {
            "doc_id": b["doc_id"],
            "page": b["page"],
            "block_id": b["block_id"],
            "section_path": b.get("section_path", ""),
        }
        for b in blocks
    ]

    vecs = embed_texts_transformers(
        dense_texts,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        use_fp16=(not args.no_fp16),
        max_length=args.max_length,
    )

    index = build_faiss_hnsw(
        vecs,
        m=args.faiss_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    bm25 = BM25Okapi([tok(t) for t in sparse_texts])

    faiss.write_index(index, str(out / "dense_hnsw.faiss"))
    np.save(out / "dense.npy", vecs)
    with (out / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    with (out / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm": bm25, "texts": sparse_texts}, f)

    # Small manifest for sanity/debugging.
    (out / "MANIFEST.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "count": len(blocks),
                "dim": int(vecs.shape[1]),
                "faiss": {"type": "HNSWFlat", "M": args.faiss_m, "efSearch": args.ef_search},
                "bm25_corpus_size": len(sparse_texts),
                "corpus_prefix": args.corpus_prefix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()