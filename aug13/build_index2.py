#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

TOK = re.compile(r"[A-Za-z0-9_./-]+")


def tok(s: str) -> List[str]:
    return TOK.findall((s or "").lower())


def load_blocks(jsonl_path: Path) -> List[Dict[str, Any]]:
    lines = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading JSONL"):
            lines.append(json.loads(line))
    return lines


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


@torch.inference_mode()
def embed_texts(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    out_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
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
        out_vecs.append(normed.cpu().numpy().astype(np.float32))
    return np.vstack(out_vecs)


def build_faiss_hnsw(vecs: np.ndarray, m: int, ef_construction: int, ef_search: int) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vecs.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vecs)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--faiss_m", type=int, default=64)
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--ef_search", type=int, default=256)
    ap.add_argument("--corpus_prefix", default="passage: ")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(Path(args.jsonl))

    dense_texts = [(args.corpus_prefix + (b.get("text") or b.get("table_markdown") or "").strip()) for b in blocks]
    sparse_texts = [(b.get("section_path") or "") + "\n" + (b.get("text") or "") for b in blocks]
    meta = [{"doc_id": b["doc_id"], "page": b["page"], "block_id": b["block_id"], "section_path": b.get("section_path", "")} for b in blocks]

    vecs = embed_texts(
        dense_texts,
        model_name=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("Building FAISS index...")
    index = build_faiss_hnsw(vecs, args.faiss_m, args.ef_construction, args.ef_search)

    print("Building BM25 index...")
    bm25 = BM25Okapi([tok(t) for t in tqdm(sparse_texts, desc="BM25 tokenization")])

    print("Saving artifacts...")
    faiss.write_index(index, str(out / "dense_hnsw.faiss"))
    np.save(out / "dense.npy", vecs)
    with (out / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    with (out / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm": bm25, "texts": sparse_texts}, f)


if __name__ == "__main__":
    main()