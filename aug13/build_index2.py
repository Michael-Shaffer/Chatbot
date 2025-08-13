#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pickle, re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

TOK = re.compile(r"[A-Za-z0-9_./-]+")
def tok(s: str) -> List[str]: return TOK.findall((s or "").lower())

def norm_text(b: Dict[str, Any]) -> str:
    t = (b.get("text") or b.get("table_markdown") or "").strip()
    return t

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.inference_mode()
def embed_corpus(
    texts: List[str],
    model_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 512,
    prefix: str = "passage: ",
    use_fp16: bool = True,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else None
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype)
    mdl.to(device).eval()

    out = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}{t}" if prefix else t for t in texts[i:i+batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        rep = mdl(**enc).last_hidden_state
        pooled = mean_pool(rep, enc["attention_mask"])
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(normed.detach().cpu().to(torch.float32).numpy())
    return np.vstack(out) if out else np.zeros((0, mdl.config.hidden_size), dtype=np.float32)

def build_hnsw(vecs: np.ndarray, m: int = 64, ef_con: int = 200, ef_search: int = 256) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vecs.shape[1], m)
    index.hnsw.efConstruction = ef_con
    index.hnsw.efSearch = ef_search
    index.add(vecs.astype(np.float32))
    return index

def main() -> None:
    ap = argparse.ArgumentParser(description="Build offline FAISS+BM25 index with local HF encoder.")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model_path", required=True, help="Local dir for encoder (e.g., /models/bge-base-en-v1.5)")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--faiss_m", type=int, default=64)
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--ef_search", type=int, default=256)
    ap.add_argument("--corpus_prefix", default="passage: ", help='Use "" to disable (non-BGE).')
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device

    blocks = [json.loads(l) for l in Path(args.jsonl).open("r", encoding="utf-8")]
    dense_texts = [norm_text(b) for b in blocks]
    sparse_texts = [(b.get("section_path") or "") + "\n" + (b.get("text") or "") for b in blocks]
    meta = [{"doc_id":b["doc_id"], "page":b["page"], "block_id":b["block_id"], "section_path":b.get("section_path","")} for b in blocks]

    vecs = embed_corpus(
        dense_texts, model_dir=args.model_path, device=device, batch_size=args.batch_size,
        max_length=args.max_length, prefix=args.corpus_prefix, use_fp16=not args.no_fp16
    )
    index = build_hnsw(vecs, m=args.faiss_m, ef_con=args.ef_construction, ef_search=args.ef_search)
    bm25 = BM25Okapi([tok(t) for t in sparse_texts])

    faiss.write_index(index, str(out/"dense_hnsw.faiss"))
    np.save(out/"dense.npy", vecs)
    with (out/"meta.pkl").open("wb") as f: pickle.dump(meta, f)
    with (out/"bm25.pkl").open("wb") as f: pickle.dump({"texts": sparse_texts, "bm": bm25}, f)
    (out/"MANIFEST.json").write_text(json.dumps({
        "encoder_dir": args.model_path, "device": device, "count": len(blocks),
        "dim": int(vecs.shape[1]), "faiss": {"M": args.faiss_m, "efSearch": args.ef_search},
        "corpus_prefix": args.corpus_prefix
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()