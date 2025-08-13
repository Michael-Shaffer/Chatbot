#!/usr/bin/env python3
from __future__ import annotations

import argparse, pickle, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

TOK = re.compile(r"[A-Za-z0-9_./-]+")
def tok(s: str) -> List[str]: return TOK.findall((s or "").lower())

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.inference_mode()
def embed_queries(
    queries: List[str],
    model_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
    prefix: str = "query: ",
    use_fp16: bool = True,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else None
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype).to(device).eval()
    vecs = []
    for i in range(0, len(queries), batch_size):
        batch = [f"{prefix}{q}" if prefix else q for q in queries[i:i+batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        rep = mdl(**enc).last_hidden_state
        pooled = mean_pool(rep, enc["attention_mask"])
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs.append(normed.detach().cpu().to(torch.float32).numpy())
    return np.vstack(vecs)

class LocalCrossEncoder:
    def __init__(self, model_dir: str, device: str = "cuda", use_fp16: bool = True) -> None:
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else None
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def score(self, query: str, passages: List[str], max_length: int = 512, batch_size: int = 32) -> np.ndarray:
        scores = []
        for i in range(0, len(passages), batch_size):
            p = passages[i:i+batch_size]
            enc = self.tok([query]*len(p), p, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(self.mdl.device) for k, v in enc.items()}
            logits = self.mdl(**enc).logits  # shape: (B, num_labels) or (B,1)
            if logits.shape[-1] == 1:
                s = logits.squeeze(-1)
            else:
                # if classification head, use positive logit
                s = logits[:, -1]
            scores.append(s.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)

def mmr(qv: np.ndarray, cand_vecs: np.ndarray, ids: List[int], k: int, lam: float = 0.5) -> List[int]:
    sel: List[int] = []
    sim = cand_vecs @ qv.reshape(-1, 1)
    sim = sim.ravel()
    R = set(range(len(ids)))
    if not R: return []
    j = int(np.argmax(sim)); sel.append(j); R.remove(j)
    while len(sel) < min(k, len(ids)):
        best, best_i = -1e9, None
        for i in R:
            rep = max(float(cand_vecs[i] @ cand_vecs[s]) for s in sel)
            score = lam * float(sim[i]) - (1 - lam) * rep
            if score > best:
                best, best_i = score, i
        sel.append(best_i); R.remove(best_i)
    return [ids[i] for i in sel]

class Retriever:
    def __init__(
        self,
        index_dir: str,
        embed_model_path: str,
        device: str = "auto",
        reranker_path: str | None = None,
        use_fp16: bool = True,
        query_prefix: str = "query: ",
    ) -> None:
        p = Path(index_dir)
        self.meta: List[Dict[str, Any]] = pickle.load((p/"meta.pkl").open("rb"))
        self.vecs: np.ndarray = np.load(p/"dense.npy")
        self.faiss = faiss.read_index(str(p/"dense_hnsw.faiss"))
        bm = pickle.load((p/"bm25.pkl").open("rb"))
        self.bm25 = bm["bm"]; self.sparse_texts: List[str] = bm["texts"]

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.embed_model_path = embed_model_path
        self.use_fp16 = use_fp16
        self.query_prefix = query_prefix

        self.reranker: LocalCrossEncoder | None = None
        if reranker_path and Path(reranker_path).exists():
            self.reranker = LocalCrossEncoder(reranker_path, device=self.device, use_fp16=use_fp16)

    def hybrid_search(self, query: str, kd: int = 80, ks: int = 50, mmr_k: int = 40, final_k: int = 12) -> List[Dict[str, Any]]:
        qv = embed_queries(
            [query], model_dir=self.embed_model_path, device=self.device,
            batch_size=32, max_length=512, prefix=self.query_prefix, use_fp16=self.use_fp16
        )[0]
        D, I = self.faiss.search(qv.reshape(1, -1).astype(np.float32), kd)
        dense_ids = I[0].tolist()

        scores = self.bm25.get_scores(tok(query))
        sparse_ids = np.argsort(-scores)[:ks].tolist()

        cand = sorted(set(dense_ids) | set(sparse_ids))
        if not cand: return []

        mmr_ids = mmr(qv, self.vecs[cand], cand, k=mmr_k, lam=0.5)

        if self.reranker:
            texts = [self.sparse_texts[i] for i in mmr_ids]
            rr = self.reranker.score(query, texts, max_length=512, batch_size=32)
            order = np.argsort(-rr)[:final_k]
        else:
            # fallback: cosine similarity on dense vectors
            cos = (self.vecs[mmr_ids] @ qv.reshape(-1,1)).ravel()
            order = np.argsort(-cos)[:final_k]

        top = [mmr_ids[i] for i in order]
        out = []
        for i in top:
            m = self.meta[i]
            out.append({
                "block_id": m["block_id"], "doc_id": m["doc_id"], "page": m["page"],
                "section_path": m.get("section_path",""), "text": self.sparse_texts[i]
            })
        return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline hybrid retriever with local HF encoder and optional local reranker.")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model_path", required=True, help="Local dir for encoder (same as used in build_index.py)")
    ap.add_argument("--reranker_path", default="", help="Local dir for cross-encoder reranker; empty disables rerank")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--query_prefix", default="query: ", help='Use "" to disable (non-BGE).')
    args = ap.parse_args()

    RET = Retriever(
        index_dir=args.index_dir,
        embed_model_path=args.embed_model_path,
        device=args.device,
        reranker_path=(args.reranker_path or None),
        use_fp16=not args.no_fp16,
        query_prefix=args.query_prefix,
    )

    # quick smoke test
    res = RET.hybrid_search("power on self test sequence", final_k=5)
    for r in res:
        print(f"[{r['doc_id']}:{r['page']}] {r['section_path']} -> {r['text'][:160].replace('\\n',' ')}")