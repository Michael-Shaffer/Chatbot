#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# --------------------------- Utils ---------------------------

TOK = re.compile(r"[A-Za-z0-9_./-]+")

def tok(s: str) -> List[str]:
    return TOK.findall((s or "").lower())

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def safe_read_text(path: Optional[str], max_chars: int) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    return (txt[:max_chars] + " …") if len(txt) > max_chars else txt

# --------------------------- Optional local cross-encoder ---------------------------

class LocalCrossEncoder:
    def __init__(self, model_dir: str, device: str = "cpu", use_fp16: bool = False) -> None:
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else None
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()
        self.max_len = int(getattr(self.mdl.config, "max_position_embeddings", 512) or 512)

    @torch.inference_mode()
    def score(self, query: str, passages: List[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
        scores: List[np.ndarray] = []
        it = range(0, len(passages), batch_size)
        it = tqdm(it, desc="Reranking", leave=False) if progress else it
        for i in it:
            p = passages[i : i + batch_size]
            enc = self.tok(
                [query] * len(p),
                p,
                padding="max_length",
                truncation="only_second",   # keep query intact, trim passage
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.mdl.device) for k, v in enc.items()}
            logits = self.mdl(**enc).logits
            s = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
            scores.append(s.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)

# --------------------------- MMR ---------------------------

def mmr(qv: np.ndarray, cand_vecs: np.ndarray, ids: List[int], k: int, lam: float = 0.5, progress: bool = False) -> List[int]:
    sel: List[int] = []
    sim = (cand_vecs @ qv.reshape(-1, 1)).ravel()
    remaining = set(range(len(ids)))
    if not remaining:
        return []
    j0 = int(np.argmax(sim))
    sel.append(j0)
    remaining.remove(j0)
    steps = min(k, len(ids)) - 1
    it = range(steps)
    it = tqdm(it, desc="MMR diversify", leave=False) if progress and steps > 0 else it
    for _ in it:
        best, best_i = -1e9, None
        for i in remaining:
            rep = max(float(cand_vecs[i] @ cand_vecs[s]) for s in sel)
            score = lam * float(sim[i]) - (1 - lam) * rep
            if score > best:
                best, best_i = score, i
        sel.append(best_i)  # type: ignore
        remaining.remove(best_i)  # type: ignore
    return [ids[i] for i in sel]

# --------------------------- Retriever ---------------------------

class Retriever:
    def __init__(
        self,
        index_dir: str,
        embed_model_path: str,
        reranker_path: Optional[str] = None,
        device: str = "auto",
        query_prefix: str = "query: ",
        use_fp16: bool = True,
        progress: bool = True,
        # NEW: page excerpt options
        append_page_excerpt: bool = True,
        page_excerpt_chars: int = 450,
        page_excerpt_limit: int = 3,
    ) -> None:
        p = Path(index_dir)
        self.progress = progress
        self.query_prefix = query_prefix
        self.append_page_excerpt = append_page_excerpt
        self.page_excerpt_chars = page_excerpt_chars
        self.page_excerpt_limit = page_excerpt_limit

        # Dense + metadata
        self.index = faiss.read_index(str(p / "dense_hnsw.faiss"))
        self.vecs = np.load(p / "dense.npy", mmap_mode="r")

        with (p / "meta.pkl").open("rb") as f:
            self.meta: List[Dict[str, Any]] = pickle.load(f)

        with (p / "bm25.pkl").open("rb") as f:
            bm25_data = pickle.load(f)
        self.bm25: BM25Okapi = bm25_data["bm"]
        self.sparse_texts: List[str] = bm25_data["texts"]

        # Device + embedder
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        dtype = torch.float16 if (use_fp16 and self.device.startswith("cuda")) else None
        self.embed_tok = AutoTokenizer.from_pretrained(embed_model_path, local_files_only=True)
        self.embed_mdl = AutoModel.from_pretrained(
            embed_model_path, local_files_only=True, torch_dtype=dtype
        ).to(self.device).eval()
        self.embed_max_len = int(getattr(self.embed_mdl.config, "max_position_embeddings", 512) or 512)

        # Optional reranker
        self.reranker: Optional[LocalCrossEncoder] = None
        if reranker_path:
            self.reranker = LocalCrossEncoder(reranker_path, device=self.device, use_fp16=use_fp16)

        # Tame CPU threads a bit
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        try:
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        except Exception:
            pass

    @torch.inference_mode()
    def embed_query(self, query: str) -> np.ndarray:
        text = self.query_prefix + query if self.query_prefix else query
        enc = self.embed_tok(
            text,
            padding=True,
            truncation=True,
            max_length=self.embed_max_len,
            return_tensors="pt",
        ).to(self.embed_mdl.device)
        out = self.embed_mdl(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normed.detach().cpu().to(torch.float32).numpy()

    def _append_page_excerpts(self, top_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Given the ranked final content ids, optionally append one page excerpt per unique page,
        up to self.page_excerpt_limit pages (ordered by first appearance).
        """
        results: List[Dict[str, Any]] = []
        # 1) pack normal content blocks
        for idx in top_ids:
            m = self.meta[idx]
            results.append(
                {
                    "block_id": m.get("block_id"),
                    "doc_id": m.get("doc_id"),
                    "page": m.get("page"),
                    "block_type": m.get("block_type", "paragraph"),
                    "section_path": m.get("section_path", ""),
                    "section_label": m.get("section_label", ""),
                    "section_intro": m.get("section_intro", ""),
                    "text": self.sparse_texts[idx],
                    "page_md_path": m.get("page_md_path"),
                }
            )

        if not self.append_page_excerpt or self.page_excerpt_limit <= 0:
            return results

        # 2) pick unique pages in the same order and add an excerpt block per page (after content)
        seen_pages: set = set()
        page_slots = 0
        for idx in top_ids:
            m = self.meta[idx]
            page = int(m.get("page", 0) or 0)
            if page <= 0 or page in seen_pages:
                continue
            seen_pages.add(page)
            if page_slots >= self.page_excerpt_limit:
                continue
            page_slots += 1

            doc_id = str(m.get("doc_id", ""))
            path = m.get("page_md_path")
            excerpt = safe_read_text(path, self.page_excerpt_chars)
            if not excerpt:
                continue

            results.append(
                {
                    "block_id": f"{doc_id}_p{page:03d}_page_excerpt",
                    "doc_id": doc_id,
                    "page": page,
                    "block_type": "page_excerpt",
                    "section_path": "",
                    "section_label": "",
                    "section_intro": "",
                    "text": f"[Page overview] {excerpt}",
                    "page_md_path": path,
                    "is_page_excerpt": True,
                }
            )
        return results

    def hybrid_search(
        self,
        query: str,
        kd: int = 80,
        ks: int = 50,
        mmr_k: int = 40,
        final_k: int = 12,
    ) -> List[Dict[str, Any]]:
        t0 = time.time()
        if self.progress:
            tqdm.write(f"Query: {query}")

        # 1) Embed
        with tqdm(total=1, desc="Embedding query", leave=False, disable=not self.progress):
            qv = self.embed_query(query)[0]

        # 2) Dense FAISS
        t_dense0 = time.time()
        _, I = self.index.search(qv.reshape(1, -1).astype(np.float32), kd)
        dense_ids = I[0].tolist()
        if self.progress:
            tqdm.write(f"Dense search: {len(dense_ids)} hits in {time.time()-t_dense0:.3f}s")

        # 3) BM25
        t_bm0 = time.time()
        scores = self.bm25.get_scores(tok(query))
        sparse_ids = np.argsort(-scores)[:ks].tolist()
        if self.progress:
            tqdm.write(f"Sparse search: {len(sparse_ids)} hits in {time.time()-t_bm0:.3f}s")

        # 4) Union + MMR
        cand = sorted(set(dense_ids) | set(sparse_ids))
        if not cand:
            return []

        mmr_ids = mmr(qv, self.vecs[cand], cand, k=mmr_k, lam=0.5, progress=self.progress)

        # 5) Rerank or cosine fallback
        if self.reranker:
            texts = [self.sparse_texts[i] for i in mmr_ids]
            rr = self.reranker.score(query, texts, batch_size=32, progress=self.progress)
            order = np.argsort(-rr)[:final_k]
        else:
            cos = (self.vecs[mmr_ids] @ qv.reshape(-1, 1)).ravel()
            order = np.argsort(-cos)[:final_k]

        top_ids = [mmr_ids[i] for i in order]

        # 6) Pack results (+ optional page excerpts)
        out = self._append_page_excerpts(top_ids)

        if self.progress:
            tqdm.write(f"Done in {time.time()-t0:.3f}s (content_k={len(top_ids)}, returned={len(out)})")
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Hybrid retriever with optional page-level Markdown excerpts.")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model_path", required=True, help="Local dir for encoder")
    ap.add_argument("--reranker_path", default="", help="Local dir for cross-encoder; empty disables")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--query_prefix", default="query: ")
    ap.add_argument("--query", default="power on self test sequence")

    ap.add_argument("--kd", type=int, default=80)
    ap.add_argument("--ks", type=int, default=50)
    ap.add_argument("--mmr_k", type=int, default=40)
    ap.add_argument("--final_k", type=int, default=10)
    ap.add_argument("--no_progress", action="store_true")

    # NEW flags
    ap.add_argument("--no_page_excerpts", action="store_true", help="Disable appending page-level Markdown excerpts.")
    ap.add_argument("--page_excerpt_chars", type=int, default=450)
    ap.add_argument("--page_excerpt_limit", type=int, default=3)

    args = ap.parse_args()

    RET = Retriever(
        index_dir=args.index_dir,
        embed_model_path=args.embed_model_path,
        reranker_path=(args.reranker_path or None),
        device=args.device,
        query_prefix=args.query_prefix,
        use_fp16=not args.no_fp16,
        progress=not args.no_progress,
        append_page_excerpt=not args.no_page_excerpts,
        page_excerpt_chars=args.page_excerpt_chars,
        page_excerpt_limit=args.page_excerpt_limit,
    )

    res = RET.hybrid_search(args.query, kd=args.kd, ks=args.ks, mmr_k=args.mmr_k, final_k=args.final_k)
    for r in res:
        tag = f"[{r.get('doc_id','?')}:{r.get('page','?')}]"
        bt = r.get("block_type", "?")
        prefix = "(PAGE)" if r.get("is_page_excerpt") else ""
        snippet = (r.get("text","") or "").replace("\n", " ")
        print(f"{tag} {bt} {prefix} -> {snippet[:160]}")
