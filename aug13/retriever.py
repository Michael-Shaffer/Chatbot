#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer


class Retriever:
    def __init__(
        self,
        index_dir: str,
        embed_model_path: str,
        reranker_path: Optional[str] = None,
        device: str = "cpu",
        query_prefix: str = "",
    ):
        p = Path(index_dir)
        self.query_prefix = query_prefix
        self.device = device

        # Dense
        self.index = faiss.read_index(str(p / "dense_hnsw.faiss"))
        self.vecs = np.load(p / "dense.npy", mmap_mode="r")

        # Metadata
        with (p / "meta.pkl").open("rb") as f:
            self.meta = pickle.load(f)

        # BM25
        with (p / "bm25.pkl").open("rb") as f:
            bm25_data = pickle.load(f)
        self.bm25: BM25Okapi = bm25_data["bm"]
        self.sparse_texts: List[str] = bm25_data["texts"]

        # Embedder
        self.embed_tok = AutoTokenizer.from_pretrained(embed_model_path)
        self.embed_model = AutoModel.from_pretrained(embed_model_path)
        self.embed_model.to(device).eval()

        # Optional reranker
        self.reranker = None
        if reranker_path:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_path, device=device)

    def embed_query(self, query: str) -> np.ndarray:
        enc = self.embed_tok(
            self.query_prefix + query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            out = self.embed_model(**enc)
            pooled = (out.last_hidden_state * enc["attention_mask"].unsqueeze(-1)).sum(1)
            pooled /= enc["attention_mask"].sum(1, keepdim=True)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def hybrid_search(
        self,
        query: str,
        kd: int = 50,
        ks: int = 50,
        mmr_k: int = 30,
        final_k: int = 8,
    ) -> List[Dict[str, Any]]:
        # Dense
        qvec = self.embed_query(query)
        _, dense_ids = self.index.search(qvec, kd)

        # Sparse
        bm25_scores = self.bm25.get_scores(self.query_prefix.split() + query.lower().split())
        sparse_ids = np.argsort(bm25_scores)[::-1][:ks]

        # Merge
        cand_ids = list(set(dense_ids[0]) | set(sparse_ids))
        cand_blocks = [(i, self.meta[i], bm25_scores[i]) for i in cand_ids]

        # Optional rerank
        if self.reranker:
            pairs = [(query, self.sparse_texts[i]) for i in cand_ids]
            scores = self.reranker.predict(pairs)
            cand_blocks = [(i, meta, score) for (i, meta, _), score in zip(cand_blocks, scores)]

        # Sort by score and return
        cand_blocks.sort(key=lambda x: x[2], reverse=True)
        top_ids = [i for i, _, _ in cand_blocks[:final_k]]

        results = []
        for idx in top_ids:
            block = self.meta[idx].copy()
            block["text"] = self.sparse_texts[idx]
            results.append(block)
        return results