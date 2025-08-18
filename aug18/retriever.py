# retriever.py
# Local hybrid retrieval: BM25 + TF-IDF cosine, intro fusion, table intent boost, MMR diversification.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re

@dataclass
class Doc:
    chunk_id: str
    block_type: str
    text: str
    section_intro: str
    section_path: str
    section_label: str
    table_markdown: str
    table_summary: str
    page: int
    hidden_terms: List[str]

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _detect_table_intent(q: str) -> bool:
    ql = q.lower()
    for k in ("table", "schema", "matrix", "columns", "parameters", "fields"):
        if k in ql:
            return True
    return False

class Retriever:
    def __init__(self):
        self.docs: List[Doc] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tok_corpus: List[List[str]] = []
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X = None
        self.intro_X = None
        self.acro_longforms: List[str] = []  # global set of expansions

    # --------------------------
    # Fitting
    # --------------------------

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        self.docs = [
            Doc(
                chunk_id=c["chunk_id"],
                block_type=c["block_type"],
                text=_normalize(c.get("text")),
                section_intro=_normalize(c.get("section_intro")),
                section_path=_normalize(c.get("section_path")),
                section_label=_normalize(c.get("section_label")),
                table_markdown=c.get("table_markdown", ""),
                table_summary=_normalize(c.get("table_summary")),
                page=c.get("page", 0),
                hidden_terms=list({*_to_list(c.get("hidden_terms", []))}),
            )
            for c in chunks
        ]
        bm25_texts = [
            " ".join([d.section_path, d.section_intro, d.text, " ".join(d.hidden_terms)])
            for d in self.docs
        ]
        self.tok_corpus = [t.split() for t in bm25_texts]
        self.bm25 = BM25Okapi(self.tok_corpus)

        body_corpus = [d.text for d in self.docs]
        intro_corpus = [d.section_intro for d in self.docs]
        self.X = self.tfidf.fit_transform(body_corpus)
        self.intro_X = self.tfidf.transform(intro_corpus)

        # collect expansions once (helps query expansion)
        all_exp = set()
        for d in self.docs:
            all_exp.update(d.hidden_terms)
        self.acro_longforms = sorted(all_exp)

    # --------------------------
    # Search
    # --------------------------

    def search(self, query: str, top_k: int = 10, mmr_lambda: float = 0.7) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        q = _normalize(query)
        q_bm25 = self._expand_for_bm25(q)
        bm25_scores = self._bm25_scores(q_bm25)

        intro, dense = self._tfidf_scores(q)
        fused_dense = [max(dense[i], intro[i] * 1.1) for i in range(len(self.docs))]

        scores = self._combine(bm25_scores, fused_dense, weight_bm25=0.6)
        if _detect_table_intent(q):
            scores = self._boost_tables(scores, factor=1.25)

        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:100]
        mmr = self._mmr(ranked, scores, mmr_lambda, top_k)
        return [self._to_result(i, scores[i]) for i in mmr]

    # --------------------------
    # Internals
    # --------------------------

    def _expand_for_bm25(self, q: str) -> str:
        # simple expansion: append up to 3 longforms if query looks like an acronym
        tokens = q.split()
        acro = [t for t in tokens if t.isupper() and 2 < len(t) <= 12]
        if not acro or not self.acro_longforms:
            return q
        return q + " " + " ".join(self.acro_longforms[:3])

    def _bm25_scores(self, q: str) -> List[float]:
        tokens = q.split()
        return self.bm25.get_scores(tokens) if self.bm25 else [0.0] * len(self.docs)

    def _tfidf_scores(self, q: str) -> Tuple[List[float], List[float]]:
        qvec = self.tfidf.transform([q])
        body = cosine_similarity(qvec, self.X).ravel().tolist()
        intro = cosine_similarity(qvec, self.intro_X).ravel().tolist()
        return intro, body

    def _combine(self, a: List[float], b: List[float], weight_bm25: float = 0.6) -> List[float]:
        w = weight_bm25
        return [w * _z(a[i]) + (1 - w) * _z(b[i]) for i in range(len(a))]

    def _boost_tables(self, scores: List[float], factor: float) -> List[float]:
        out = scores[:]
        for i, d in enumerate(self.docs):
            if d.block_type == "table":
                out[i] *= factor
        return out

    def _mmr(self, ranked: List[int], scores: List[float], lamb: float, k: int) -> List[int]:
        selected: List[int] = []
        cand = set(ranked)
        while cand and len(selected) < k:
            if not selected:
                i = max(cand, key=lambda x: scores[x])
                selected.append(i)
                cand.remove(i)
                continue
            best_i, best_val = None, -1e9
            for i in list(cand):
                sim = max(self._pair_sim(i, j) for j in selected) if selected else 0.0
                val = lamb * scores[i] - (1 - lamb) * sim
                if val > best_val:
                    best_i, best_val = i, val
            selected.append(best_i)
            cand.remove(best_i)
        return selected

    def _pair_sim(self, i: int, j: int) -> float:
        # cosine over TF-IDF body vectors
        vi = self.X[i]
        vj = self.X[j]
        num = (vi.multiply(vj)).sum()
        denom = math.sqrt(vi.multiply(vi).sum()) * math.sqrt(vj.multiply(vj).sum())
        return float(num / denom) if denom else 0.0

    def _to_result(self, i: int, score: float) -> Dict[str, Any]:
        d = self.docs[i]
        return {
            "chunk_id": d.chunk_id,
            "score": float(score),
            "page": d.page,
            "block_type": d.block_type,
            "section_path": d.section_path,
            "section_label": d.section_label,
            "section_intro": d.section_intro,
            "text": d.text,
            "table_summary": d.table_summary,
            "table_markdown": d.table_markdown,
        }

# --------------------------
# Utilities
# --------------------------

def _to_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(y) for y in x]
    return [str(x)]
