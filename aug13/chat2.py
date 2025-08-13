# retriever.py
from __future__ import annotations
import json, pickle, re, numpy as np, faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
TOK = re.compile(r"[A-Za-z0-9_./-]+")
def tok(s:str)->List[str]: return TOK.findall((s or "").lower())

def mmr(q:np.ndarray, C:np.ndarray, ids:List[int], k:int, lam:float=0.5)->List[int]:
    sel=[]; sim=(C @ q.reshape(-1,1)).ravel()
    R=set(range(len(ids)))
    if not R: return []
    j=int(np.argmax(sim)); sel.append(j); R.remove(j)
    while len(sel)<min(k,len(ids)):
        best=-1e9; best_i=None
        for i in R:
            rep=max(float(C[i] @ C[s]) for s in sel)
            score=lam*float(sim[i]) - (1-lam)*rep
            if score>best: best, best_i = score, i
        sel.append(best_i); R.remove(best_i)
    return [ids[i] for i in sel]

class Retriever:
    def __init__(self, index_dir:str, emb_model:str="BAAI/bge-base-en-v1.5", rerank_model:str="BAAI/bge-reranker-large"):
        p=Path(index_dir)
        self.meta:List[Dict[str,Any]] = pickle.load((p/"meta.pkl").open("rb"))
        self.vecs:np.ndarray = np.load(p/"dense.npy")
        self.faiss = faiss.read_index(str(p/"dense_hnsw.faiss"))
        bm = pickle.load((p/"bm25.pkl").open("rb"))
        self.bm25, self.sparse_texts = bm["bm"], bm["texts"]
        self.embed = SentenceTransformer(emb_model)
        self.rerank = CrossEncoder(rerank_model)

    def hybrid_search(self, query:str, kd:int=80, ks:int=50, mmr_k:int=40, final_k:int=12)->List[Dict[str,Any]]:
        qv = self.embed.encode([query], normalize_embeddings=True)
        D,I = self.faiss.search(qv.astype(np.float32), kd)
        dense_ids = I[0].tolist()
        # sparse
        scores = self.bm25.get_scores(tok(query))
        sparse_ids = np.argsort(-scores)[:ks].tolist()
        # union
        cand = sorted(set(dense_ids) | set(sparse_ids))
        # MMR diversity
        mmr_ids = mmr(qv[0], self.vecs[cand], cand, k=mmr_k, lam=0.5)
        # rerank
        texts = [self.sparse_texts[i] for i in mmr_ids]
        pairs = [(query, t) for t in texts]
        rr = self.rerank.predict(pairs, show_progress_bar=False)
        order = np.argsort(-rr)[:final_k]
        top = [mmr_ids[i] for i in order]
        out=[]
        for i in top:
            m=self.meta[i]
            out.append({"block_id":m["block_id"],"doc_id":m["doc_id"],"page":m["page"],
                        "section_path":m.get("section_path",""),"text":self.sparse_texts[i]})
        return out