# build_index.py
from __future__ import annotations
import json, pickle, numpy as np, faiss, re
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

TOK = re.compile(r"[A-Za-z0-9_./-]+")

def tok(s:str)->List[str]: return TOK.findall((s or "").lower())

def load_blocks(jsonl:Path)->List[Dict[str,Any]]:
    return [json.loads(l) for l in jsonl.open("r", encoding="utf-8")]

def embed_texts(model:SentenceTransformer, texts:List[str])->np.ndarray:
    v = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(v, dtype=np.float32)

def main(inp:str, outdir:str, emb_model:str="BAAI/bge-base-en-v1.5")->None:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    blocks = load_blocks(Path(inp))
    # what we embed (short and semantic); tables embed their markdown
    texts = [ (b.get("text") or b.get("table_markdown") or "") for b in blocks ]
    # what BM25 sees (keeps acronyms/paths helpful)
    sparse_texts = [ (b.get("section_path") or "") + "\n" + (b.get("text") or "") for b in blocks ]
    meta = [ {"doc_id":b["doc_id"], "page":b["page"], "block_id":b["block_id"], "section_path":b.get("section_path","")} for b in blocks ]

    emb = SentenceTransformer(emb_model)
    vecs = embed_texts(emb, texts)

    idx = faiss.IndexHNSWFlat(vecs.shape[1], 64)
    idx.hnsw.efSearch = 256
    idx.hnsw.efConstruction = 200
    idx.add(vecs)

    bm25 = BM25Okapi([tok(t) for t in sparse_texts])

    faiss.write_index(idx, str(out/"dense_hnsw.faiss"))
    np.save(out/"dense.npy", vecs)
    with (out/"meta.pkl").open("wb") as f: pickle.dump(meta, f)
    with (out/"bm25.pkl").open("wb") as f: pickle.dump({"bm": bm25, "texts": sparse_texts}, f)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--emb", default="BAAI/bge-base-en-v1.5")
    a=ap.parse_args()
    main(a.jsonl, a.outdir, a.emb)