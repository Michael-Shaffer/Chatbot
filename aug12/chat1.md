Here’s a single-file, minimal local RAG pipeline wired for FAISS+BM25+bge rerank + vLLM Llama (OpenAI-compatible server on localhost). No comments, no error checks.

# rag_min.py
import os, glob, json, math, re, fitz, numpy as np, faiss, requests
from dataclasses import dataclass
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

@dataclass
class Chunk:
    doc_id: str
    page: int
    section_path: str
    text: str

def read_pdf(path):
    doc = fitz.open(path)
    out = []
    for i in range(len(doc)):
        page = doc[i]
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b:(b[1],b[0]))
        lines = []
        for b in blocks:
            t = b[4].strip()
            if t:
                lines.append(t)
        text = "\n".join(lines)
        out.append((i+1, text))
    return out

def split_sections(text):
    heads = [m.start() for m in re.finditer(r"\n(?=[A-Z0-9][^\n]{0,80}\n)", text)]
    idxs = [0]+heads+[len(text)]
    return [text[idxs[i]:idxs[i+1]].strip() for i in range(len(idxs)-1) if text[idxs[i]:idxs[i+1]].strip()]

def tokenize_for_bm25(s):
    return re.findall(r"[A-Za-z0-9_./-]+", s.lower())

def chunk_doc(doc_id, pages, target=700, overlap=80):
    chunks = []
    for pnum, ptxt in pages:
        secs = split_sections(ptxt) if len(ptxt)>1200 else [ptxt]
        for sec in secs:
            toks = sec.split()
            i=0
            while i<len(toks):
                j=min(len(toks), i+target)
                seg=" ".join(toks[i:j])
                header = toks[max(0,i-30):i]
                pre = " ".join(header)
                sp = ""
                if pre:
                    sp = pre[:120]
                chunks.append(Chunk(doc_id, pnum, sp, seg))
                i = max(j-overlap, j)
    return chunks

class Dense:
    def __init__(self, name="BAAI/bge-base-en-v1.5", device=None):
        self.m = SentenceTransformer(name, device=device)
        self.dim = self.m.get_sentence_embedding_dimension()
    def encode(self, texts:List[str]):
        v = self.m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

class Reranker:
    def __init__(self, name="BAAI/bge-reranker-large", device=None):
        self.m = CrossEncoder(name, device=device)
    def score(self, query, texts:List[str]):
        pairs = [(query, t) for t in texts]
        s = self.m.predict(pairs, batch_size=32, show_progress_bar=False)
        return np.asarray(s, dtype=np.float32)

class Index:
    def __init__(self, dim, hnsw_m=64, ef_search=256):
        self.index = faiss.IndexHNSWFlat(dim, hnsw_m)
        self.index.hnsw.efSearch = ef_search
        self.vecs = None
        self.meta = []
        self.bm25 = None
        self.bm25_texts = []
        self.bm25_tok = []
    def add(self, vecs:np.ndarray, meta:List[Chunk]):
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
        self.index.add(vecs)
        self.meta.extend(meta)
    def add_bm25(self, texts:List[str]):
        self.bm25_texts = texts
        self.bm25_tok = [tokenize_for_bm25(t) for t in texts]
        self.bm25 = BM25Okapi(self.bm25_tok)
    def search_dense(self, qv, k):
        D,I = self.index.search(qv, k)
        return I[0], D[0]
    def search_bm25(self, q, k):
        scores = self.bm25.get_scores(tokenize_for_bm25(q))
        I = np.argsort(-scores)[:k]
        return I, scores[I]

def mmr(query_vec, cand_vecs, cand_idx, k, lam=0.5):
    selected=[]
    sim = cand_vecs @ query_vec
    sim = sim.reshape(-1)
    remaining=set(range(len(cand_idx)))
    if not remaining: return []
    cur = int(np.argmax(sim))
    selected.append(cur)
    remaining.remove(cur)
    while len(selected)<min(k,len(cand_idx)):
        best=-1e9; best_i=None
        for i in remaining:
            rep = max(cand_vecs[i] @ cand_vecs[j] for j in selected)
            score = lam*sim[i] - (1-lam)*rep
            if score>best:
                best=score; best_i=i
        selected.append(best_i); remaining.remove(best_i)
    return [cand_idx[i] for i in selected]

def build_corpus(input_glob):
    pdfs = sorted(glob.glob(input_glob))
    all_chunks=[]
    for p in pdfs:
        doc_id=os.path.splitext(os.path.basename(p))[0]
        pages=read_pdf(p)
        ch=chunk_doc(doc_id, pages)
        all_chunks.extend(ch)
    return all_chunks

def build_indexes(chunks:List[Chunk], embed:Dense):
    texts=[c.text for c in chunks]
    vecs=embed.encode(texts)
    idx=Index(vecs.shape[1],64,256)
    idx.add(vecs, chunks)
    idx.add_bm25([c.section_path+"\n"+c.text for c in chunks])
    return idx

def union_dense_bm25(idx:Index, q, qv, kd=80, ks=50):
    I_d,_=idx.search_dense(qv, kd)
    I_s,_=idx.search_bm25(q, ks)
    s=set(I_d.tolist())|set(I_s.tolist())
    cand=sorted(list(s))
    return cand

def rerank_subset(rerank:Reranker, q, idx:Index, ids:List[int], topn=20):
    texts=[idx.meta[i].text for i in ids]
    scores=rerank.score(q, texts)
    order=np.argsort(-scores)[:min(topn,len(ids))]
    return [ids[i] for i in order], scores[order]

def pack_context(chunks:List[Chunk], ids:List[int], budget_chars=9000):
    out=[]
    used=0
    for i in ids:
        c=chunks[i]
        tag=f"[{c.doc_id}:{c.page}]"
        t=tag+" "+c.text.strip()
        if used+len(t)+2>budget_chars: break
        out.append(t); used+=len(t)+2
    return "\n\n".join(out)

def llama_complete(endpoint, model, system, user, max_new_tokens=800, temperature=0.2, top_p=0.9):
    payload={"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"max_tokens":max_new_tokens,"temperature":temperature,"top_p":top_p}
    r=requests.post(endpoint,headers={"Content-Type":"application/json"},data=json.dumps(payload),timeout=300)
    j=r.json()
    return j["choices"][0]["message"]["content"]

def answer(q, embed:Dense, idx:Index, rerank:Reranker, k_d=80, k_s=50, mmr_k=40, final_k=10, ctx_chars=9000, llama_url="http://localhost:8000/v1/chat/completions", llama_model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    qv=embed.encode([q])
    cand_ids=union_dense_bm25(idx, q, qv, k_d, k_s)
    cand_vecs=idx.vecs[cand_ids]
    mmr_ids=mmr(qv[0], cand_vecs, cand_ids, k=mmr_k, lam=0.5)
    final_ids,_=rerank_subset(rerank, q, idx, mmr_ids, topn=final_k)
    ctx=pack_context(idx.meta, final_ids, budget_chars=ctx_chars)
    system="You answer strictly from the provided sources. Cite using bracketed tags like [doc_id:page] after each claim. If an answer is not supported, say you don't know."
    user=f"Query:\n{q}\n\nSources:\n{ctx}\n\nAnswer with citations."
    return llama_complete(llama_url, llama_model, system, user)

def build_and_query(pdf_glob, query):
    embed=Dense("BAAI/bge-base-en-v1.5")
    rerank=Reranker("BAAI/bge-reranker-large")
    chunks=build_corpus(pdf_glob)
    idx=build_indexes(chunks, embed)
    out=answer(query, embed, idx, rerank)
    return out

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--pdf_glob", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--llama_url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--llama_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    a=ap.parse_args()
    embed=Dense("BAAI/bge-base-en-v1.5")
    rerank=Reranker("BAAI/bge-reranker-large")
    chunks=build_corpus(a.pdf_glob)
    idx=build_indexes(chunks, embed)
    print(answer(a.query, embed, idx, rerank, llama_url=a.llama_url, llama_model=a.llama_model))

Run:

python3 rag_min.py --pdf_glob "/path/to/corpus/*.pdf" --query "What are the acceptance test steps for subsystem X?" \
  --llama_url "http://localhost:8000/v1/chat/completions" \
  --llama_model "meta-llama/Meta-Llama-3.1-8B-Instruct"

Swap models if needed:
	•	Embeddings: BAAI/bge-large-en-v1.5
	•	Reranker: jinaai/jina-reranker-v2-base-multilingual
	•	Llama served via vLLM in INT4/AWQ.