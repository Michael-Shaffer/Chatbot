Yep—start with PDF ingestion + chunking, prove it’s sane, then layer retrieval. Here’s a tiny, no-frills way to do that, plus exactly how to test it.

Step 1 — Minimal ingestion & chunking

Save as ingest_min.py:

import os, glob, json, re, fitz

def read_pdf(path):
    doc = fitz.open(path)
    out = []
    for i in range(len(doc)):
        b = doc[i].get_text("blocks")
        b = sorted(b, key=lambda x:(x[1],x[0]))
        txt = "\n".join([t[4].strip() for t in b if t[4].strip()])
        out.append((i+1, txt))
    return out

def split_sections(text):
    heads = [m.start() for m in re.finditer(r"\n(?=[A-Z0-9][^\n]{0,80}\n)", text)]
    idx = [0]+heads+[len(text)]
    return [text[idx[i]:idx[i+1]].strip() for i in range(len(idx)-1) if text[idx[i]:idx[i+1]].strip()]

def chunk_doc(doc_id, pages, target=700, overlap=80):
    chunks=[]
    for pnum, ptxt in pages:
        secs = split_sections(ptxt) if len(ptxt)>1200 else [ptxt]
        for sec in secs:
            toks = sec.split()
            i=0
            while i<len(toks):
                j=min(len(toks), i+target)
                seg=" ".join(toks[i:j])
                pre=" ".join(toks[max(0,i-30):i])[:120]
                chunks.append({"doc_id":doc_id,"page":pnum,"section_path":pre,"text":seg})
                i=max(j-overlap,j)
    return chunks

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--pdf_glob", required=True)
    ap.add_argument("--out_jsonl", required=True)
    a=ap.parse_args()
    pdfs=sorted(glob.glob(a.pdf_glob))
    with open(a.out_jsonl,"w") as f:
        for p in pdfs:
            doc_id=os.path.splitext(os.path.basename(p))[0]
            pages=read_pdf(p)
            chunks=chunk_doc(doc_id, pages)
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False)+"\n")

Run:

python3 ingest_min.py --pdf_glob "/path/to/corpus/*.pdf" --out_jsonl corpus.chunks.jsonl

Step 2 — Sanity tests for ingestion

Save as test_ingestion_min.py:

import os, glob, json, random, fitz, statistics

def read_pdf_text_map(paths):
    out={}
    for p in paths:
        doc_id=os.path.splitext(os.path.basename(p))[0]
        d=fitz.open(p)
        s=[]
        empties=0
        for i in range(len(d)):
            t="\n".join([b[4].strip() for b in sorted(d[i].get_text("blocks"), key=lambda x:(x[1],x[0])) if b[4].strip()])
            if not t: empties+=1
            s.append(t)
        out[doc_id]={"full":"\n".join(s), "empty_pages":empties}
    return out

def load_chunks(jsonl):
    chunks=[]
    with open(jsonl,"r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def sample_hit_rate(src, hay, n=30, span=40):
    if len(src)<span: return 0.0
    hits=0
    for _ in range(n):
        i=random.randint(0, max(0,len(src)-span))
        snip=src[i:i+span]
        if snip in hay: hits+=1
    return hits/n

if __name__=="__main__":
    import argparse, re
    random.seed(7)
    ap=argparse.ArgumentParser()
    ap.add_argument("--pdf_glob", required=True)
    ap.add_argument("--chunks_jsonl", required=True)
    a=ap.parse_args()
    pdfs=sorted(glob.glob(a.pdf_glob))
    src=read_pdf_text_map(pdfs)
    chunks=load_chunks(a.chunks_jsonl)
    by_doc={}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    lens=[]
    total_hits=[]; total_empties=0; total_docs=0; total_chunks=0
    for doc_id,data in src.items():
        total_docs+=1
        total_empties+=data["empty_pages"]
        ch=by_doc.get(doc_id,[])
        total_chunks+=len(ch)
        hay=" ".join([x["text"] for x in ch])
        hits=sample_hit_rate(data["full"], hay, n=30, span=40)
        total_hits.append(hits)
        lens.extend([len(x["text"].split()) for x in ch])
    q = lambda L,p: L[int(max(0,min(len(L)-1, round(p*(len(L)-1)))))]

    lens_sorted=sorted(lens) if lens else [0]
    print("docs", total_docs)
    print("chunks", total_chunks)
    print("chunk_tokens_min_med_p95", q(lens_sorted,0.0), q(lens_sorted,0.5), q(lens_sorted,0.95))
    print("mean_hit_rate", round(sum(total_hits)/max(1,len(total_hits)),3))
    print("empty_pages", total_empties)
    print("status", "OK" if (sum(total_hits)/max(1,len(total_hits)))>=0.85 else "LOW")

Run:

python3 test_ingestion_min.py --pdf_glob "/path/to/corpus/*.pdf" --chunks_jsonl corpus.chunks.jsonl

What “good” looks like:
	•	mean_hit_rate ≥ 0.85
	•	chunk_tokens_med ≈ 600–800 with your current settings
	•	empty_pages = 0 for digital PDFs; if >0, you’ll need OCR (e.g., Tesseract) later

Step 3 — Quick BM25 smoke test (optional now, 2 minutes)

This lets you probe whether your chunks are queryable before embeddings.

# probe_bm25.py
import json, re, sys
from rank_bm25 import BM25Okapi

def tok(s): return re.findall(r"[A-Za-z0-9_./-]+", s.lower())
chunks=[json.loads(l) for l in open(sys.argv[1])]
texts=[c.get("section_path","")+"\n"+c["text"] for c in chunks]
bm=BM25Okapi([tok(t) for t in texts])
q=" ".join(sys.argv[2:]) or "acceptance test"
scores=bm.get_scores(tok(q))
top=sorted(range(len(scores)), key=lambda i:-scores[i])[:5]
for i in top:
    c=chunks[i]
    print(f"[{c['doc_id']}:{c['page']}] {texts[i][:240].replace('\n',' ')}")

Run:

python3 probe_bm25.py corpus.chunks.jsonl "your query here"

What to build next (after ingestion passes)
	1.	Dense embeddings + FAISS on the same chunks; run a retrieval-only probe (print top 5).
	2.	Hybrid union + MMR to improve diversity.
	3.	Reranker to clean the top set.
	4.	Context packer + Llama only after retrieval quality looks good.

If you want, I can drop a minimal retrieval-only probe for FAISS next.