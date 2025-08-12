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
