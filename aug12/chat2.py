# ingest_v2.py
import os, glob, re, json, time, fitz, camelot, csv
from html import escape

def norm_ws(s): return re.sub(r"\s+", " ", s).strip()

def detect_headings(lines):
    H=[]
    for i,t in enumerate(lines):
        if len(t)<=100 and re.match(r"^[A-Z0-9][A-Za-z0-9 .:/()-]{0,99}$", t):
            H.append(i)
    return set(H)

def blocks_from_page(page):
    B = sorted(page.get_text("blocks"), key=lambda x:(x[1],x[0]))
    lines = [b[4].strip() for b in B if b[4].strip()]
    hidx = detect_headings(lines)
    out=[]
    for i,t in enumerate(lines):
        if i in hidx:
            out.append(("heading", t))
        elif re.match(r"^(\- |\* |\d+[\.)] )", t): 
            out.append(("list", t))
        elif re.search(r"[{};]|^(code|snippet)", t.lower()):
            out.append(("code", t))
        else:
            out.append(("paragraph", t))
    return out

def make_section_path(stack, t):
    s = " > ".join(stack[-3:])
    return s

def extract_tables(pdf_path, page_num, doc_id, tables_dir):
    os.makedirs(tables_dir, exist_ok=True)
    tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="lattice")
    items=[]
    for i, tbl in enumerate(tables):
        tid=f"{doc_id}_tbl_{page_num:03d}_{i:02d}"
        csv_path=os.path.join(tables_dir, f"{tid}.csv")
        tbl.to_csv(csv_path)
        md="| " + " | ".join(tbl.df.columns) + " |\n| " + " | ".join(["---"]*len(tbl.df.columns)) + " |\n"
        for _,row in tbl.df.iterrows():
            md += "| " + " | ".join([str(x) for x in row.tolist()]) + " |\n"
        items.append((tid, csv_path, md))
    return items

def ingest_pdf(pdf_path, out_jsonl, tables_dir):
    doc = fitz.open(pdf_path)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    canon=[]; offset=0
    with open(out_jsonl, "a") as fo:
        section_stack=[]
        table_cache={}
        for p in range(len(doc)):
            page = doc[p]
            page_no = p+1
            tbl_items = extract_tables(pdf_path, page_no, doc_id, tables_dir)
            for tid,cpath,md in tbl_items:
                table_cache[tid]=(cpath,md)
            blocks = blocks_from_page(page)
            for bi,(bt,raw) in enumerate(blocks):
                t = norm_ws(raw)
                if bt=="heading":
                    section_stack.append(t)
                section_path = make_section_path(section_stack, t)
                html = "<p>"+escape(t)+"</p>"
                md = t
                table_id=None; table_md=None; table_csv_path=None
                if bt=="list":
                    html = "<li>"+escape(t)+"</li>"
                if bt=="code":
                    html = "<pre><code>"+escape(t)+"</code></pre>"
                if bt=="heading":
                    html = "<h3>"+escape(t)+"</h3>"
                if bt=="paragraph" and re.match(r"^(Table|TAB)\s*\d+[:\s-]", t):
                    prev = tbl_items[0] if tbl_items else None
                context_before = norm_ws(blocks[bi-1][1]) if bi>0 else ""
                context_after  = norm_ws(blocks[bi+1][1]) if bi+1<len(blocks) else ""
                span_start=offset; span_end=offset+len(t)+1
                offset=span_end
                rec = {
                    "doc_id":doc_id,"page":page_no,"block_id":f"{doc_id}_p{page_no:03d}_b{bi:03d}",
                    "block_type":bt,"section_path":section_path,
                    "html":html,"markdown":md,"text":t,
                    "table_id":table_id,"table_markdown":table_md,"table_csv_path":table_csv_path,
                    "context_before":context_before,"context_after":context_after,
                    "span_start":span_start,"span_end":span_end,
                    "ts_extracted":ts,"source_path":pdf_path
                }
                fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
            for i,(tid,cpath,mdtbl) in enumerate(tbl_items):
                rec = {
                    "doc_id":doc_id,"page":page_no,"block_id":f"{tid}",
                    "block_type":"table","section_path":make_section_path(section_stack, ""),
                    "html":mdtbl,"markdown":mdtbl,"text":mdtbl,
                    "table_id":tid,"table_markdown":mdtbl,"table_csv_path":cpath,
                    "context_before":"","context_after":"",
                    "span_start":offset,"span_end":offset+len(mdtbl),
                    "ts_extracted":ts,"source_path":pdf_path
                }
                offset+=len(mdtbl)
                fo.write(json.dumps(rec, ensure_ascii=False)+"\n")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--pdf_glob", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--tables_dir", default="tables")
    a=ap.parse_args()
    for p in sorted(glob.glob(a.pdf_glob)):
        ingest_pdf(p, a.out_jsonl, a.tables_dir)