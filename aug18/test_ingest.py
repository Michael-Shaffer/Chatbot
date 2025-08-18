# test_ingest2.py
# Quick checks + demo printouts of chunks to eyeball quality.
# Run:
#   python test_ingest2.py --pdf path/to/file.pdf
# Or to just run unit tests (no PDF needed):
#   python -m pytest -q

from __future__ import annotations
import argparse
from pprint import pprint
from ingest import make_chunks, Block, assign_sections, compute_strict_section_intros
from retriever import Retriever

def demo_pdf(pdf_path: str, topn: int = 6) -> None:
    chunks = make_chunks(pdf_path)
    print(f"\n=== Sample Chunks ({min(topn, len(chunks))}/{len(chunks)}) ===")
    for ch in chunks[:topn]:
        print(f"- id={ch.chunk_id} page={ch.page} type={ch.block_type}")
        print(f"  section: {ch.section_path}")
        if ch.section_intro:
            print(f"  intro  : {ch.section_intro[:140]}")
        if ch.block_type == "table":
            print(f"  table  : {ch.table_summary}")
        print(f"  text   : {ch.text[:180]}\n")

    # Tiny retrieval smoke test
    r = Retriever()
    r.fit([ch.__dict__ for ch in chunks])
    for q in ("track smoothing parameters", "radar table", "ASR coverage"):
        hits = r.search(q, top_k=5)
        print(f"\n=== Query: {q} ===")
        for h in hits:
            print(f"* [{h['score']:.3f}] p{h['page']} {h['block_type']} :: {h['section_label']}")
            print(f"  {h['text'][:160]}\n")

def test_section_intro_strict_paragraph_only():
    # Synthetic mini-doc: Heading -> List -> Paragraph (intro must be empty)
    blocks = [
        Block(page=1, y0=10, x0=0, kind="heading", text="3.1 Track Smoothing", level=2),
        Block(page=1, y0=20, x0=0, kind="list", text="- bullet"),
        Block(page=1, y0=30, x0=0, kind="paragraph", text="This is a paragraph.", level=0),
        Block(page=1, y0=40, x0=0, kind="heading", text="3.2 Coasting", level=2),
        Block(page=1, y0=50, x0=0, kind="paragraph", text="Coasting intro.", level=0),
    ]
    assign_sections(blocks)
    intros = compute_strict_section_intros(blocks)
    assert intros.get(blocks[0].section_path, "") == ""
    assert intros.get(blocks[3].section_path, "") == "Coasting intro."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="", help="Path to a PDF to ingest and demo")
    args = ap.parse_args()
    if args.pdf:
        demo_pdf(args.pdf)
    else:
        print("No --pdf given; ran unit test stubs.")

if __name__ == "__main__":
    main()
