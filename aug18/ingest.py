#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest PDFs -> JSONL for RAG with correct section_intro (first paragraph after header).

Features:
- Robust numbered heading detection (e.g., "2", "3.1", "4.2.7 – Title").
- Excludes heading lines from paragraph stream so section_intro is the *real* intro paragraph.
- Section metadata: section_path ("1.2.3"), section_label ("1.2.3 Title"), section_intro.
- Optional table extraction via pdfplumber (graceful fallback if unavailable).
- Page-level markdown snapshots (basic), banner filtering, stable IDs, token estimate.
- Clean, shallow control flow for readability and maintainability.

Requires: PyMuPDF (fitz). Optional: pdfplumber.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF

try:
    import pdfplumber  # optional
except Exception:  # noqa: BLE001
    pdfplumber = None  # type: ignore

# --------------------------- Config ---------------------------

HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,6})[)\s\-–—:]+(?P<title>\S.+)$"
)
# Accept bare numbers as headings only when followed by non-empty title on same line.
# Tweak tolerance to your PDFs if a few headers slip through.
Y_TOL = 0.6  # points; align with text extraction jitter
LEFT_MARGIN_MAX_FOR_HEAD = 120.0  # points; small x0 suggests a real block heading

BANNER_MIN_Y_FRAC = 0.88  # bottom banner lines live below this fraction of page height

BLOCK_ID_FMT = "{doc_id}_p{page:03d}_b{i:04d}"
MD_PAGE_ID_FMT = "{doc_id}_p{page:03d}"

# --------------------------- Types ---------------------------

@dataclass(frozen=True)
class Line:
    page: int
    x0: float
    y0: float
    size: float
    text: str

@dataclass(frozen=True)
class Heading:
    parts: List[str]
    title: str
    label: str      # "1.2.3 Title"
    page: int
    y0: float
    x0: float
    size: float

@dataclass(frozen=True)
class Chunk:
    page: int
    block_type: str          # "heading" | "paragraph" | "table"
    text: str                # for table, can be markdown/plain
    section_path: str        # "1.2.3"
    section_label: str       # "1.2.3 Title"

@dataclass(frozen=True)
class Record:
    doc_id: str
    page: int
    block_id: string
    block_type: str
    section_path: str
    section_label: str
    section_intro: str
    text: str
    tokens_est: int
    ts_extracted: str
    source_path: str
    md_page_id: str

# --------------------------- Small utils ---------------------------

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def est_tokens(s: str) -> int:
    # Rough/fast token estimate ~ 4 chars/token
    return max(1, (len(s) + 3) // 4)

def is_banner_line(line: Line, page_h: float) -> bool:
    return line.y0 >= page_h * BANNER_MIN_Y_FRAC

# --------------------------- PDF -> lines ---------------------------

def extract_lines_from_page(doc: fitz.Document, page_index: int) -> Tuple[List[Line], float]:
    page = doc[page_index]
    page_h = page.rect.height
    data = page.get_text("dict")
    out: List[Line] = []

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            # A line can have multiple spans; we form a single text with basic metrics
            x0 = min((span.get("bbox", [0, 0, 0, 0])[0] for span in line.get("spans", []) if span.get("text")), default=0.0)
            y0 = min((span.get("bbox", [0, 0, 0, 0])[1] for span in line.get("spans", []) if span.get("text")), default=0.0)
            size = max((span.get("size", 0.0) for span in line.get("spans", []) if span.get("text")), default=0.0)
            text = norm_ws("".join(span.get("text", "") for span in line.get("spans", [])))
            if not text:
                continue
            out.append(Line(page=page_index + 1, x0=x0, y0=y0, size=size, text=text))

    out.sort(key=lambda l: (l.page, l.y0, l.x0))
    return out, page_h

def all_lines(doc: fitz.Document) -> Tuple[List[Line], Dict[int, float]]:
    lines: List[Line] = []
    page_heights: Dict[int, float] = {}
    for i in range(len(doc)):
        plines, ph = extract_lines_from_page(doc, i)
        lines.extend(plines)
        page_heights[i + 1] = ph
    return lines, page_heights

# --------------------------- Heading detection ---------------------------

def parse_heading_from_text(line: Line) -> Optional[Heading]:
    m = HEADING_RE.match(line.text)
    if not m:
        return None
    title = norm_ws(m.group("title"))
    if not title:
        return None
    parts = m.group("num").split(".")
    label = f"{m.group('num')} {title}"
    return Heading(
        parts=parts,
        title=title,
        label=label,
        page=line.page,
        y0=line.y0,
        x0=line.x0,
        size=line.size,
    )

def detect_headings(lines: Iterable[Line]) -> List[Heading]:
    heads: List[Heading] = []
    for ln in lines:
        h = parse_heading_from_text(ln)
        if not h:
            continue
        if ln.x0 > LEFT_MARGIN_MAX_FOR_HEAD:
            continue
        heads.append(h)
    heads.sort(key=lambda h: (h.page, h.y0))
    return heads

def heads_by_page(heads: List[Heading]) -> Dict[int, List[Heading]]:
    byp: Dict[int, List[Heading]] = {}
    for h in heads:
        byp.setdefault(h.page, []).append(h)
    for p in byp:
        byp[p].sort(key=lambda hh: hh.y0)
    return byp

# --------------------------- Section assignment ---------------------------

def nearest_active_heading(ln: Line, page_heads: List[Heading], active: Optional[Heading]) -> Optional[Heading]:
    eligible = [h for h in page_heads if h.y0 <= ln.y0 + Y_TOL]
    return eligible[-1] if eligible else active

def as_sec_path_label(h: Optional[Heading]) -> Tuple[str, str]:
    if not h:
        return "", ""
    return ".".join(h.parts), h.label

# --------------------------- Chunk building ---------------------------

def filter_non_heading_paragraphs(
    lines: List[Line],
    page_heads_map: Dict[int, List[Heading]],
    page_heights: Dict[int, float],
) -> List[Line]:
    flines: List[Line] = []
    for ln in lines:
        if is_banner_line(ln, page_heights[ln.page]):
            continue
        t = ln.text
        if not t:
            continue

        # exclude lines that are the heading itself (y-match) or look like a heading by regex + left margin/size
        p_heads = page_heads_map.get(ln.page, [])
        is_head_y = any(abs(ln.y0 - h.y0) <= Y_TOL for h in p_heads)
        is_head_like = parse_heading_from_text(ln) is not None and ln.x0 <= LEFT_MARGIN_MAX_FOR_HEAD
        if is_head_y or is_head_like:
            continue

        flines.append(ln)
    return flines

def chunk_paragraphs(
    lines: List[Line],
    page_heads_map: Dict[int, List[Heading]],
) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf: List[str] = []
    cur_page = 1
    cur_sec_path, cur_sec_label = "", ""
    active: Optional[Heading] = None

    def flush():
        if not buf:
            return
        text = norm_ws(" ".join(buf))
        if text:
            chunks.append(Chunk(page=cur_page, block_type="paragraph",
                                text=text, section_path=cur_sec_path, section_label=cur_sec_label))
        buf.clear()

    for ln in lines:
        active = nearest_active_heading(ln, page_heads_map.get(ln.page, []), active)
        cur_sec_path, cur_sec_label = as_sec_path_label(active)
        cur_page = ln.page
        buf.append(ln.text)

    flush()
    return chunks

def chunk_headings(heads: List[Heading]) -> List[Chunk]:
    out: List[Chunk] = []
    for h in heads:
        sec_path, sec_label = ".".join(h.parts), h.label
        out.append(Chunk(page=h.page, block_type="heading",
                         text=h.label, section_path=sec_path, section_label=sec_label))
    return out

# --------------------------- Section intros ---------------------------

def build_section_intro_map(chunks: List[Chunk]) -> Dict[str, str]:
    intros: Dict[str, str] = {}
    seen: set[str] = set()

    for ck in chunks:
        if ck.block_type == "heading" and ck.section_path:
            seen.add(ck.section_path)
            continue
        if ck.block_type == "paragraph" and ck.section_path and ck.section_path in seen:
            if ck.section_path not in intros:
                intros[ck.section_path] = ck.text
    return intros

# --------------------------- Page markdown ---------------------------

def page_markdown(lines: List[Line], page_no: int) -> str:
    txt = "\n".join(l.text for l in lines if l.page == page_no)
    return txt.strip()

# --------------------------- Tables (optional) ---------------------------

def extract_tables_markdown(pdf_path: Path) -> Dict[int, List[str]]:
    if pdfplumber is None:
        return {}
    out: Dict[int, List[str]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            md_blocks: List[str] = []
            try:
                tables = page.extract_tables() or []
            except Exception:  # noqa: BLE001
                tables = []
            for tbl in tables:
                if not tbl:
                    continue
                # simple GitHub-flavored markdown table rendering
                header = tbl[0]
                rows = tbl[1:] if len(tbl) > 1 else []
                line1 = "| " + " | ".join(c or "" for c in header) + " |"
                line2 = "| " + " | ".join("---" for _ in header) + " |"
                body = ["| " + " | ".join(c or "" for c in r) + " |" for r in rows]
                md = "\n".join([line1, line2, *body])
                md_blocks.append(md)
            if md_blocks:
                out[i] = md_blocks
    return out

def table_chunks_with_sections(
    table_md_by_page: Dict[int, List[str]],
    page_heads_map: Dict[int, List[Heading]],
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page, md_list in table_md_by_page.items():
        active: Optional[Heading] = None
        # We assign each table on a page to the most recent heading on that page (last heading before bottom)
        hs = page_heads_map.get(page, [])
        last = hs[-1] if hs else None
        active = last
        sec_path, sec_label = as_sec_path_label(active)
        for md in md_list:
            chunks.append(Chunk(page=page, block_type="table",
                                text=md, section_path=sec_path, section_label=sec_label))
    return chunks

# --------------------------- Emit ---------------------------

def emit_records(
    doc_id: str,
    pdf_path: Path,
    chunks: List[Chunk],
    intro_map: Dict[str, str],
    out_jsonl: Path,
    total_pages: int,
) -> None:
    ts = now_iso()
    with out_jsonl.open("a", encoding="utf-8") as fo:
        for i, ck in enumerate(chunks):
            md_page_id = MD_PAGE_ID_FMT.format(doc_id=doc_id, page=ck.page)
            rec = {
                "doc_id": doc_id,
                "page": ck.page,
                "block_id": BLOCK_ID_FMT.format(doc_id=doc_id, page=ck.page, i=i),
                "block_type": ck.block_type,
                "section_path": ck.section_path,
                "section_label": ck.section_label,
                "section_intro": intro_map.get(ck.section_path, ""),
                "text": ck.text,
                "tokens_est": est_tokens(ck.text),
                "ts_extracted": ts,
                "source_path": str(pdf_path),
                "md_page_id": md_page_id,
                "pages_total": total_pages,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------------- Driver ---------------------------

def process_pdf(pdf_path: Path, out_jsonl: Path) -> None:
    with fitz.open(pdf_path) as doc:
        lines, page_heights = all_lines(doc)
        total_pages = len(doc)

    heads = detect_headings(lines)
    page_heads_map = heads_by_page(heads)

    # Build heading chunks (for metadata/TOC purposes)
    heading_chunks = chunk_headings(heads)

    # Filter out headings from paragraph stream (this is the critical fix)
    non_head_lines = filter_non_heading_paragraphs(lines, page_heads_map, page_heights)

    # Build paragraph chunks
    para_chunks = chunk_paragraphs(non_head_lines, page_heads_map)

    # Optional table chunks
    tables_md = extract_tables_markdown(pdf_path)
    table_chunks = table_chunks_with_sections(tables_md, page_heads_map)

    # Merge in logical order: headings (for reference), paragraphs, then tables
    chunks = [*heading_chunks, *para_chunks, *table_chunks]

    # Section intros: first paragraph chunk after each heading
    intro_map = build_section_intro_map(chunks)

    emit_records(
        doc_id=pdf_path.stem,
        pdf_path=pdf_path,
        chunks=chunks,
        intro_map=intro_map,
        out_jsonl=out_jsonl,
        total_pages=total_pages,
    )

def write_page_markdown_snapshots(pdf_path: Path, md_out_dir: Path) -> None:
    md_out_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        lines_by_page: Dict[int, List[Line]] = {}
        all_lns, _ = all_lines(doc)
        for ln in all_lns:
            lines_by_page.setdefault(ln.page, []).append(ln)
        for page_no, lns in lines_by_page.items():
            md = page_markdown(lns, page_no)
            pid = MD_PAGE_ID_FMT.format(doc_id=pdf_path.stem, page=page_no)
            (md_out_dir / f"{pid}.md").write_text(md + "\n", encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDFs to JSONL with correct section_intro.")
    ap.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--md-dir", type=Path, default=None, help="Optional directory to write page markdown snapshots")
    args = ap.parse_args()

    if args.out.exists():
        args.out.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pdfs: List[Path] = []
    if args.input.is_file() and args.input.suffix.lower() == ".pdf":
        pdfs = [args.input]
    elif args.input.is_dir():
        pdfs = sorted(p for p in args.input.rglob("*.pdf"))
    else:
        raise SystemExit("Provide a PDF file or a directory containing PDFs.")

    for p in pdfs:
        process_pdf(p, args.out)
        if args.md_dir:
            write_page_markdown_snapshots(p, args.md_dir)

if __name__ == "__main__":
    main()
