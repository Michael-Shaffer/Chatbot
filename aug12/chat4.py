#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

BlockType = Literal["heading", "list", "code", "paragraph", "table"]

@dataclass(frozen=True)
class TableItem:
    table_id: str
    csv_path: Path
    markdown: str

@dataclass(frozen=True)
class BlockRecord:
    doc_id: str
    page: int
    block_id: str
    block_type: BlockType
    section_path: str
    html: str
    markdown: str
    text: str
    table_id: Optional[str]
    table_markdown: Optional[str]
    table_csv_path: Optional[str]
    context_before: str
    context_after: str
    span_start: int
    span_end: int
    ts_extracted: str
    source_path: str

_WS_RE = re.compile(r"\s+")
_TOC_HDR_RE = re.compile(r"\bTABLE OF CONTENTS\b", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+){0,6}\s+.+?\.{3,}\s*\d+\s*$")
_NUM_HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,6})[)\s\-–—:]+(?P<title>\S.+)$")
_LIST_RE = re.compile(r"^(\- |\* |\d+[\.)] )")
_CODE_HINT_RE = re.compile(r"[{};]|^(code|snippet)", re.IGNORECASE)

def norm_ws(s: Optional[str]) -> str:
    return _WS_RE.sub(" ", s or "").strip()

def is_toc_page(lines: Sequence[str]) -> bool:
    head = " ".join(lines[:10])
    if _TOC_HDR_RE.search(head):
        return True
    dotted = sum(1 for ln in lines[:200] if _TOC_LINE_RE.match(ln))
    return dotted >= 3

def extract_sorted_lines(page: fitz.Page) -> List[Tuple[str, float, float]]:
    blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
    out: List[Tuple[str, float, float]] = []
    for b in blocks:
        t = norm_ws(b[4])
        if t:
            out.append((t, float(b[0]), float(b[1])))
    return out

def parse_numbered_heading(text: str) -> Optional[Tuple[List[int], str, str]]:
    m = _NUM_HEADING_RE.match(text)
    if not m:
        return None
    num_str = m.group("num")
    title = norm_ws(m.group("title"))
    parts = [int(x) for x in num_str.split(".") if x.isdigit()]
    if not parts or not title:
        return None
    label = f"{num_str} {title}"
    return parts, title, label

def classify_line(text: str, is_heading_candidate: bool) -> BlockType:
    if is_heading_candidate and parse_numbered_heading(text):
        return "heading"
    if _LIST_RE.search(text):
        return "list"
    if _CODE_HINT_RE.search(text):
        return "code"
    return "paragraph"

def to_html(block_type: BlockType, text: str) -> str:
    esc = escape(text)
    if block_type == "heading":
        return f"<h3>{esc}</h3>"
    if block_type == "list":
        return f"<li>{esc}</li>"
    if block_type == "code":
        return f"<pre><code>{esc}</code></pre>"
    return f"<p>{esc}</p>"

def update_section_stack(stack: List[str], levels: List[int], label: str) -> None:
    new_depth = len(levels)
    cur_depth = len(stack)
    if new_depth <= 0:
        return
    if new_depth <= cur_depth:
        del stack[new_depth - 1 :]
        stack.append(label)
    elif new_depth == cur_depth + 1:
        stack.append(label)
    else:
        stack.append(label)  # tolerate jumps (e.g., 2 -> 4 by appending)

def make_section_path(stack: Sequence[str], max_depth: int = 3) -> str:
    return " > ".join(stack[-max_depth:])

def build_typed_blocks(lines_xy: Sequence[Tuple[str, float, float]], skip_headings: bool) -> List[Tuple[BlockType, str]]:
    out: List[Tuple[BlockType, str]] = []
    for text, x0, y0 in lines_xy:
        is_left_aligned = x0 < 90.0
        is_heading_candidate = (not skip_headings) and is_left_aligned
        bt = classify_line(text, is_heading_candidate)
        out.append((bt, text))
    return out

def render_markdown_table(columns: List[str], rows: List[List[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        cells = [norm_ws(c) for c in r]
        cells += [""] * (len(columns) - len(cells))
        body.append("| " + " | ".join(cells[: len(columns)]) + " |")
    return "\n".join([header, sep, *body]) + "\n"

def extract_tables_pdfplumber(pdf_path: Path, page_index0: int, doc_id: str, tables_dir: Path) -> List[TableItem]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    items: List[TableItem] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index0]
        raw = page.extract_tables() or []
        for i, tbl in enumerate(raw):
            if not tbl:
                continue
            table_id = f"{doc_id}_tbl_{page_index0+1:03d}_{i:02d}"
            csv_path = tables_dir / f"{table_id}.csv"
            cols = [norm_ws(x) for x in (tbl[0] if tbl else [])]
            rows = tbl[1:] if (cols and any(cols)) else tbl
            if not cols:
                width = max(len(r) for r in tbl)
                cols = [f"Col{j+1}" for j in range(width)]
            csv_path.write_text("\n".join([",".join([norm_ws(c) for c in row]) for row in rows]), encoding="utf-8")
            md = render_markdown_table(cols, rows)
            items.append(TableItem(table_id, csv_path, md))
    return items

def iter_page_records(
    pdf_path: Path,
    doc_id: str,
    page: fitz.Page,
    page_no: int,
    section_stack: List[str],
    offset: int,
    ts_iso: str,
    tables_dir: Path,
) -> Tuple[List[BlockRecord], int]:
    records: List[BlockRecord] = []
    lines_xy = extract_sorted_lines(page)
    skip_headings = is_toc_page([t for t, _, _ in lines_xy])
    table_items = [] if skip_headings else extract_tables_pdfplumber(pdf_path, page_no - 1, doc_id, tables_dir)
    typed_blocks = build_typed_blocks(lines_xy, skip_headings)

    for bi, (bt, text) in enumerate(typed_blocks):
        if bt == "heading":
            parsed = parse_numbered_heading(text)
            if parsed:
                levels, title, label = parsed
                update_section_stack(section_stack, levels, label)
            else:
                bt = "paragraph"
        section_path = make_section_path(section_stack)
        html = to_html(bt, text)
        prev_text = typed_blocks[bi - 1][1] if bi > 0 else ""
        next_text = typed_blocks[bi + 1][1] if bi + 1 < len(typed_blocks) else ""
        context_before = norm_ws(prev_text)
        context_after = norm_ws(next_text)
        span_start, span_end = offset, offset + len(text) + 1
        offset = span_end
        records.append(BlockRecord(
            doc_id=doc_id,
            page=page_no,
            block_id=f"{doc_id}_p{page_no:03d}_b{bi:03d}",
            block_type=bt,
            section_path=section_path,
            html=html,
            markdown=text,
            text=text,
            table_id=None,
            table_markdown=None,
            table_csv_path=None,
            context_before=context_before,
            context_after=context_after,
            span_start=span_start,
            span_end=span_end,
            ts_extracted=ts_iso,
            source_path=str(pdf_path),
        ))

    for tbl in table_items:
        section_path = make_section_path(section_stack)
        span_start, span_end = offset, offset + len(tbl.markdown)
        offset = span_end
        records.append(BlockRecord(
            doc_id=doc_id,
            page=page_no,
            block_id=tbl.table_id,
            block_type="table",
            section_path=section_path,
            html=tbl.markdown,
            markdown=tbl.markdown,
            text=tbl.markdown,
            table_id=tbl.table_id,
            table_markdown=tbl.markdown,
            table_csv_path=str(tbl.csv_path),
            context_before="",
            context_after="",
            span_start=span_start,
            span_end=span_end,
            ts_extracted=ts_iso,
            source_path=str(pdf_path),
        ))

    return records, offset

def ingest_pdf_to_jsonl(pdf_path: Path, out_jsonl: Path, tables_dir: Path) -> None:
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem
    section_stack: List[str] = []
    offset = 0

    with out_jsonl.open("a", encoding="utf-8") as fo:
        for p in tqdm(range(len(doc)), desc=f"[{doc_id}] Extracting pages", leave=False):
            page_no = p + 1
            page = doc[p]
            records, offset = iter_page_records(pdf_path, doc_id, page, page_no, section_stack, offset, ts_iso, tables_dir)
            for r in records:
                fo.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Typed PDF ingestion (robust numbered sections) for local RAG.")
    ap.add_argument("--pdf_glob", required=True, help='e.g. "/data/*.pdf"')
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--tables_dir", default="tables", help="Directory for extracted CSV tables")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    tables_dir = Path(args.tables_dir)
    pdf_paths = sorted(Path().glob(args.pdf_glob)) if any(ch in args.pdf_glob for ch in "*?[]") else [Path(args.pdf_glob)]
    for pdf in tqdm(pdf_paths, desc="[Parsing PDFs]"):
        if pdf.suffix.lower() != ".pdf":
            continue
        ingest_pdf_to_jsonl(pdf, out_jsonl, tables_dir)

if __name__ == "__main__":
    main()