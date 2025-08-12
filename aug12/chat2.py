#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber


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


# ---------- text utilities ----------

_WS_RE = re.compile(r"\s+")


def norm_ws(s: Optional[str]) -> str:
    return _WS_RE.sub(" ", s or "").strip()


_HEADING_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 .:/()\-]{0,99}$")
_LIST_RE = re.compile(r"^(\- |\* |\d+[\.)] )")
_CODE_HINT_RE = re.compile(r"[{};]|^(code|snippet)", re.IGNORECASE)


def classify_line(text: str, is_heading: bool) -> BlockType:
    if is_heading:
        return "heading"
    if _LIST_RE.search(text):
        return "list"
    if _CODE_HINT_RE.search(text):
        return "code"
    return "paragraph"


def detect_heading_indices(lines: Sequence[str]) -> set[int]:
    idx: set[int] = set()
    for i, t in enumerate(lines):
        if len(t) <= 100 and _HEADING_RE.match(t):
            idx.add(i)
    return idx


def to_html(block_type: BlockType, text: str) -> str:
    esc = escape(text)
    if block_type == "heading":
        return f"<h3>{esc}</h3>"
    if block_type == "list":
        return f"<li>{esc}</li>"
    if block_type == "code":
        return f"<pre><code>{esc}</code></pre>"
    return f"<p>{esc}</p>"


def make_section_path(stack: Sequence[str], max_depth: int = 3) -> str:
    return " > ".join(stack[-max_depth:])


# ---------- PDF text extraction ----------

def extract_sorted_lines(page: fitz.Page) -> List[str]:
    blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
    out: List[str] = []
    for b in blocks:
        t = norm_ws(b[4])
        if t:
            out.append(t)
    return out


def build_typed_blocks(lines: Sequence[str]) -> List[Tuple[BlockType, str]]:
    headings = detect_heading_indices(lines)
    out: List[Tuple[BlockType, str]] = []
    for i, t in enumerate(lines):
        bt = classify_line(t, is_heading=(i in headings))
        out.append((bt, t))
    return out


# ---------- table extraction (pdfplumber) ----------

def render_markdown_table(columns: List[str], rows: List[List[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for r in rows:
        cells = [norm_ws(c) for c in r]
        if len(cells) < len(columns):
            cells += [""] * (len(columns) - len(cells))
        body_lines.append("| " + " | ".join(cells[:len(columns)]) + " |")
    return "\n".join([header, sep, *body_lines]) + "\n"


def extract_tables_pdfplumber(pdf_path: Path, page_index0: int, doc_id: str, tables_dir: Path) -> List[TableItem]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    items: List[TableItem] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index0]
        raw_tables = page.extract_tables() or []
        for i, tbl in enumerate(raw_tables):
            if not tbl:
                continue
            table_id = f"{doc_id}_tbl_{page_index0+1:03d}_{i:02d}"
            csv_path = tables_dir / f"{table_id}.csv"
            # Determine columns and rows
            cols = [norm_ws(x) for x in (tbl[0] if tbl else [])]
            if cols and any(col for col in cols):
                rows = tbl[1:]
            else:
                width = max(len(r) for r in tbl)
                cols = [f"Col{j+1}" for j in range(width)]
                rows = tbl
            # Write CSV
            csv_path.write_text(
                "\n".join([",".join([norm_ws(c) for c in row]) for row in rows]),
                encoding="utf-8"
            )
            # Markdown
            md = render_markdown_table(cols, rows)
            items.append(TableItem(table_id=table_id, csv_path=csv_path, markdown=md))
    return items


# ---------- record emission ----------

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

    table_items = extract_tables_pdfplumber(pdf_path, page_no - 1, doc_id, tables_dir)

    lines = extract_sorted_lines(page)
    typed_blocks = build_typed_blocks(lines)

    # Emit text blocks
    for bi, (bt, text) in enumerate(typed_blocks):
        if bt == "heading":
            section_stack.append(text)
        section_path = make_section_path(section_stack)
        html = to_html(bt, text)
        context_before = norm_ws(typed_blocks[bi - 1][1]) if bi > 0 else ""
        context_after = norm_ws(typed_blocks[bi + 1][1]) if bi + 1 < len(typed_blocks) else ""
        span_start, span_end = offset, offset + len(text) + 1
        offset = span_end
        rec = BlockRecord(
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
        )
        records.append(rec)

    # Emit table blocks
    for tbl in table_items:
        section_path = make_section_path(section_stack)
        span_start, span_end = offset, offset + len(tbl.markdown)
        offset = span_end
        rec = BlockRecord(
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
        )
        records.append(rec)

    return records, offset


def ingest_pdf_to_jsonl(pdf_path: Path, out_jsonl: Path, tables_dir: Path) -> None:
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem
    section_stack: List[str] = []
    offset = 0

    with out_jsonl.open("a", encoding="utf-8") as fo:
        for p in range(len(doc)):
            page_no = p + 1
            page = doc[p]
            records, offset = iter_page_records(
                pdf_path=pdf_path,
                doc_id=doc_id,
                page=page,
                page_no=page_no,
                section_stack=section_stack,
                offset=offset,
                ts_iso=ts_iso,
                tables_dir=tables_dir,
            )
            for r in records:
                fo.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Typed PDF ingestion (blocks + tables) for local RAG.")
    ap.add_argument("--pdf_glob", required=True, help='e.g. "/data/*.pdf"')
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--tables_dir", default="tables", help="Directory for extracted CSV tables")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    tables_dir = Path(args.tables_dir)
    for pdf in sorted(Path().glob(args.pdf_glob) if any(ch in args.pdf_glob for ch in "*?[]") else [Path(args.pdf_glob)]):
        if pdf.suffix.lower() != ".pdf":
            continue
        ingest_pdf_to_jsonl(pdf, out_jsonl, tables_dir)


if __name__ == "__main__":
    main()