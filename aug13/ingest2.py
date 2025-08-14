#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

# --------------------------- Types & Records ---------------------------

BlockType = Literal["heading", "paragraph", "list", "code", "table"]

@dataclass(frozen=True)
class Line:
    text: str
    x0: float
    y0: float
    size: float  # char-weighted avg size
    page: int

@dataclass(frozen=True)
class HeadingCand:
    page: int
    y0: float
    parts: List[str]           # ["4","1","3"] or ["A","1"] if you later add appendices
    label: str                 # "4.1.3 Title"
    title: str                 # "Title"
    size: float
    x0: float
    source: str                # "outline" or "text"

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
    section_label: str
    section_intro: str
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

# --------------------------- Regexes ---------------------------

_WS_RE = re.compile(r"\s+")
_TOC_HDR_RE = re.compile(r"\bTABLE OF CONTENTS\b", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+){0,6}\s+.+?\.{3,}\s*\d+\s*$")

# Allow numbered headings only (1..99 with dotted numeric sublevels)
_NUM_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,6})[)\s\-–—:]+(?P<title>\S.+)$"
)

_MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE_LIKE = re.compile(
    rf"^(?:\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}|{_MONTHS}\s+\d{{1,2}},\s*\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}/\d{{2}}/\d{{4}})$",
    re.IGNORECASE,
)

_LIST_RE = re.compile(r"^(\- |\* |\d+[\.)] )")
_CODE_HINT_RE = re.compile(r"[{};]|^(code|snippet|example)$", re.IGNORECASE)

def norm_ws(s: Optional[str]) -> str:
    return _WS_RE.sub(" ", s or "").strip()

def is_date_like(text: str) -> bool:
    return bool(_DATE_LIKE.match(text.strip()))

def is_toc_page(lines: Sequence[str]) -> bool:
    head = " ".join(lines[:12])
    if _TOC_HDR_RE.search(head):
        return True
    dotted = sum(1 for ln in lines[:300] if _TOC_LINE_RE.match(ln))
    return dotted >= 3

def parse_numbered_heading(text: str) -> Optional[Tuple[List[str], str, str]]:
    if is_date_like(text):
        return None
    m = _NUM_HEADING_RE.match(text.strip())
    if not m:
        return None
    num = m.group("num")
    title = norm_ws(m.group("title"))
    if not title:
        return None
    return num.split("."), title, f"{num} {title}"

# --------------------------- Extraction (PyMuPDF) ---------------------------

def extract_lines(doc: fitz.Document, page_index: int) -> List[Line]:
    page = doc[page_index]
    d = page.get_text("dict")
    lines: List[Line] = []
    for blk in d.get("blocks", []):
        if blk.get("type", 0) != 0:
            continue
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            text = norm_ws("".join(s.get("text", "") for s in spans))
            if not text:
                continue
            x0 = min(float(s["bbox"][0]) for s in spans)
            y0 = min(float(s["bbox"][1]) for s in spans)
            sizes = [(float(s.get("size", 0.0)), len(s.get("text", ""))) for s in spans]
            total = sum(w for _, w in sizes) or 1
            avg_size = sum(sz * w for sz, w in sizes) / total
            lines.append(Line(text=text, x0=x0, y0=y0, size=avg_size, page=page_index+1))
    lines.sort(key=lambda L: (L.y0, L.x0))
    return lines

def page_median_size(lines: List[Line]) -> float:
    return median([ln.size for ln in lines]) if lines else 10.0

# --------------------------- Tables (pdfplumber) ---------------------------

def extract_tables_pdfplumber(pdf_path: Path, page_index0: int, doc_id: str, tables_dir: Path) -> List[TableItem]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    items: List[TableItem] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_index0 < 0 or page_index0 >= len(pdf.pages):
            return items
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
            csv_path.write_text(
                "\n".join([",".join([norm_ws(c) for c in row]) for row in rows]),
                encoding="utf-8",
            )
            md = render_markdown_table(cols, rows)
            items.append(TableItem(table_id, csv_path, md))
    return items

def render_markdown_table(columns: List[str], rows: List[List[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        cells = [norm_ws(c) for c in r]
        cells += [""] * (len(columns) - len(cells))
        body.append("| " + " | ".join(cells[: len(columns)]) + " |")
    return "\n".join([header, sep, *body]) + "\n"

# --------------------------- Headings: candidates & global healing ---------------------------

def outline_to_candidates(doc: fitz.Document) -> List[HeadingCand]:
    cands: List[HeadingCand] = []
    try:
        toc = doc.get_toc(simple=True)  # [(level, title, page), ...]
    except Exception:
        toc = []
    for lv, title, page1 in toc or []:
        if page1 <= 0:
            continue
        # Heuristic: TOC titles often include numbering already
        parsed = parse_numbered_heading(title)
        if not parsed:
            continue
        parts, title_txt, label = parsed
        # y0 unknown from outline; keep 0 to sort before body on that page
        cands.append(HeadingCand(page=page1, y0=0.0, parts=parts, label=label,
                                 title=title_txt, size=999.0, x0=0.0, source="outline"))
    return cands

def text_to_candidates(lines: List[Line], allow_headings: bool, med_size: float) -> List[HeadingCand]:
    if not allow_headings:
        return []
    out: List[HeadingCand] = []
    for ln in lines:
        if ln.x0 >= 120.0:
            continue
        parsed = parse_numbered_heading(ln.text)
        if not parsed:
            continue
        parts, title, label = parsed
        # prefer visible headings: slightly larger than median or clearly left-aligned numbered
        if ln.size + 0.01 < med_size:
            continue
        out.append(HeadingCand(page=ln.page, y0=ln.y0, parts=parts, label=label, title=title,
                               size=ln.size, x0=ln.x0, source="text"))
    return out

def reconcile_headings(cands: List[HeadingCand]) -> List[HeadingCand]:
    """
    Build a prefix-consistent, monotonic hierarchy across the document:
      - sort by (page, y0), with outline candidates already having y0=0 and huge size
      - majors are non-decreasing; subsections must share prefix with current chain
      - if a valid subsection appears with a different major, start a new chain (healing)
    """
    cands_sorted = sorted(cands, key=lambda c: (c.page, c.y0, -c.size))
    accepted: List[HeadingCand] = []
    chain: List[str] = []  # tokens like ["4","1","3"]
    last_major: Optional[int] = None

    for c in cands_sorted:
        parts = c.parts
        is_major = len(parts) == 1 and parts[0].isdigit()
        if is_major:
            major = int(parts[0])
            if last_major is not None and major < last_major:
                # ignore backwards majors
                continue
            accepted.append(c)
            chain = parts[:]
            last_major = major
            continue

        # subsections: if different major, heal by starting a new chain at this major
        if not chain or parts[0] != chain[0]:
            if parts[0].isdigit():
                accepted.append(c)
                chain = parts[:]
                if len(parts) == 1:
                    last_major = int(parts[0])
            else:
                # non-digit major not supported in this version
                continue
            continue

        # enforce prefix; if missing parents, synthesize by resetting chain
        prefix_ok = True
        for i in range(min(len(chain), len(parts)-1)):
            if chain[i] != parts[i]:
                prefix_ok = False
                break
        if not prefix_ok:
            accepted.append(c)
            chain = parts[:]
            continue

        # normal child advance
        accepted.append(c)
        chain = parts[:]

    return accepted

# --------------------------- Section map, intro capture ---------------------------

def index_headings_by_page(headings: List[HeadingCand]) -> Dict[int, List[HeadingCand]]:
    byp: Dict[int, List[HeadingCand]] = {}
    for h in headings:
        byp.setdefault(h.page, []).append(h)
    for p in byp:
        byp[p].sort(key=lambda h: h.y0)
    return byp

@dataclass
class SectionState:
    tokens: List[str]
    label: str
    intro_captured: bool

def assign_section_for_blocks(
    lines_by_page: Dict[int, List[Line]],
    headings_by_page: Dict[int, List[HeadingCand]],
) -> Dict[Tuple[int, float], SectionState]:
    """
    For each line location (page,y), track the most recent heading.
    Returns a map keyed by (page,y_anchor) -> SectionState for fast lookup while emitting blocks.
    """
    section_for_pos: Dict[Tuple[int, float], SectionState] = {}
    active_tokens: List[str] = []
    active_label: str = ""
    for page in sorted(lines_by_page.keys()):
        heads = headings_by_page.get(page, [])
        line_list = lines_by_page[page]
        hi = 0
        for ln in line_list:
            # advance heading pointer if any headings occur before this line
            while hi < len(heads) and heads[hi].y0 <= ln.y0 + 0.01:
                h = heads[hi]
                active_tokens = h.parts[:]
                active_label = h.label
                hi += 1
            section_for_pos[(page, ln.y0)] = SectionState(tokens=active_tokens[:], label=active_label, intro_captured=False)
    return section_for_pos

def section_key(tokens: List[str], max_depth: int = 3) -> str:
    if not tokens:
        return ""
    return " > ".join([".".join(tokens[:i+1]) for i in range(min(len(tokens), max_depth))])

# --------------------------- Typing & Rendering ---------------------------

def classify_block(text: str, x0: float) -> BlockType:
    if _LIST_RE.search(text):
        return "list"
    if _CODE_HINT_RE.search(text):
        return "code"
    # headings are resolved separately; here we only classify content blocks
    return "paragraph"

def to_html(bt: BlockType, text: str) -> str:
    esc = escape(text)
    if bt == "list":
        return f"<li>{esc}</li>"
    if bt == "code":
        return f"<pre><code>{esc}</code></pre>"
    if bt == "paragraph":
        return f"<p>{esc}</p>"
    if bt == "heading":
        return f"<h3>{esc}</h3>"
    return esc

# --------------------------- Main ingestion ---------------------------

def ingest_pdf_to_jsonl(pdf_path: Path, out_jsonl: Path, tables_dir: Path) -> None:
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem

    # A) Build per-page lines and detect TOC pages
    lines_by_page: Dict[int, List[Line]] = {}
    tocs: Dict[int, bool] = {}
    page_meds: Dict[int, float] = {}

    for p in tqdm(range(len(doc)), desc=f"[{doc_id}] Pass A: read pages", leave=False):
        page_idx0 = p
        lines = extract_lines(doc, page_idx0)
        page_no = page_idx0 + 1
        tocs[page_no] = is_toc_page([ln.text for ln in lines])
        lines_by_page[page_no] = lines
        page_meds[page_no] = page_median_size(lines)

    # B) Heading candidates from outline + text
    outline_cands = outline_to_candidates(doc)
    text_cands: List[HeadingCand] = []
    for page_no, lines in tqdm(lines_by_page.items(), desc=f"[{doc_id}] Pass A: heading candidates", leave=False):
        if tocs.get(page_no, False):
            continue
        text_cands.extend(text_to_candidates(lines, allow_headings=True, med_size=page_meds[page_no]))

    # Reconcile to a global, healed heading sequence
    headings_all = reconcile_headings([*outline_cands, *text_cands])
    heads_by_page = index_headings_by_page(headings_all)

    # C) Build a position->section map and capture section intros
    pos2section = assign_section_for_blocks(lines_by_page, heads_by_page)
    section_intro_map: Dict[str, str] = {}

    # D) Emit blocks with context; tables too
    ts = ts_iso
    offset = 0
    with out_jsonl.open("a", encoding="utf-8") as fo:
        for p in tqdm(range(len(doc)), desc=f"[{doc_id}] Pass B: emit blocks", leave=False):
            page_no = p + 1
            lines = lines_by_page.get(page_no, [])
            if tocs.get(page_no, False):
                # still allow plain paragraphs if needed? we skip headings and tables here
                pass

            # Build typed content stream for this page (no headings here; headings only inform section state)
            typed: List[Tuple[BlockType, str, float]] = []  # (type, text, y0)
            for ln in lines:
                bt = classify_block(ln.text, ln.x0)
                typed.append((bt, ln.text, ln.y0))

            # Tables for page (skip on TOC pages)
            table_items = [] if tocs.get(page_no, False) else extract_tables_pdfplumber(pdf_path, page_no - 1, doc_id, tables_dir)

            # Walk through typed blocks, assign section, manage intro capture, emit records
            for i, (bt, text, y0) in enumerate(typed):
                sec_state = pos2section.get((page_no, y0), SectionState(tokens=[], label="", intro_captured=False))
                tokens = sec_state.tokens
                sec_label = sec_state.label
                sec_path = section_key(tokens) if tokens else ""

                # capture section intro: first paragraph under a section boundary
                # heuristic: if this line is the first content after a heading on this page,
                # the pos2section graph will have just updated tokens. We memo intro per sec_path.
                intro = section_intro_map.get(sec_path, "")
                if not intro and bt == "paragraph" and sec_path:
                    # Only capture if this is likely near the top (first few content lines after heading)
                    section_intro_map[sec_path] = text
                    intro = text

                prev_text = typed[i - 1][1] if i > 0 else ""
                next_text = typed[i + 1][1] if i + 1 < len(typed) else ""
                context_before = norm_ws(prev_text)
                context_after = norm_ws(next_text)

                html = to_html(bt, text)
                span_start, span_end = offset, offset + len(text) + 1
                offset = span_end

                rec = BlockRecord(
                    doc_id=doc_id,
                    page=page_no,
                    block_id=f"{doc_id}_p{page_no:03d}_b{i:03d}",
                    block_type=bt,
                    section_path=sec_path,
                    section_label=sec_label,
                    section_intro=section_intro_map.get(sec_path, ""),
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
                    ts_extracted=ts,
                    source_path=str(pdf_path),
                )
                fo.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

            # Emit tables at the end of the page with the current section (last line’s section on page)
            if table_items:
                # Determine section state near end of page
                if lines:
                    last_ln = lines[-1]
                    sec_state_last = pos2section.get((page_no, last_ln.y0), SectionState(tokens=[], label="", intro_captured=False))
                    sec_path_last = section_key(sec_state_last.tokens) if sec_state_last.tokens else ""
                    sec_label_last = sec_state_last.label
                    sec_intro_last = section_intro_map.get(sec_path_last, "")
                else:
                    sec_path_last = ""
                    sec_label_last = ""
                    sec_intro_last = ""
                for t_item in table_items:
                    span_start, span_end = offset, offset + len(t_item.markdown)
                    offset = span_end
                    trec = BlockRecord(
                        doc_id=doc_id,
                        page=page_no,
                        block_id=t_item.table_id,
                        block_type="table",
                        section_path=sec_path_last,
                        section_label=sec_label_last,
                        section_intro=sec_intro_last,
                        html=t_item.markdown,
                        markdown=t_item.markdown,
                        text=t_item.markdown,
                        table_id=t_item.table_id,
                        table_markdown=t_item.markdown,
                        table_csv_path=str(t_item.csv_path),
                        context_before="",
                        context_after="",
                        span_start=span_start,
                        span_end=span_end,
                        ts_extracted=ts,
                        source_path=str(pdf_path),
                    )
                    fo.write(json.dumps(asdict(trec), ensure_ascii=False) + "\n")

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Accurate PDF ingestion for technical manuals (sections + healing + context).")
    ap.add_argument("--pdf_glob", required=True, help='Path or glob (e.g., "/data/*.pdf")')
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--tables_dir", default="tables", help="Directory for extracted CSV tables")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    tables_dir = Path(args.tables_dir)
    # Expand glob manually to support absolute or single-file paths
    if any(ch in args.pdf_glob for ch in "*?[]"):
        pdf_paths = sorted(Path().glob(args.pdf_glob))
    else:
        pdf_paths = [Path(args.pdf_glob)]
    for pdf in tqdm(pdf_paths, desc="[Ingest PDFs]"):
        if pdf.suffix.lower() != ".pdf":
            continue
        ingest_pdf_to_jsonl(pdf, out_jsonl, tables_dir)

if __name__ == "__main__":
    main()
