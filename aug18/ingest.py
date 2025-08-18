#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Set

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

# --------------------------- Config ---------------------------

# Target chunk sizes (characters)
TARGET_CHARS = 900
MAX_CHARS = 1400
MIN_CHARS = 400

# Header/footer detection
TOP_MARGIN_PCT = 0.09
BOTTOM_MARGIN_PCT = 0.09
BANNER_FREQ_THRESHOLD = 0.30  # >=30% pages

# Regex patterns to drop anywhere (even mid-page)
BAN_PATTERNS = [
    r"\b(?:Export(?:\s+and)?\s+Distribution\s+Controlled)\b.*",
    r"\bSee\s+Cover\s+Page\b.*",
    r"\bDRAFT\b",
    r"\bREV(?:ISION)?\s*[A-Z0-9.\-]+\b",
    r"\bS\d+(?:\.\d+)?R\d+\b",           # e.g., S8.00R3
    r"\bTI\s*\d{3,}\.\d{3,}\b",          # e.g., TI 6191.408
    r"\b(?:Release|Revision)\s*[0-9A-Za-z.\-]+\b",
]

_MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_LIKE = re.compile(
    rf"^(?:\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}|{_MONTHS}\s+\d{{1,2}},\s*\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}/\d{{2}}/\d{{4}})$",
    re.IGNORECASE,
)

# --------------------------- Types & Records ---------------------------

BlockType = Literal["heading", "paragraph", "list", "code", "table"]

@dataclass(frozen=True)
class Line:
    text: str
    x0: float
    y0: float
    size: float
    page: int

@dataclass(frozen=True)
class HeadingCand:
    page: int
    y0: float
    parts: List[str]
    label: str
    title: str
    size: float
    x0: float
    source: str  # "outline" or "text"

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
    # page-level markdown pointers
    page_md_path: Optional[str]
    page_md_hash: Optional[str]
    page_md_excerpt: Optional[str]
    page_tokens_est: Optional[int]

# --------------------------- Regexes ---------------------------

_WS_RE = re.compile(r"\s+")
_TOC_HDR_RE = re.compile(r"\bTABLE OF CONTENTS\b", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+){0,6}\s+.+?\.{3,}\s*\d+\s*$")

NUM_HEADING_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,6})[)\s\-:]+(?P<title>\S.+)$")
_LIST_RE = re.compile(r"^(\- |\* |\d+[\.)] )")
_CODE_HINT_RE = re.compile(r"[{};]|^(code|snippet|example)$", re.IGNORECASE)
BAN_RE_LIST = [re.compile(pat, re.IGNORECASE) for pat in BAN_PATTERNS]

def norm_ws(s: Optional[str]) -> str:
    return _WS_RE.sub(" ", s or "").strip()

def is_date_like_line(text: str) -> bool:
    return bool(DATE_LIKE.match(text.strip()))

def is_toc_page(lines: Sequence[str]) -> bool:
    head = " ".join(lines[:12])
    if _TOC_HDR_RE.search(head):
        return True
    dotted = sum(1 for ln in lines[:300] if _TOC_LINE_RE.match(ln))
    return dotted >= 3

def parse_numbered_heading(text: str) -> Optional[Tuple[List[str], str, str]]:
    if is_date_like_line(text):
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

def extract_lines(doc: fitz.Document, page_index: int) -> Tuple[List[Line], float]:
    page = doc[page_index]
    height = float(page.rect.height)
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
            lines.append(Line(text=text, x0=x0, y0=y0, size=avg_size, page=page_index + 1))
    lines.sort(key=lambda L: (L.y0, L.x0))
    return lines, height

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
            csv_path.write_text("\n".join([",".join([norm_ws(c) for c in row]) for row in rows]), encoding="utf-8")
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
        toc = doc.get_toc(simple=True)
    except Exception:
        toc = []
    for lv, title, page1 in toc or []:
        if page1 <= 0:
            continue
        parsed = parse_numbered_heading(title)
        if not parsed:
            continue
        parts, title_txt, label = parsed
        cands.append(HeadingCand(page=page1, y0=0.0, parts=parts, label=label, title=title_txt, size=999.0, x0=0.0, source="outline"))
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
        if ln.size + 0.01 < med_size:
            continue
        out.append(HeadingCand(page=ln.page, y0=ln.y0, parts=parts, label=label, title=title, size=ln.size, x0=ln.x0, source="text"))
    return out

def reconcile_headings(cands: List[HeadingCand]) -> List[HeadingCand]:
    cands_sorted = sorted(cands, key=lambda c: (c.page, c.y0, -c.size))
    accepted: List[HeadingCand] = []
    chain: List[str] = []
    last_major: Optional[int] = None

    for c in cands_sorted:
        parts = c.parts
        is_major = len(parts) == 1 and parts[0].isdigit()
        if is_major:
            major = int(parts[0])
            if last_major is not None and major < last_major:
                continue
            accepted.append(c)
            chain = parts[:]
            last_major = major
            continue

        if not chain or parts[0] != chain[0]:
            if parts[0].isdigit():
                accepted.append(c)
                chain = parts[:]
                if len(parts) == 1:
                    last_major = int(parts[0])
            continue

        prefix_ok = True
        for i in range(min(len(chain), len(parts) - 1)):
            if chain[i] != parts[i]:
                prefix_ok = False
                break
        if not prefix_ok:
            accepted.append(c)
            chain = parts[:]
            continue

        accepted.append(c)
        chain = parts[:]

    return accepted

def index_headings_by_page(headings: List[HeadingCand]) -> Dict[int, List[HeadingCand]]:
    byp: Dict[int, List[HeadingCand]] = {}
    for h in headings:
        byp.setdefault(h.page, []).append(h)
    for p in byp:
        byp[p].sort(key=lambda h: h.y0)
    return byp

# --------------------------- Banner (header/footer) Filtering ---------------------------

def normalized_banner_key(text: str) -> str:
    t = norm_ws(text)
    t = re.sub(r"\bpage\s*\d+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+\s*/\s*\d+\b", "", t)  # "12 / 250"
    return t.strip().lower()

def collect_banners(lines_by_page: Dict[int, List[Line]], page_heights: Dict[int, float]) -> Dict[str, str]:
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}
    n_pages = len(lines_by_page)

    for page, lines in lines_by_page.items():
        h = page_heights[page]
        top_y = TOP_MARGIN_PCT * h
        bot_y = (1.0 - BOTTOM_MARGIN_PCT) * h

        seen_top: set = set()
        seen_bot: set = set()
        for ln in lines:
            key = normalized_banner_key(ln.text)
            if not key:
                continue
            if ln.y0 <= top_y:
                if key not in seen_top:
                    top_counts[key] = top_counts.get(key, 0) + 1
                    seen_top.add(key)
            elif ln.y0 >= bot_y:
                if key not in seen_bot:
                    bot_counts[key] = bot_counts.get(key, 0) + 1
                    seen_bot.add(key)

    banners: Dict[str, str] = {}
    thresh = max(1, int(BANNER_FREQ_THRESHOLD * n_pages))
    for k, c in top_counts.items():
        if c >= thresh:
            banners[k] = "top"
    for k, c in bot_counts.items():
        if c >= thresh:
            banners[k] = "bottom"
    return banners

def is_banner_line(ln: Line, page_height: float, banners: Dict[str, str]) -> bool:
    top_y = TOP_MARGIN_PCT * page_height
    bot_y = (1.0 - BOTTOM_MARGIN_PCT) * page_height
    key = normalized_banner_key(ln.text)
    if not key:
        return False
    if key in banners:
        if (banners[key] == "top" and ln.y0 <= top_y) or (banners[key] == "bottom" and ln.y0 >= bot_y):
            return True
    for pat in BAN_RE_LIST:
        if pat.search(ln.text):
            return True
    if is_date_like_line(ln.text) and (ln.y0 <= top_y or ln.y0 >= bot_y):
        return True
    return False

# --------------------------- Section mapping ---------------------------

@dataclass
class SectionState:
    tokens: List[str]
    label: str

def section_key(tokens: List[str], max_depth: int = 3) -> str:
    if not tokens:
        return ""
    return " > ".join([".".join(tokens[:i+1]) for i in range(min(len(tokens), max_depth))])

def assign_section_for_lines(
    lines_by_page: Dict[int, List[Line]],
    headings_by_page: Dict[int, List[HeadingCand]],
) -> Dict[Tuple[int, float], SectionState]:
    mapping: Dict[Tuple[int, float], SectionState] = {}
    active_tokens: List[str] = []
    active_label: str = ""
    for page in sorted(lines_by_page.keys()):
        heads = headings_by_page.get(page, [])
        line_list = lines_by_page[page]
        hi = 0
        for ln in line_list:
            while hi < len(heads) and heads[hi].y0 <= ln.y0 + 0.01:
                h = heads[hi]
                active_tokens = h.parts[:]
                active_label = h.label
                hi += 1
            mapping[(page, ln.y0)] = SectionState(tokens=active_tokens[:], label=active_label)
    return mapping

# --------------------------- Helper to identify heading lines ---------------------------

def get_heading_lines_set(headings_by_page: Dict[int, List[HeadingCand]]) -> Set[Tuple[int, str]]:
    """
    Create a set of (page, normalized_text) tuples for all heading lines
    to efficiently check if a line is a heading.
    """
    heading_lines = set()
    for page, heads in headings_by_page.items():
        for h in heads:
            # The heading label contains the full heading text
            normalized = norm_ws(h.label)
            heading_lines.add((page, normalized))
            # Also add just the title part in case it appears separately
            normalized_title = norm_ws(h.title)
            heading_lines.add((page, normalized_title))
    return heading_lines

# --------------------------- Typing & Rendering ---------------------------

def classify_block(text: str, x0: float) -> BlockType:
    if _LIST_RE.search(text):
        return "list"
    if _CODE_HINT_RE.search(text):
        return "code"
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

# --------------------------- Chunk packer ---------------------------

@dataclass
class TypedLine:
    bt: BlockType
    text: str
    y0: float
    section_path: str
    section_label: str

def pack_chunks(typed_lines: List[TypedLine]) -> List[Tuple[BlockType, str, str, str]]:
    out: List[Tuple[BlockType, str, str, str]] = []
    buf_texts: List[str] = []
    buf_len = 0
    buf_bt: Optional[BlockType] = None
    buf_sec_path = ""
    buf_sec_label = ""

    def flush():
        nonlocal buf_texts, buf_len, buf_bt, buf_sec_path, buf_sec_label
        if buf_texts:
            text = " ".join(buf_texts)
            out.append((buf_bt or "paragraph", text, buf_sec_path, buf_sec_label))
        buf_texts = []
        buf_len = 0
        buf_bt = None
        buf_sec_path = ""
        buf_sec_label = ""

    for tl in typed_lines:
        if tl.bt == "table":
            flush()
            continue
        if buf_texts and tl.section_path != buf_sec_path:
            flush()
        if not buf_texts:
            buf_bt = tl.bt
            buf_sec_path = tl.section_path
            buf_sec_label = tl.section_label
            buf_texts = [tl.text]
            buf_len = len(tl.text)
            continue
        if tl.bt == buf_bt and (buf_len < TARGET_CHARS or (buf_len < MAX_CHARS and not tl.text.endswith((".", ":", ";")))):
            buf_texts.append(tl.text)
            buf_len += len(tl.text) + 1
            if buf_len >= MAX_CHARS:
                flush()
            continue
        flush()
        buf_bt = tl.bt
        buf_sec_path = tl.section_path
        buf_sec_label = tl.section_label
        buf_texts = [tl.text]
        buf_len = len(tl.text)
    flush()

    glued: List[Tuple[BlockType, str, str, str]] = []
    for item in out:
        if not glued:
            glued.append(item)
            continue
        bt, tx, sp, sl = item
        pbt, ptx, psp, psl = glued[-1]
        if sp == psp and len(ptx) < MIN_CHARS:
            glued[-1] = (pbt, ptx + " " + tx, psp, psl)
        else:
            glued.append(item)
    return glued

# --------------------------- Page Markdown ---------------------------

def render_page_markdown(
    page_no: int,
    page_height: float,
    lines: List[Line],
    heads: List[HeadingCand],
    tables: List[TableItem],
    banners: Dict[str, str],
) -> str:
    """Produce a simple, cleaned Markdown representation of the page."""
    # Filter out banner lines & banned patterns
    flines = [ln for ln in lines if not is_banner_line(ln, page_height, banners) and ln.text.strip()]
    # Merge headings with body stream using y0 ordering (headings first if y0 tie)
    events: List[Tuple[float, str]] = []
    for h in heads:
        events.append((h.y0, f"### {h.label}"))
    for ln in flines:
        if _LIST_RE.search(ln.text):
            events.append((ln.y0, f"- {ln.text.lstrip('- ').lstrip('* ').strip()}"))
        elif _CODE_HINT_RE.search(ln.text):
            events.append((ln.y0, f"```\n{ln.text}\n```"))
        else:
            events.append((ln.y0, ln.text))
    events.sort(key=lambda x: x[0])

    # Add table stubs at end (or could inject by approximate y if available from pdfplumber; we keep simple)
    md_lines: List[str] = []
    last_was_para = False
    for _, txt in events:
        if txt.startswith("### "):
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            md_lines.append(txt)
            md_lines.append("")
            last_was_para = False
        elif txt.startswith("- "):
            md_lines.append(txt)
            last_was_para = False
        elif txt.startswith("```"):
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            md_lines.append(txt)
            md_lines.append("")
            last_was_para = False
        else:
            # paragraph
            if last_was_para:
                md_lines[-1] = md_lines[-1] + " " + txt
            else:
                if md_lines and md_lines[-1] != "":
                    md_lines.append("")
                md_lines.append(txt)
                last_was_para = True

    if tables:
        if md_lines and md_lines[-1] != "":
            md_lines.append("")
        for t in tables:
            md_lines.append(f"> [Table: {t.table_id}] (CSV: {t.csv_path})")

    # Ensure trailing newline
    if not md_lines or md_lines[-1] != "":
        md_lines.append("")
    return "\n".join(md_lines)

def quick_token_estimate(text: str) -> int:
    # crude but stable estimator for auditing/packing decisions
    return max(1, len(text.split()))

# --------------------------- Main ingestion ---------------------------

def ingest_pdf_to_jsonl(pdf_path: Path, out_jsonl: Path, tables_dir: Path, emit_page_md: bool, pages_dir: Path) -> None:
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem

    # Pass A: read pages
    lines_by_page: Dict[int, List[Line]] = {}
    page_heights: Dict[int, float] = {}
    tocs: Dict[int, bool] = {}
    page_meds: Dict[int, float] = {}

    for p in tqdm(range(len(doc)), desc=f"[{doc_id}] Pass A: read pages", leave=False):
        lines, height = extract_lines(doc, p)
        page_no = p + 1
        lines_by_page[page_no] = lines
        page_heights[page_no] = height
        tocs[page_no] = is_toc_page([ln.text for ln in lines])
        page_meds[page_no] = page_median_size(lines)

    # A2: banners
    banners = collect_banners(lines_by_page, page_heights)

    # Headings
    outline_cands = outline_to_candidates(doc)
    text_cands: List[HeadingCand] = []
    for page_no, lines in tqdm(lines_by_page.items(), desc=f"[{doc_id}] Pass A: heading candidates", leave=False):
        if tocs.get(page_no, False):
            continue
        filt = [ln for ln in lines if not is_banner_line(ln, page_heights[page_no], banners)]
        text_cands.extend(text_to_candidates(filt, allow_headings=True, med_size=page_meds[page_no]))
    headings_all = reconcile_headings([*outline_cands, *text_cands])
    heads_by_page = index_headings_by_page(headings_all)
    pos2section = assign_section_for_lines(lines_by_page, heads_by_page)
    
    # Create set of heading lines for efficient lookup
    heading_lines_set = get_heading_lines_set(heads_by_page)
    
    section_intro_map: Dict[str, str] = {}

    # Emit
    offset = 0
    with out_jsonl.open("a", encoding="utf-8") as fo:
        for p in tqdm(range(len(doc)), desc=f"[{doc_id}] Pass B: emit", leave=False):
            page_no = p + 1
            lines = lines_by_page.get(page_no, [])
            if not lines:
                continue

            # page-level tables (single call)
            table_items = [] if tocs.get(page_no, False) else extract_tables_pdfplumber(pdf_path, page_no - 1, doc_id, tables_dir)

            # ---- Page Markdown (cleaned) ----
            page_md_path: Optional[str] = None
            page_md_hash: Optional[str] = None
            page_md_excerpt: Optional[str] = None
            page_tokens_est: Optional[int] = None
            if emit_page_md:
                pages_dir.mkdir(parents=True, exist_ok=True)
                page_md_text = render_page_markdown(
                    page_no=page_no,
                    page_height=page_heights[page_no],
                    lines=lines,
                    heads=heads_by_page.get(page_no, []),
                    tables=table_items,
                    banners=banners,
                )
                # write file
                md_file = pages_dir / f"{doc_id}_p{page_no:03d}.md"
                md_file.write_text(page_md_text, encoding="utf-8")
                page_md_path = str(md_file)
                page_md_hash = hashlib.sha1(page_md_text.encode("utf-8")).hexdigest()
                page_md_excerpt = page_md_text.strip().replace("\n", " ")[:400]
                page_tokens_est = quick_token_estimate(page_md_text)

            # Filter lines for chunking
            flines: List[Line] = []
            for ln in lines:
                if is_banner_line(ln, page_heights[page_no], banners):
                    continue
                t = ln.text.strip()
                if not t:
                    continue
                # Check if this line is a heading line - if so, skip it
                normalized_text = norm_ws(ln.text)
                if (page_no, normalized_text) in heading_lines_set:
                    continue
                # Also check if it matches the heading pattern (backup check)
                if parse_numbered_heading(ln.text):
                    continue
                flines.append(ln)

            # Build typed lines (now without heading lines)
            typed_lines: List[TypedLine] = []
            for ln in flines:
                sec = pos2section.get((page_no, ln.y0), SectionState(tokens=[], label=""))
                sec_path = section_key(sec.tokens) if sec.tokens else ""
                bt = classify_block(ln.text, ln.x0)
                typed_lines.append(TypedLine(bt=bt, text=ln.text, y0=ln.y0, section_path=sec_path, section_label=sec.label))

            # Capture section intros - now we're guaranteed to get the first real paragraph
            # after the heading, not the heading itself
            for tl in typed_lines:
                if tl.bt == "paragraph" and tl.section_path and tl.section_path not in section_intro_map:
                    # This is the first paragraph in this section
                    section_intro_map[tl.section_path] = tl.text

            # Pack
            packed = pack_chunks(typed_lines)

            def ctx(i: int) -> Tuple[str, str]:
                before = packed[i - 1][1] if i > 0 else ""
                after = packed[i + 1][1] if i + 1 < len(packed) else ""
                return norm_ws(before), norm_ws(after)

            # Emit text chunks
            for i, (bt, text, sec_path, sec_label) in enumerate(packed):
                section_intro = section_intro_map.get(sec_path, "")
                html = to_html(bt, text)
                span_start, span_end = offset, offset + len(text) + 1
                offset = span_end
                cb, ca = ctx(i)
                rec = BlockRecord(
                    doc_id=doc_id,
                    page=page_no,
                    block_id=f"{doc_id}_p{page_no:03d}_c{i:03d}",
                    block_type=bt,
                    section_path=sec_path,
                    section_label=sec_label,
                    section_intro=section_intro,
                    html=html,
                    markdown=text,
                    text=text,
                    table_id=None,
                    table_markdown=None,
                    table_csv_path=None,
                    context_before=cb,
                    context_after=ca,
                    span_start=span_start,
                    span_end=span_end,
                    ts_extracted=ts_iso,
                    source_path=str(pdf_path),
                    page_md_path=page_md_path,
                    page_md_hash=page_md_hash,
                    page_md_excerpt=page_md_excerpt,
                    page_tokens_est=page_tokens_est,
                )
                fo.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

            # Emit tables under last section of page
            if table_items:
                sec_path_last = packed[-1][2] if packed else ""
                sec_label_last = packed[-1][3] if packed else ""
                sec_intro_last = section_intro_map.get(sec_path_last, "")
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
                        ts_extracted=ts_iso,
                        source_path=str(pdf_path),
                        page_md_path=page_md_path,
                        page_md_hash=page_md_hash,
                        page_md_excerpt=page_md_excerpt,
                        page_tokens_est=page_tokens_est,
                    )
                    fo.write(json.dumps(asdict(trec), ensure_ascii=False) + "\n")

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Accurate PDF ingestion (healed sections, banner filtering, larger chunks, page MD).")
    ap.add_argument("--pdf_glob", required=True, help='Path or glob (e.g., "/data/*.pdf")')
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--tables_dir", default="tables", help="Directory for extracted CSV tables")
    ap.add_argument("--pages_dir", default="pages", help="Directory to write per-page Markdown")
    ap.add_argument("--no_page_md", action="store_true", help="Disable writing per-page Markdown")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    tables_dir = Path(args.tables_dir)
    pages_dir = Path(args.pages_dir)
    emit_page_md = not args.no_page_md

    if any(ch in args.pdf_glob for ch in "*?[]"):
        pdf_paths = sorted(Path().glob(args.pdf_glob))
    else:
        pdf_paths = [Path(args.pdf_glob)]
    for pdf in tqdm(pdf_paths, desc="[Ingest PDFs]"):
        if pdf.suffix.lower() != ".pdf":
            continue
        ingest_pdf_to_jsonl(pdf, out_jsonl, tables_dir, emit_page_md, pages_dir)

if __name__ == "__main__":
    main()
