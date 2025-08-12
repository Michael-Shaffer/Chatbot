#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from statistics import median
from typing import List, Literal, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

# ---------- Types ----------

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

@dataclass(frozen=True)
class Line:
    text: str
    x0: float
    y0: float
    size: float  # char-weighted average font size for the line

# ---------- Regexes & helpers ----------

_WS_RE = re.compile(r"\s+")
_TOC_HDR_RE = re.compile(r"\bTABLE OF CONTENTS\b", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+){0,6}\s+.+?\.{3,}\s*\d+\s*$")

# Allowed headings:
#   Numeric majors 1..6 with optional dotted numeric sublevels, e.g., 4.1.3
#   Appendices A..I with dotted numeric sublevels, e.g., C.2 or C.2.4
_NUM_MAJOR = r"(?:[1-6])"
_NUM_SUB   = r"(?:\.[0-9]{1,2}){0,6}"
_APP_MAJOR = r"(?:[A-I])"
_APP_SUB   = r"(?:\.[0-9]{1,2}){0,6}"

_HEADING_RE = re.compile(
    rf"^(?P<label>(?:{_NUM_MAJOR}{_NUM_SUB})|(?:{_APP_MAJOR}{_APP_SUB}))"
    r"[)\s\-–—:]+(?P<title>\S.+)$"
)

_MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE_LIKE = re.compile(
    rf"^(?:\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}|{_MONTHS}\s+\d{{1,2}},\s*\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}/\d{{2}}/\d{{4}})$",
    re.IGNORECASE,
)

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

def is_date_like(text: str) -> bool:
    return bool(_DATE_LIKE.match(text.strip()))

def parse_allowed_heading(text: str) -> Optional[Tuple[List[str], str, str]]:
    if is_date_like(text):
        return None
    m = _HEADING_RE.match(text.strip())
    if not m:
        return None
    label = m.group("label")
    title = m.group("title").strip()
    if not title:
        return None
    parts = label.split(".")  # tokens like ["4","1","3"] or ["C","2"]
    return parts, title, f"{label} {title}"

# ---------- Rendering ----------

def to_html(block_type: BlockType, text: str) -> str:
    esc = escape(text)
    if block_type == "heading":
        return f"<h3>{esc}</h3>"
    if block_type == "list":
        return f"<li>{esc}</li>"
    if block_type == "code":
        return f"<pre><code>{esc}</code></pre>"
    return f"<p>{esc}</p>"

def render_markdown_table(columns: List[str], rows: List[List[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        cells = [norm_ws(c) for c in r]
        cells += [""] * (len(columns) - len(cells))
        body.append("| " + " | ".join(cells[: len(columns)]) + " |")
    return "\n".join([header, sep, *body]) + "\n"

# ---------- Extraction ----------

def extract_lines(page: fitz.Page) -> List[Line]:
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
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            sizes = [(float(s.get("size", 0.0)), len(s.get("text", ""))) for s in spans]
            total = sum(w for _, w in sizes) or 1
            avg_size = sum(sz * w for sz, w in sizes) / total
            lines.append(Line(text=text, x0=float(x0), y0=float(y0), size=float(avg_size)))
    lines.sort(key=lambda L: (L.y0, L.x0))
    return lines

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

# ---------- Section tracker (strict, prefix-consistent, with healing) ----------

class SectionTracker:
    def __init__(self) -> None:
        self.tokens: List[str] = []     # numeric prefix chain, e.g., ["4","1","3"] or ["C","2"]
        self.labels: List[str] = []     # human labels (e.g., "4.1.3 Title")
        self.phase: Literal["numeric","appendix"] = "numeric"

    @staticmethod
    def _is_numeric(token: str) -> bool:
        return token.isdigit()

    def path(self, max_depth: int = 3) -> str:
        return " > ".join(self.labels[-max_depth:])

    def accept(self, parts: List[str], label: str, *, page_median_size: float, line_size: float, x0: float) -> bool:
        # cheap layout gate: headings are usually left-ish
        if x0 >= 120.0:
            return False

        major = parts[0]
        is_num = self._is_numeric(major)

        # phase control: once appendices begin, don't accept numeric majors later
        if self.phase == "numeric" and not is_num:
            self.phase = "appendix"
        if self.phase == "appendix" and is_num:
            return False

        # validate majors
        if is_num:
            if not (1 <= int(major) <= 6):
                return False
            # majors should be visually prominent
            if len(parts) == 1 and line_size + 0.01 < page_median_size:
                return False
        else:
            if not ("A" <= major <= "I"):
                return False

        # healing/prefix enforcement
        parent = parts[:-1]
        if len(parts) == 1:
            # start new chain
            self.tokens = [major]
            self.labels = [label]
            return True

        # different major than current? reset chain (heals missing parent)
        if not self.tokens or self.tokens[0] != major:
            self.tokens = parent[:] + [parts[-1]]
            # synthesize missing parent labels numerically ("4", "4.1", or "C", "C.1")
            synth = [".".join(parent[:i+1]) for i in range(len(parent))]
            self.labels = synth + [label]
            return True

        # ensure parent prefix exists; synthesize if needed
        k = 0
        while k < min(len(self.tokens), len(parent)) and self.tokens[k] == parent[k]:
            k += 1
        if k != len(parent):
            self.tokens = parent[:] + [parts[-1]]
            synth = [".".join(parent[:i+1]) for i in range(len(parent))]
            self.labels = synth + [label]
            return True

        # normal child advance
        self.tokens = parts[:]
        if len(self.labels) == len(parent):
            self.labels.append(label)
        else:
            self.labels = self.labels[:len(parent)] + [label]
        return True

# ---------- Page -> records ----------

def iter_page_records(
    pdf_path: Path,
    doc_id: str,
    page: fitz.Page,
    page_no: int,
    tracker: SectionTracker,
    offset: int,
    ts_iso: str,
    tables_dir: Path,
) -> Tuple[List[BlockRecord], int]:
    records: List[BlockRecord] = []
    lines = extract_lines(page)

    # Skip headings/tables on TOC-like pages
    toc = is_toc_page([ln.text for ln in lines])
    table_items: List[TableItem] = [] if toc else extract_tables_pdfplumber(pdf_path, page_no - 1, doc_id, tables_dir)

    sizes = [ln.size for ln in lines] or [10.0]
    page_med = median(sizes)

    # First pass: tag candidates (we'll finalize with tracker.accept)
    typed: List[Tuple[BlockType, str]] = []
    cands: List[Tuple[int, List[str], str, float, float]] = []  # (idx, parts, label, size, x0)
    for i, ln in enumerate(lines):
        if not toc:
            parsed = parse_allowed_heading(ln.text)
        else:
            parsed = None

        if parsed:
            parts, title, label = parsed
            cands.append((i, parts, label, ln.size, ln.x0))
            typed.append(("heading", ln.text))  # tentative; may be demoted
        elif _LIST_RE.search(ln.text):
            typed.append(("list", ln.text))
        elif _CODE_HINT_RE.search(ln.text):
            typed.append(("code", ln.text))
        else:
            typed.append(("paragraph", ln.text))

    # Second pass: strict acceptance; demote false headings
    cand_map = {idx: (parts, label, size, x0) for idx, parts, label, size, x0 in cands}
    for j, (bt, text) in enumerate(typed):
        if bt != "heading":
            continue
        parts, label, size, x0 = cand_map[j]
        if not tracker.accept(parts, label, page_median_size=page_med, line_size=size, x0=x0):
            typed[j] = ("paragraph", text)

    # Emit text blocks
    for bi, (bt, text) in enumerate(typed):
        section_path = tracker.path()
        html = to_html(bt, text)
        prev_text = typed[bi - 1][1] if bi > 0 else ""
        next_text = typed[bi + 1][1] if bi + 1 < len(typed) else ""
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

    # Emit table blocks
    for tbl in table_items:
        section_path = tracker.path()
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

# ---------- Orchestration ----------

def ingest_pdf_to_jsonl(pdf_path: Path, out_jsonl: Path, tables_dir: Path) -> None:
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem
    tracker = SectionTracker()
    offset = 0

    with out_jsonl.open("a", encoding="utf-8") as fo:
        for p in tqdm(range(len(doc)), desc=f"[{doc_id}] Extracting pages", leave=False):
            page_no = p + 1
            page = doc[p]
            records, offset = iter_page_records(pdf_path, doc_id, page, page_no, tracker, offset, ts_iso, tables_dir)
            for r in records:
                fo.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Typed PDF ingestion for local RAG (strict headings, TOC skip, progress).")
    ap.add_argument("--pdf_glob", required=True, help='e.g. "/data/*.pdf"')
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--tables_dir", default="tables", help="Directory for extracted CSV tables")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    tables_dir = Path(args.tables_dir)
    # glob manually to support absolute or single-file paths
    if any(ch in args.pdf_glob for ch in "*?[]"):
        pdf_paths = sorted(Path().glob(args.pdf_glob))
    else:
        pdf_paths = [Path(args.pdf_glob)]
    for pdf in tqdm(pdf_paths, desc="[Parsing PDFs]"):
        if pdf.suffix.lower() != ".pdf":
            continue
        ingest_pdf_to_jsonl(pdf, out_jsonl, tables_dir)

if __name__ == "__main__":
    main()