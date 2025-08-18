# ingest.py
# Air-gapped PDF -> structured chunks with strict Section Intro (first paragraph after header).
# Dependencies: pdfplumber, scikit-learn (for optional utilities), rapidfuzz (optional), python>=3.9

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import pdfplumber
import re
import json
import hashlib
import argparse
from pathlib import Path

# --------------------------
# Data models
# --------------------------

@dataclass
class Block:
    page: int
    y0: float
    x0: float
    kind: str          # "heading" | "paragraph" | "list" | "table"
    text: str = ""
    level: int = 0     # heading level if kind=="heading"
    table_markdown: str = ""
    table_summary: str = ""
    section_path: str = ""
    section_label: str = ""

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    section_path: str
    section_label: str
    section_intro: str
    block_type: str
    text: str
    table_markdown: str
    table_summary: str
    context_before: str
    context_after: str
    neighbors: List[str]
    hidden_terms: List[str]  # acronym expansions etc.

# --------------------------
# Heuristics
# --------------------------

HEADER_RE = re.compile(r"^\s*(\d+(\.\d+)*)?\s*[A-Z][^\n]{0,120}$")
NUM_PREFIX_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+")
ACRO1 = re.compile(r"\b([A-Z]{2,})\s*\(([^)]+)\)")              # e.g., ASR (airport surveillance radar)
ACRO2 = re.compile(r"\b([A-Za-z][A-Za-z\s-]{2,})\s*\(([A-Z]{2,})\)")  # e.g., airport surveillance radar (ASR)
LIST_BULLET_RE = re.compile(r"^\s*(?:[-*•]\s+|\(\w\)\s+)")

def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if HEADER_RE.match(s):
        return True
    # Title-ish: short, many capitals or words are Title Case
    words = s.split()
    caps_ratio = sum(1 for c in s if c.isupper()) / max(1, sum(1 for c in s if c.isalpha()))
    title_case = sum(1 for w in words if w[:1].isupper()) / max(1, len(words))
    return len(words) <= 15 and (caps_ratio > 0.5 or title_case > 0.8)

def heading_level(line: str) -> int:
    m = NUM_PREFIX_RE.match(line.strip())
    if m:
        return min(6, m.group(1).count(".") + 1)
    # fallback: infer by length/case
    s = line.strip()
    if s.isupper():
        return 1
    if len(s.split()) <= 4:
        return 2
    return 3

def classify_block(line: str) -> str:
    if is_heading(line):
        return "heading"
    if LIST_BULLET_RE.match(line):
        return "list"
    return "paragraph"

def md_table_from_plumber(table: List[List[str]]) -> str:
    if not table or not table[0]:
        return ""
    # Normalize cells
    rows = [[(c or "").strip() for c in row] for row in table]
    header = rows[0]
    widths = [max(len((row[i] if i < len(row) else "")) for row in rows[:50]) for i in range(len(header))]
    def fmt_row(row: List[str]) -> str:
        cells = [(row[i] if i < len(row) else "") for i in range(len(header))]
        return "| " + " | ".join(cells) + " |"
    out = [fmt_row(header)]
    out.append("| " + " | ".join("-" * max(3, w) for w in widths) + " |")
    for r in rows[1:12]:  # keep modest size
        out.append(fmt_row(r))
    return "\n".join(out)

def summarize_table_md(md: str) -> str:
    if not md:
        return ""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    cols = [c.strip() for c in header.strip("|").split("|")] if header.startswith("|") else []
    cols = [c for c in cols if c]
    if cols:
        head = ", ".join(cols[:8])
        return f"Table with columns: {head}."
    return "Technical table with structured fields."

def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def normalize_text(text: str) -> str:
    t = text.replace("\u00ad", "")       # soft hyphen
    t = dehyphenate(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()

def content_hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()

# --------------------------
# Acronyms
# --------------------------

def build_acronym_map(all_text: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for short, longf in ACRO1.findall(all_text):
        m[short.strip()] = longf.strip()
    for longf, short in ACRO2.findall(all_text):
        m[short.strip()] = longf.strip()
    return {k: v for k, v in m.items() if 2 < len(k) <= 12 and len(v) <= 120}

# --------------------------
# PDF -> Blocks
# --------------------------

def extract_blocks(pdf_path: str) -> List[Block]:
    blocks: List[Block] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False) or []
            lines_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            for w in words:
                key = (int(round(w["top"])), int(round(w["x0"])))
                lines_map.setdefault(key, []).append(w)
            lines: List[Tuple[float, float, str]] = []
            for (top, x0), wlist in sorted(lines_map.items()):
                text = " ".join(w["text"] for w in sorted(wlist, key=lambda x: x["x0"]))
                lines.append((float(top), float(x0), normalize_text(text)))
            # tables
            md_tables: List[Tuple[float, str]] = []
            try:
                tables = page.extract_tables() or []
                for t in tables:
                    md = md_table_from_plumber(t)
                    if md:
                        top_guess = float(lines[0][0]) if lines else 0.0
                        md_tables.append((top_guess, md))
            except Exception:
                pass
            # classify into blocks
            for (y0, x0, text) in lines:
                if not text:
                    continue
                kind = classify_block(text)
                lvl = heading_level(text) if kind == "heading" else 0
                blocks.append(Block(page=pno, y0=y0, x0=x0, kind=kind, text=text, level=lvl))
            for (y0, md) in md_tables:
                blocks.append(Block(page=pno, y0=y0, x0=0.0, kind="table", table_markdown=md))
    # sort
    blocks.sort(key=lambda b: (b.page, b.y0, b.x0, 0 if b.kind == "heading" else 1))
    return blocks

# --------------------------
# Section assignment & intros
# --------------------------

def assign_sections(blocks: List[Block]) -> None:
    """Assign section_path and label from the running heading stack."""
    stack: List[Block] = []
    for b in blocks:
        if b.kind == "heading":
            while stack and stack[-1].level >= b.level:
                stack.pop()
            stack.append(b)
        label = " > ".join(x.text.strip() for x in stack)
        b.section_label = stack[-1].text.strip() if stack else ""
        b.section_path = label

def compute_strict_section_intros(blocks: List[Block]) -> Dict[str, str]:
    """First block immediately after the header on the same page; must be a paragraph."""
    intros: Dict[str, str] = {}
    last_heading_by_page_path: Dict[Tuple[int, str], float] = {}
    for b in blocks:
        if b.kind == "heading" and b.section_path:
            last_heading_by_page_path[(b.page, b.section_path)] = b.y0
    seen_first_after: Dict[Tuple[int, str], Block] = {}
    for b in blocks:
        key = (b.page, b.section_path)
        if key not in last_heading_by_page_path:
            continue
        if b.y0 <= last_heading_by_page_path[key] + 0.01:
            continue
        if key in seen_first_after:
            continue
        seen_first_after[key] = b
    for (page, sp), first in seen_first_after.items():
        if first.kind == "paragraph":
            intros[sp] = first.text.strip()
        else:
            intros.setdefault(sp, "")
    return intros

# --------------------------
# Table summaries
# --------------------------

def attach_table_summaries(blocks: List[Block]) -> None:
    for b in blocks:
        if b.kind != "table":
            continue
        if not b.table_markdown:
            b.table_summary = ""
            continue
        b.table_summary = summarize_table_md(b.table_markdown)

# --------------------------
# Chunking
# --------------------------

def neighbors_by_section(blocks: List[Block]) -> Dict[str, List[str]]:
    sections = [b.section_label for b in blocks if b.kind == "heading" and b.section_label]
    out: Dict[str, List[str]] = {}
    for i, lbl in enumerate(sections):
        prev_lbl = sections[i - 1] if i > 0 else ""
        next_lbl = sections[i + 1] if i + 1 < len(sections) else ""
        out[lbl] = [p for p in (prev_lbl, next_lbl) if p]
    return out

def visible_text_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text))

def make_chunks(pdf_path: str, doc_id: Optional[str] = None) -> List[Chunk]:
    doc_id = doc_id or Path(pdf_path).stem
    blocks = extract_blocks(pdf_path)
    assign_sections(blocks)
    attach_table_summaries(blocks)
    intros = compute_strict_section_intros(blocks)
    # acronym map from the whole document
    all_text = "\n".join(b.text for b in blocks if b.text) + "\n" + "\n".join(b.table_markdown for b in blocks if b.table_markdown)
    acro_map = build_acronym_map(all_text)
    neigh = neighbors_by_section(blocks)

    chunks: List[Chunk] = []
    for i, b in enumerate(blocks):
        if b.kind == "paragraph":
            if len(b.text.strip()) < 25 or visible_text_ratio(b.text) < 0.35:
                continue
        section_intro = intros.get(b.section_path, "")
        before = blocks[i - 1].text.strip() if i > 0 and blocks[i - 1].kind == "paragraph" else ""
        after = blocks[i + 1].text.strip() if i + 1 < len(blocks) and blocks[i + 1].kind == "paragraph" else ""
        text = b.text.strip() if b.kind != "table" else b.table_summary
        chunk = Chunk(
            chunk_id=f"{doc_id}-{b.page}-{content_hash((b.section_path or '') + (text or '')[:120])}",
            doc_id=doc_id,
            page=b.page,
            section_path=b.section_path,
            section_label=b.section_label,
            section_intro=section_intro,
            block_type=b.kind,
            text=text,
            table_markdown=b.table_markdown,
            table_summary=b.table_summary,
            context_before=before,
            context_after=after,
            neighbors=neigh.get(b.section_label, []),
            hidden_terms=[v for v in acro_map.values()],
        )
        chunks.append(chunk)
    return chunks

# --------------------------
# CLI
# --------------------------

def write_jsonl(chunks: List[Chunk], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs into high-quality chunks.")
    ap.add_argument("pdf", help="Path to a PDF file")
    ap.add_argument("--out", default="", help="Output .jsonl (defaults to <pdf>.jsonl)")
    args = ap.parse_args()

    out = args.out or (str(Path(args.pdf).with_suffix(".jsonl")))
    chunks = make_chunks(args.pdf)
    write_jsonl(chunks, out)
    print(f"Wrote {len(chunks)} chunks -> {out}")

if __name__ == "__main__":
    main()
