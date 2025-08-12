#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---------- Allowed labels and parsers ----------

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

def parse_heading_text(text: str) -> Optional[Tuple[List[str], str, str]]:
    if not text:
        return None
    t = text.strip()
    if _DATE_LIKE.match(t):
        return None
    m = _HEADING_RE.match(t)
    if not m:
        return None
    label = m.group("label")
    title = (m.group("title") or "").strip()
    if not title:
        return None
    parts = label.split(".")  # ["4","1","3"] or ["C","2"]
    return parts, title, f"{label} {title}"

def is_numeric(tok: str) -> bool:
    return tok.isdigit()

def valid_major(tok: str, phase: str) -> bool:
    if is_numeric(tok):
        return phase != "appendix" and 1 <= int(tok) <= 6
    return "A" <= tok <= "I"

# ---------- Data model ----------

@dataclass
class Rec:
    raw: Dict
    is_heading: bool
    parts: Optional[List[str]]
    label: Optional[str]
    doc_id: str
    page: int

# ---------- IO ----------

def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: Path, recs: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Repair Engine ----------

class SectionRepair:
    def __init__(self, mode: str = "heal") -> None:
        self.mode = mode  # "heal" or "demote"
        self.tokens: List[str] = []
        self.labels: List[str] = []
        self.phase: str = "numeric"
        self.stats = {
            "total": 0,
            "headings_seen": 0,
            "headings_repaired": 0,
            "headings_demoted": 0,
            "appendix_started": 0,
            "docs": set(),  # type: ignore
        }

    def path(self, depth: int = 3) -> str:
        return " > ".join(self.labels[-depth:])

    def set_chain(self, parts: List[str], end_label: str) -> None:
        parent = parts[:-1]
        synth = [".".join(parent[:i+1]) for i in range(len(parent))]
        self.tokens = parts[:]
        self.labels = synth + [end_label]

    def accept(self, parts: List[str], label: str) -> bool:
        major = parts[0]
        numeric = is_numeric(major)

        # phase control
        if self.phase == "numeric" and not numeric:
            self.phase = "appendix"
            self.stats["appendix_started"] += 1
        if self.phase == "appendix" and numeric:
            return False

        # major validity
        if not valid_major(major, self.phase):
            return False

        # majors (depth 1)
        if len(parts) == 1:
            self.tokens = [major]
            self.labels = [label]
            return True

        # different major than current -> handle
        if not self.tokens or self.tokens[0] != major:
            if self.mode == "demote":
                return False
            # heal by resetting chain
            self.set_chain(parts, label)
            return True

        # ensure exact prefix; synthesize missing parents if needed
        parent = parts[:-1]
        k = 0
        while k < min(len(self.tokens), len(parent)) and self.tokens[k] == parent[k]:
            k += 1
        if k != len(parent):
            if self.mode == "demote":
                return False
            self.set_chain(parts, label)
            return True

        # normal child advance
        self.tokens = parts[:]
        if len(self.labels) == len(parent):
            self.labels.append(label)
        else:
            self.labels = self.labels[:len(parent)] + [label]
        return True

    def process(self, records: List[Rec]) -> List[Dict]:
        out: List[Dict] = []
        cur_doc = None
        self.tokens, self.labels, self.phase = [], [], "numeric"

        for r in records:
            self.stats["total"] += 1
            if r.doc_id != cur_doc:
                cur_doc = r.doc_id
                self.stats["docs"].add(cur_doc)
                self.tokens, self.labels, self.phase = [], [], "numeric"

            raw = r.raw

            if r.is_heading and r.parts:
                self.stats["headings_seen"] += 1
                accepted = self.accept(r.parts, r.label or ".".join(r.parts))
                if not accepted:
                    raw["block_type"] = "paragraph"
                    self.stats["headings_demoted"] += 1
                elif ".".join(self.tokens) != ".".join(r.parts) or (self.labels and self.labels[-1] != (r.label or "")):
                    # chain changed -> considered a repair
                    self.stats["headings_repaired"] += 1

            raw["section_path"] = self.path()
            out.append(raw)

        return out

# ---------- Loading & dispatch ----------

def to_recs(objs: Iterable[Dict]) -> List[Rec]:
    out: List[Rec] = []
    for o in objs:
        bt = o.get("block_type")
        text = o.get("markdown") or o.get("text") or ""
        parsed = parse_heading_text(text)
        parts = parsed[0] if parsed else None
        label = parsed[2] if parsed else None
        out.append(Rec(
            raw=o,
            is_heading=(bt == "heading" and parsed is not None),
            parts=parts,
            label=label,
            doc_id=o.get("doc_id",""),
            page=int(o.get("page", 0)),
        ))
    return out

def audit_print(sr: SectionRepair) -> None:
    docs = len(sr.stats["docs"])
    print(f"Processed records: {sr.stats['total']}")
    print(f"Docs: {docs}")
    print(f"Headings seen: {sr.stats['headings_seen']}")
    print(f"Headings repaired: {sr.stats['headings_repaired']}")
    print(f"Headings demoted: {sr.stats['headings_demoted']}")
    print(f"Appendix started (docs/pages encountering A..I): {sr.stats['appendix_started']}")

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Repair or sanitize section paths in ingested JSONL.")
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL from ingestion")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with repaired paths")
    ap.add_argument("--mode", choices=["heal","demote"], default="heal", help="heal = rebuild chain; demote = drop inconsistent headings")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)

    objs = list(read_jsonl(inp))
    recs = to_recs(objs)
    sr = SectionRepair(mode=args.mode)
    fixed = sr.process(recs)
    write_jsonl(outp, fixed)
    audit_print(sr)

if __name__ == "__main__":
    main()