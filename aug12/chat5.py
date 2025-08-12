#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_NUM_HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,6})[)\s\-–—:]+(?P<title>\S.+)$")

@dataclass
class Rec:
    raw: Dict
    is_heading: bool
    levels: Optional[List[int]]
    label: Optional[str]

def parse_heading(text: str) -> Optional[Tuple[List[int], str, str]]:
    m = _NUM_HEADING_RE.match(text or "")
    if not m: return None
    num = m.group("num"); title = (m.group("title") or "").strip()
    parts = [int(x) for x in num.split(".") if x.isdigit()]
    if not parts: return None
    label = f"{num} {title}" if title else num
    return parts, title, label

def load_recs(path: Path) -> List[Rec]:
    out: List[Rec] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h = r.get("block_type") == "heading"
            parsed = parse_heading(r.get("markdown") or r.get("text") or "")
            levels = parsed[0] if parsed else None
            label = parsed[2] if parsed else None
            out.append(Rec(raw=r, is_heading=h and bool(parsed), levels=levels, label=label))
    return out

def heal_sections(recs: List[Rec], mode: str = "heal") -> None:
    cur_nums: List[int] = []
    cur_labels: List[str] = []
    last_major: Optional[int] = None

    def set_stack(nums: List[int], labels: List[str]) -> None:
        cur_nums[:] = nums
        cur_labels[:] = labels

    def path(max_depth: int = 3) -> str:
        return " > ".join(cur_labels[-max_depth:])

    for rec in recs:
        r = rec.raw
        if rec.is_heading and rec.levels:
            lv = rec.levels
            depth = len(lv)
            if depth == 1:
                major = lv[0]
                if last_major is not None and mode == "demote" and major < last_major:
                    r["block_type"] = "paragraph"
                else:
                    set_stack([major], [rec.label or str(major)])
                    last_major = major
            else:
                if not cur_nums or (mode == "demote" and lv[0] != cur_nums[0]):
                    if mode == "heal":
                        # start a new chain at this major
                        labels = [str(lv[0])]
                        for i in range(1, depth):
                            labels.append(".".join(map(str, lv[:i+1])))
                        labels[-1] = rec.label or labels[-1]
                        set_stack(lv, labels)
                        last_major = lv[0]
                    else:
                        r["block_type"] = "paragraph"
                else:
                    # ensure prefix; if missing parents, synthesize
                    k = 0
                    while k < min(len(cur_nums), depth-1) and cur_nums[k] == lv[k]:
                        k += 1
                    nums = lv[:k]
                    labels = cur_labels[:k]
                    while len(nums) < depth-1:
                        nums.append(lv[len(nums)])
                        labels.append(".".join(map(str, nums)))
                    nums = lv
                    labels.append(rec.label or ".".join(map(str, nums)))
                    set_stack(nums, labels)
            r["section_path"] = path()
            r["html"] = r["html"] if r.get("block_type") == "heading" else r["html"]
        else:
            r["section_path"] = path()

def main() -> None:
    ap = argparse.ArgumentParser(description="Repair section hierarchy in RAG JSONL.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--mode", choices=["heal","demote"], default="heal")
    args = ap.parse_args()

    recs = load_recs(Path(args.in_jsonl))
    heal_sections(recs, mode=args.mode)
    with Path(args.out_jsonl).open("w", encoding="utf-8") as fo:
        for rec in recs:
            fo.write(json.dumps(rec.raw, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()