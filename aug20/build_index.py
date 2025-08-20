#!/usr/bin/env python3
# build_index.py
# Build a local RAG index from JSONL files produced by ingest.py

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Union
from joblib import dump, load

# Import your retriever
from retriever import Retriever

# --------------------------
# Helpers
# --------------------------

def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read chunks from JSONL file"""
    items: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        print(f"Error: {p} does not exist", file=sys.stderr)
        return items
    
    with p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"[read] Loaded {len(items)} chunks from {p}")
    return items

def normalize_chunk_fields(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert field names from ingest.py format to retriever.py format.
    ingest.py outputs 'block_id' while retriever.py expects 'chunk_id'
    """
    normalized = chunk.copy()
    
    # Rename block_id to chunk_id
    if 'block_id' in normalized and 'chunk_id' not in normalized:
        normalized['chunk_id'] = normalized['block_id']
    
    # Ensure all required fields exist with defaults
    defaults = {
        'chunk_id': '',
        'block_type': 'paragraph',
        'text': '',
        'section_intro': '',
        'section_path': '',
        'section_label': '',
        'table_markdown': '',
        'table_summary': '',
        'page': 0,
        'hidden_terms': []
    }
    
    for key, default_value in defaults.items():
        if key not in normalized:
            normalized[key] = default_value
    
    return normalized

def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate chunks based on chunk_id"""
    seen = {}
    for chunk in chunks:
        cid = chunk.get('chunk_id', '')
        if cid:
            seen[cid] = chunk
    deduped = list(seen.values())
    if len(deduped) < len(chunks):
        print(f"[dedupe] Removed {len(chunks) - len(deduped)} duplicate chunks")
    return deduped

# --------------------------
# Build Index
# --------------------------

def build_index(
    jsonl_files: List[str],
    output_dir: Union[str, Path] = "index",
    save_retriever: bool = True
) -> Dict[str, Path]:
    """
    Build index from JSONL files
    
    Args:
        jsonl_files: List of JSONL file paths
        output_dir: Directory to save the index
        save_retriever: Whether to save the retriever.pkl file
    
    Returns:
        Dict with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all chunks from all JSONL files
    all_chunks = []
    for jsonl_path in jsonl_files:
        chunks = read_jsonl(jsonl_path)
        if chunks:
            all_chunks.extend(chunks)
    
    if not all_chunks:
        print("Error: No chunks found in any JSONL files", file=sys.stderr)
        return {}
    
    print(f"[total] Loaded {len(all_chunks)} chunks from {len(jsonl_files)} file(s)")
    
    # Normalize field names for retriever compatibility
    all_chunks = [normalize_chunk_fields(chunk) for chunk in all_chunks]
    
    # Deduplicate
    all_chunks = dedupe_chunks(all_chunks)
    
    # Save normalized chunks
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[write] Saved {len(all_chunks)} chunks to {chunks_path}")
    
    result = {"chunks": chunks_path}
    
    # Build and save retriever
    if save_retriever:
        print("[build] Building retriever index...")
        retriever = Retriever()
        retriever.fit(all_chunks)
        
        retriever_path = output_dir / "retriever.pkl"
        dump(retriever, retriever_path)
        print(f"[save] Saved retriever to {retriever_path}")
        result["retriever"] = retriever_path
    
    return result

# --------------------------
# Search
# --------------------------

def load_retriever(index_dir: Union[str, Path]) -> Retriever:
    """Load retriever from index directory"""
    index_dir = Path(index_dir)
    
    # Try to load pre-built retriever
    retriever_path = index_dir / "retriever.pkl"
    if retriever_path.exists():
        print(f"[load] Loading retriever from {retriever_path}")
        return load(retriever_path)
    
    # Fallback: build retriever from chunks.jsonl
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No retriever.pkl or chunks.jsonl found in {index_dir}")
    
    print(f"[load] Building retriever from {chunks_path}")
    chunks = read_jsonl(chunks_path)
    retriever = Retriever()
    retriever.fit(chunks)
    return retriever

def search(index_dir: Union[str, Path], query: str, top_k: int = 5) -> None:
    """Search the index and print results"""
    retriever = load_retriever(index_dir)
    results = retriever.search(query, top_k=top_k)
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        page = result.get("page", 0)
        block_type = result.get("block_type", "")
        section = result.get("section_label", "") or result.get("section_path", "")
        text = result.get("text", "") or result.get("table_summary", "")
        
        print(f"{i}. [Score: {score:.3f}] Page {page} | Type: {block_type}")
        if section:
            print(f"   Section: {section}")
        print(f"   {text[:200]}...")
        print()

# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build and search a RAG index from JSONL files produced by ingest.py"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build index from JSONL files")
    build_parser.add_argument(
        "jsonl_files", 
        nargs="+", 
        help="JSONL files to build index from (e.g., 409.corpus.blocks.jsonl)"
    )
    build_parser.add_argument(
        "--output-dir", 
        default="index", 
        help="Output directory for index files (default: index)"
    )
    build_parser.add_argument(
        "--no-retriever", 
        action="store_true", 
        help="Skip building retriever.pkl (only normalize chunks.jsonl)"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument(
        "--index-dir", 
        default="index", 
        help="Index directory containing retriever.pkl or chunks.jsonl (default: index)"
    )
    search_parser.add_argument(
        "-q", "--query", 
        required=True, 
        help="Search query"
    )
    search_parser.add_argument(
        "-k", "--top-k", 
        type=int, 
        default=5, 
        help="Number of results to return (default: 5)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "build":
        build_index(
            jsonl_files=args.jsonl_files,
            output_dir=args.output_dir,
            save_retriever=not args.no_retriever
        )
    elif args.command == "search":
        search(
            index_dir=args.index_dir,
            query=args.query,
            top_k=args.top_k
        )

if __name__ == "__main__":
    main()
