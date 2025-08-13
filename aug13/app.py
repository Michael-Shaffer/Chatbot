#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any, Dict, List

import requests
from flask import Flask, jsonify, request

from retriever import Retriever  # <- your tqdm-enabled retriever (set progress=False below)

# ---------- Config (edit these) ----------
INDEX_DIR = "rag_index"                          # output dir from build_index.py
EMBED_MODEL_PATH = "/models/bge-base-en-v1.5"    # local encoder dir
RERANKER_PATH = ""                               # optional local cross-encoder dir; "" disables
DEVICE = "auto"                                  # "cuda" | "cpu" | "auto"
QUERY_PREFIX = "query: "                         # "" for non-BGE models

LLAMA_URL = "http://localhost:8000/v1/chat/completions"  # your local OpenAI-compatible endpoint
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Retrieval knobs
KD = 80        # dense candidates
KS = 50        # BM25 candidates
MMR_K = 40     # diversified pool size
FINAL_K = 10   # blocks passed to the model

# Generation knobs
TEMP = 0.2
TOP_P = 0.9
MAX_TOK = 900

SYSTEM_PROMPT = (
    "You are a careful technical assistant. Answer ONLY from the provided sources.\n"
    "Cite every factual sentence using bracketed tags like [doc:page].\n"
    "If the answer is not supported by the sources, say you don't know.\n"
    "For tables, compute requested aggregates explicitly and show the working."
)

# ---------- App ----------
app = Flask(__name__)

RET = Retriever(
    index_dir=INDEX_DIR,
    embed_model_path=EMBED_MODEL_PATH,
    reranker_path=(RERANKER_PATH or None),
    device=DEVICE,
    query_prefix=QUERY_PREFIX,
    progress=False,          # keep server logs clean
)

# ---------- Helpers ----------
def format_context(blocks: List[Dict[str, Any]]) -> str:
    parts = []
    for b in blocks:
        tag = f"[{b['doc_id']}:{b['page']}]"
        parts.append(f"{tag} {b['text']}")
    return "\n\n".join(parts)

def call_llama(query: str, blocks: List[Dict[str, Any]]) -> str:
    ctx = format_context(blocks)
    user_msg = f"Query:\n{query}\n\nSources:\n{ctx}\n\nAnswer with citations like [doc:page]."
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMP,
        "top_p": TOP_P,
        "max_tokens": MAX_TOK,
    }
    r = requests.post(LLAMA_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------- Routes ----------
@app.route("/ask", methods=["POST"])
def ask() -> Any:
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400

    blocks = RET.hybrid_search(query, kd=KD, ks=KS, mmr_k=MMR_K, final_k=FINAL_K)
    if not blocks:
        return jsonify({"answer": "I don’t have enough evidence in the provided sources to answer.", "sources": []})

    answer = call_llama(query, blocks)
    return jsonify({"answer": answer, "sources": blocks})

@app.route("/healthz", methods=["GET"])
def healthz() -> Any:
    return jsonify({"ok": True})

# ---------- Main ----------
if __name__ == "__main__":
    # Run:  python3 app_rag.py
    # Then: curl -s localhost:5050/ask -H 'Content-Type: application/json' -d '{"query":"..."}' | jq
    app.run(host="0.0.0.0", port=5050, debug=False)