#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, send_from_directory, request, jsonify, Response, stream_with_context
from transformers import AutoTokenizer

# Local imports (your tree)
from core.retrieval import acronyms, context_window
from core.retrieval import retriever  # must expose class `Retriever`

# ---------------- Config ----------------

VLLM_HOST = "http://0.0.0.0:8001"
URL = f"{VLLM_HOST}/v1/chat/completions"

INDEX_DIR = "/devfiles/git_repos_big2/shaffem/polaris/src/core/ingestion/rag_index"
EMBED_MODEL_PATH = "/devfiles/git_repos_big2/shaffem/polaris/models/bge-base-en-v1.5"
RERANKER_PATH = ""  # keep empty to disable if you had shape issues
DEVICE = "cpu"
QUERY_PREFIX = "query: "

KD = 80
KS = 50
MMR_K = 40
FINAL_K = 10

# Page excerpts appended by Retriever (synthetic "page overview" blocks)
APPEND_PAGE_EXCERPTS = True
PAGE_EXCERPT_CHARS = 450
PAGE_EXCERPT_LIMIT = 2
INCLUDE_EXCERPTS_IN_PROMPT = True  # include those page overviews inside the prompt

MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
TEMP = 0.2
TOP_P = 0.9
MAX_TOK = 900
TOKENIZER_PATH = "/devfiles/git_repos_big2/shaffem/polaris/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# ---------------- App ----------------

app = Flask(__name__, static_folder="web", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_fallback_secret_key_please_change_this")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

RET = retriever.Retriever(
    index_dir=INDEX_DIR,
    embed_model_path=EMBED_MODEL_PATH,
    reranker_path=(RERANKER_PATH or None),
    device=DEVICE,
    query_prefix=QUERY_PREFIX,
    progress=False,
    append_page_excerpt=APPEND_PAGE_EXCERPTS,
    page_excerpt_chars=PAGE_EXCERPT_CHARS,
    page_excerpt_limit=PAGE_EXCERPT_LIMIT,
)

# ---------------- Helpers ----------------

def _tag(b: Dict[str, Any]) -> str:
    return f"[{b.get('doc_id','?')}:{b.get('page','?')}]"

def format_context(blocks: List[Dict[str, Any]], include_excerpts: bool = True) -> str:
    """
    Put normal retrieved chunks first, then optional page-overview excerpts.
    Citations are [doc:page].
    """
    header = "<relevant_chunks>Here are chunks relevant to the user’s query"
    normal, overviews = [header], []
    for b in blocks:
        line = f"{_tag(b)} {b.get('text','')}"
        if b.get("is_page_excerpt") or b.get("block_type") == "page_excerpt":
            if include_excerpts:
                overviews.append(line)
        else:
            normal.append(line)

    parts = []
    if len(normal) > 1:
        parts.append("\n".join(normal))
    if include_excerpts and overviews:
        parts.append("\n-- Page Overviews --\n" + "\n".join(overviews))
    return "\n\n".join(parts) if parts else ""

def _ensure_system_message(messages: List[Dict[str, str]], system_text: str) -> List[Dict[str, str]]:
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": system_text}, *messages]
    messages[0]["content"] = system_text
    return messages

def _inject_context_into_last_user(messages: List[Dict[str, str]], ctx: str) -> None:
    """
    Prepend Sources to the final user message content.
    """
    if not messages:
        return
    last = messages[-1]
    user_text = acronyms.resolve(last.get("content", ""))
    if ctx:
        last["content"] = f"{user_text}\n\nSources:\n{ctx}\n\nAnswer with bracketed citations like [doc:page]."
    else:
        last["content"] = user_text

def _clip_messages_to_budget(messages: List[Dict[str, str]], max_tokens: int = 3500) -> List[Dict[str, str]]:
    total = sum(len(tokenizer.encode(m.get("content", ""))) for m in messages)
    if total <= max_tokens:
        return messages
    return context_window.manage(messages)

# ---------------- Routes ----------------

@app.route("/")
def home():
    return send_from_directory("web", "index.html")

@app.route("/chat/")
def chat():
    return send_from_directory("web/chat/", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "missing messages"}), 400

    # Retrieve
    user_query = messages[-1].get("content", "")
    blocks = RET.hybrid_search(user_query, kd=KD, ks=KS, mmr_k=MMR_K, final_k=FINAL_K)

    # System prompt
    try:
        with open("system.md", "r", encoding="utf-8") as f:
            system_md = f.read()
    except Exception:
        system_md = (
            "You are a careful technical assistant. Answer ONLY from the provided sources. "
            "Cite every factual statement like [doc:page]. If the sources don’t support the answer, say you don’t know. "
            "‘Page overview’ snippets summarize a page; do not cite them for specific numbers—cite original chunks."
        )

    messages = _ensure_system_message(messages, system_md)

    # Context -> last user message
    ctx = format_context(blocks, include_excerpts=INCLUDE_EXCERPTS_IN_PROMPT)
    _inject_context_into_last_user(messages, ctx)

    # Token budget
    messages = _clip_messages_to_budget(messages, max_tokens=3500)

    # vLLM request
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMP,
        "top_p": TOP_P,
        "max_tokens": MAX_TOK,
        "stream": True,
    }

    def generate_stream():
        try:
            vllm_response = requests.post(
                URL,
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=300,
            )
            vllm_response.raise_for_status()
            for chunk in vllm_response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with vLLM: {e}")
            yield f"data: {json.dumps({'error': f'LLM connection error: {e}'})}\n\n".encode("utf-8")
        except Exception as e:
            print(f"Unexpected error during streaming: {e}")
            yield f"data: {json.dumps({'error': 'Internal server error.'})}\n\n".encode("utf-8")

    return Response(stream_with_context(generate_stream()), mimetype="text/event-stream")

# ---------------- Main ----------------

if __name__ == "__main__":
    if not app.secret_key:
        raise ValueError("No SECRET_KEY set for polaris flask application")
    app.run(debug=True, port=800)
