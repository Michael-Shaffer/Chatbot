# generator.py
from __future__ import annotations
import json, requests
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are a careful technical assistant. Answer ONLY from the provided sources.\n"
    "Cite every factual sentence using bracketed tags like [doc:page].\n"
    "If the answer is not supported by the sources, say you don't know.\n"
    "For tables, compute requested aggregates explicitly and show the working."
)

def format_context(blocks:List[Dict[str,Any]])->str:
    # keep it tight; leave tokens for the answer
    lines=[]
    for b in blocks:
        tag=f"[{b['doc_id']}:{b['page']}]"
        # show just the content—model doesn't need metadata here
        lines.append(f"{tag} {b['text']}")
    return "\n\n".join(lines)

def ask_llama(query:str, blocks:List[Dict[str,Any]], llama_url:str, model:str)->str:
    ctx = format_context(blocks)
    user = f"Query:\n{query}\n\nSources:\n{ctx}\n\nAnswer with citations like [doc:page]."
    payload = {
        "model": model,
        "messages": [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 900
    }
    r = requests.post(llama_url, headers={"Content-Type":"application/json"}, data=json.dumps(payload), timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]