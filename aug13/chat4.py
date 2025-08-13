# app.py  (only the RAG-specific bits)
from __future__ import annotations
from flask import Flask, request, jsonify
from retriever import Retriever
from generator import ask_llama

app = Flask(__name__)
RET = Retriever(index_dir="rag_index")  # adjust path
LLAMA_URL = "http://localhost:8000/v1/chat/completions"
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = data.get("query","").strip()
    if not query:
        return jsonify({"error":"empty query"}), 400
    blocks = RET.hybrid_search(query, kd=80, ks=50, mmr_k=40, final_k=10)
    answer = ask_llama(query, blocks, LLAMA_URL, LLAMA_MODEL)
    # return answer plus the exact sources for your UI to display
    return jsonify({"answer": answer, "sources": blocks})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)