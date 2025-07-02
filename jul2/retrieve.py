# src/retrieval/retriever.py

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from typing import List
# Import the RAGChunk class from our other module
from ..ingestion.chunker import RAGChunk

class HybridRetriever:
    def __init__(self, chunks: List[RAGChunk], embedding_model_name: str = 'all-MiniLM-L6-v2'):
        print("Initializing Hybrid Retriever...")
        self.chunks = chunks
        self.documents = [chunk.content for chunk in chunks]

        # BM25 Retriever
        tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Semantic Retriever
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print("Retriever initialization complete.")

    def search(self, query: str, k: int = 5) -> List[RAGChunk]:
        """Performs a hybrid search and returns the top k chunks."""
        print(f"\n--- Performing Hybrid Search for query: '{query}' ---")

        # Keyword Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = {idx: score for idx, score in enumerate(bm25_scores) if idx in top_bm25_indices}

        # Semantic Search
        query_embedding = self.embedding_model.encode([query])
        distances, top_semantic_indices = self.index.search(query_embedding, k)
        semantic_results = {idx: dist for idx, dist in zip(top_semantic_indices[0], distances[0])}

        # Reciprocal Rank Fusion
        fused_scores = {}
        rrf_k = 60 

        sorted_bm25 = sorted(bm25_results.items(), key=lambda item: item[1], reverse=True)
        for rank, (doc_index, score) in enumerate(sorted_bm25):
            fused_scores[doc_index] = fused_scores.get(doc_index, 0) + 1 / (rrf_k + rank + 1)

        sorted_semantic = sorted(semantic_results.items(), key=lambda item: item[1])
        for rank, (doc_index, score) in enumerate(sorted_semantic):
            fused_scores[doc_index] = fused_scores.get(doc_index, 0) + 1 / (rrf_k + rank + 1)
            
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_chunks = [self.chunks[doc_index] for doc_index, score in reranked_results[:k]]
        return final_chunks

