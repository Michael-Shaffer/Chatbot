from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import numpy as np
import pickle
import faiss

EMBEDDING_DATABASE_PATH = "embedding_database.pkl"
EMBEDDING_PATH = "../models/all-mpnet-base-v2"
EMBEDDER = SentenceTransformer(EMBEDDING_PATH)

def initialize():
    return Embedding_Database()

def fetch():
    with open(EMBEDDING_DATABASE_PATH, "rb") as f:
        embedding_database = pickle.load(f)
    return embedding_database

class Embedding_Database:

    def __init__(self, dim=768):
        self.dimension = dim
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.save()

    def save(self):

        with open(EMBEDDING_DATABASE_PATH, "wb") as f:
            pickle.dump(self, f, -1)

    def embed(self, text: str):
        return np.array(EMBEDDER.encode([text]))

    def add(self, chunks: Document):

        self.chunks += chunks
        for chunk in self.chunks:
            embedding = self.embed(chunk.page_content)
            self.index.add(embedding)

        self.save()

    def query(self, question: str, k: int):

        question_embedding = self.embed(question)
        distances, indices = self.index.search(question_embedding, k)

        relevant_chunks = []
        relevant_indices = indices.tolist()[0]

        for i in relevant_indices:
            chunk = self.chunks[i].page_content
            # Preserve formatting for tables and figures
            if "[TABLE]" in chunk or "[FIGURE]" in chunk:
                relevant_chunks.append(chunk)
            else:
                relevant_chunks.append(chunk)

        return relevant_chunks
        