# vectordb.py
# Contains functions for interacting with the ChromaDB vector store.
# Last updated: 2025-05-22

import chromadb
import os

# Import the embedding function directly for query embedding
from embed import get_embedding_model 

# Global variables to cache ChromaDB client and collection
_chroma_client = None
_chroma_collection = None

def initialize_vector_store(collection_name="atc_documents", db_path="./chroma_db_store"):
    """Initializes ChromaDB client and collection, persisting to disk."""
    global _chroma_client, _chroma_collection
    
    if _chroma_collection is None: # Initialize only if not already done
        print(f"Initializing ChromaDB vector store at: {db_path} with collection: {collection_name}")
        # Ensure the parent directory for the DB path exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(path=db_path)
        try:
            _chroma_collection = _chroma_client.get_or_create_collection(name=collection_name)
            print(f"Vector store initialized. Collection '{collection_name}' ({_chroma_collection.count()} items) ready.")
        except Exception as e: # Handle potential db access issues, e.g. from another process.
            print(f"Error getting or creating ChromaDB collection '{collection_name}': {e}")
            # Attempt to get the collection if creation failed but it might exist
            try:
                _chroma_collection = _chroma_client.get_collection(name=collection_name)
                print(f"Successfully got existing collection '{collection_name}' after create error.")
            except Exception as e_get:
                print(f"Failed to get collection '{collection_name}' as fallback: {e_get}")
                raise # Re-raise if cannot get or create
    return _chroma_collection # Return the collection object

def add_chunks_to_vector_store(chunks_with_embeddings_list):
    """Adds chunks (with pre-computed embeddings and metadata) to the initialized ChromaDB collection."""
    global _chroma_collection
    if _chroma_collection is None:
        print("Error: Vector store not initialized. Call initialize_vector_store first.")
        return
    if not chunks_with_embeddings_list:
        print("No chunks provided to add to vector store.")
        return

    ids = []
    documents_content = [] 
    metadatas_list = []
    embeddings_list = []

    for i, chunk_data in enumerate(chunks_with_embeddings_list):
        # Construct a unique ID for each chunk
        doc_id = chunk_data['metadata'].get('document_id', 'unknown_doc')
        page_num = chunk_data['metadata'].get('source_page', 'pNA')
        item_id = chunk_data['metadata'].get('original_item_id', 'iNA')
        chunk_id = chunk_data['metadata'].get('chunk_id_in_item', i) # Use chunk_id_in_item or fallback
        
        unique_id = f"doc_{doc_id}_page_{page_num}_item_{item_id}_chunk_{chunk_id}"
        ids.append(unique_id)
        
        documents_content.append(chunk_data['page_content'])
        metadatas_list.append(chunk_data['metadata']) # Store all metadata
        
        embedding_val = chunk_data['embedding']
        if hasattr(embedding_val, 'tolist'): # Convert numpy array to list if needed
            embedding_val = embedding_val.tolist()
        embeddings_list.append(embedding_val)

    print(f"Adding/updating {len(ids)} chunks in ChromaDB collection '{_chroma_collection.name}'...")
    try:
        # Use upsert to add new or update existing chunks by ID
        _chroma_collection.upsert(
            ids=ids,
            documents=documents_content,
            metadatas=metadatas_list,
            embeddings=embeddings_list
        )
        print(f"Chunks successfully upserted. Collection count: {_chroma_collection.count()}")
    except Exception as e:
        print(f"Error upserting chunks to ChromaDB: {e}")
        # Consider logging more details or partial adds if critical

def retrieve_relevant_chunks(query_text, n_results=5, embedding_model_name='all-MiniLM-L6-v2'):
    """Embeds a query and retrieves relevant chunks from the initialized ChromaDB collection."""
    global _chroma_collection
    if _chroma_collection is None:
        print("Error: Vector store not initialized. Call initialize_vector_store first.")
        return None # Or raise an error

    query_embedding_model = get_embedding_model(embedding_model_name) # From embed.py
    
    # print(f"Embedding query: '{query_text}'")
    query_embedding = query_embedding_model.encode(query_text).tolist()
    
    # print(f"Querying ChromaDB collection '{_chroma_collection.name}' for {n_results} results...")
    try:
        results = _chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] 
        )
        # print("Query results received from ChromaDB.")
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return None
