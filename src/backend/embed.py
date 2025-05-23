# embed.py
# Contains functions for generating embeddings for text chunks.
# Last updated: 2025-05-22

from sentence_transformers import SentenceTransformer

# Global variable to cache the embedding model
_embedding_model_instance = None

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Loads or returns a cached SentenceTransformer model."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print(f"Loading embedding model: {model_name}...")
        try:
            _embedding_model_instance = SentenceTransformer(model_name)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            raise
    return _embedding_model_instance

def embed_chunks(chunks_with_metadata_list, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for the 'page_content' of each chunk.
    Returns the list of chunks with an 'embedding' key added to each.
    """
    model = get_embedding_model(model_name) # Ensures model is loaded
    
    contents_to_embed = [chunk['page_content'] for chunk in chunks_with_metadata_list if 'page_content' in chunk]
    
    if not contents_to_embed:
        print("No content found in chunks to embed.")
        return chunks_with_metadata_list # Return original list if no content

    print(f"Generating embeddings for {len(contents_to_embed)} chunks...")
    try:
        embeddings = model.encode(contents_to_embed, show_progress_bar=True)
        print("Embeddings generated.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise
    
    # Add embeddings back to each chunk dictionary
    # This assumes the order of embeddings matches the order of chunks_with_metadata_list
    embedding_idx = 0
    for chunk in chunks_with_metadata_list:
        if 'page_content' in chunk: # Only add embedding if there was content
            chunk['embedding'] = embeddings[embedding_idx]
            embedding_idx += 1
            
    return chunks_with_metadata_list
