# main_chatbot.py
# Main orchestrator for the RAG chatbot.
# Last updated: 2025-05-22

import os
import json

# --- Import functions from your custom modules ---
from extract1 import process_pdf_for_rag
from chunk import chunk_extracted_data
from embed import get_embedding_model, embed_chunks
from vectordb import initialize_vector_store, add_chunks_to_vector_store 
# retrieve_relevant_chunks is used by rag.py

from llm import get_llm_instance # For pre-loading/checking LLM
# get_llm_response_self_contained is used by rag.py

from rag import answer_query_with_rag # Renamed from answer_query_with_rag_self_contained for clarity here

if __name__ == "__main__":
    print("Initializing ATC RAG Chatbot...")

    # --- Critical Configuration - USER MUST SET THESE ---
    # Path to your Air Traffic Control PDF document
    pdf_document_path = "../../docs/atc325.pdf"  # <--- !!! SET THIS PATH !!!
    # Path to your downloaded GGUF LLM model file
    llm_model_file_path = "../../models/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf" # <--- !!! SET THIS PATH !!!
    # --- End Critical Configuration ---

    # General Configuration
    rag_data_root_dir = "rag_atc_data_self_contained" # Top-level directory for all RAG data
    
    # Derived path for the specific PDF's processed data
    pdf_filename_base = os.path.splitext(os.path.basename(pdf_document_path))[0]
    pdf_processed_data_dir = os.path.join(rag_data_root_dir, pdf_filename_base)
    chunks_with_embeddings_json_path = os.path.join(pdf_processed_data_dir, f"{pdf_filename_base}_embedded_chunks.json")
    
    # Vector DB configuration
    vector_db_persistent_path = os.path.join(rag_data_root_dir, "vector_store_chroma") # DB stored here
    vector_db_collection_name = f"atc_docs_{pdf_filename_base}" # Collection name specific to the PDF
    
    # Embedding model
    embedding_model_to_use = 'all-MiniLM-L6-v2' # Keep consistent with embed.py and vectordb.py defaults if not passed

    # LLM Parameters
    llm_context_window_config = 4096  # Max context for Llama.cpp
    llm_gpu_layers_config = -1        # -1: all to GPU, 0: CPU only. Adjust based on your VRAM (32GB should handle -1 for 8B models)

    # --- Initial Checks ---
    if not os.path.exists(pdf_document_path):
        print(f"FATAL ERROR: PDF document not found at '{pdf_document_path}'. Please check the path.")
        exit()
    if not os.path.exists(llm_model_file_path):
        print(f"FATAL ERROR: LLM model file not found at '{llm_model_file_path}'. Please check the path.")
        exit()

    # --- Step 1: Data Preparation (Extraction, Chunking, Embedding) ---
    # This step runs if the final embedded chunks JSON doesn't exist.
    os.makedirs(pdf_processed_data_dir, exist_ok=True) # Ensure PDF's specific data directory exists

    if not os.path.exists(chunks_with_embeddings_json_path):
        print(f"No pre-processed embedded chunks found at '{chunks_with_embeddings_json_path}'.")
        print("Starting data preparation pipeline (this may take some time)...")
        
        # 1a. Extract content from PDF
        raw_extracted_content_list = process_pdf_for_rag(pdf_document_path, output_base_dir=rag_data_root_dir)
        if not raw_extracted_content_list:
            print(f"No content could be extracted from '{pdf_document_path}'. Exiting.")
            exit()

        # 1b. Chunk the extracted content
        print("\nChunking extracted content...")
        document_chunks_list = chunk_extracted_data(raw_extracted_content_list) # Using defaults for chunk_size/overlap
        if not document_chunks_list:
            print("No chunks were created from the extracted content. Exiting.")
            exit()
        print(f"Created {len(document_chunks_list)} chunks for embedding.")

        # 1c. Generate embeddings for chunks
        print("\nGenerating embeddings for chunks...")
        # get_embedding_model is called within embed_chunks to load/cache the model
        chunks_with_their_embeddings_list = embed_chunks(document_chunks_list, model_name=embedding_model_to_use)
        if not chunks_with_their_embeddings_list or not chunks_with_their_embeddings_list[0].get('embedding', None) is not None :
            print("Failed to generate embeddings or embeddings are missing. Exiting.")
            exit()

        # Save the processed chunks with embeddings to JSON for future runs
        temp_serializable_list = []
        for chunk_data in chunks_with_their_embeddings_list:
            # Convert numpy embeddings to lists for JSON serialization
            if 'embedding' in chunk_data and hasattr(chunk_data['embedding'], 'tolist'):
                chunk_data_copy = chunk_data.copy() # Avoid modifying original if it's used later
                chunk_data_copy['embedding'] = chunk_data_copy['embedding'].tolist()
                temp_serializable_list.append(chunk_data_copy)
            else: # Should not happen if embed_chunks worked
                temp_serializable_list.append(chunk_data.copy())
        
        with open(chunks_with_embeddings_json_path, "w", encoding="utf-8") as f_out:
            json.dump(temp_serializable_list, f_out, indent=2)
        print(f"Processed chunks with embeddings saved to '{chunks_with_embeddings_json_path}'")
        
        # Use the original list with numpy arrays for vector DB (Chroma handles numpy)
        final_chunks_for_db = chunks_with_their_embeddings_list

    else:
        print(f"Loading pre-processed chunks with embeddings from '{chunks_with_embeddings_json_path}'...")
        with open(chunks_with_embeddings_json_path, "r", encoding="utf-8") as f_in:
            # Embeddings here are lists, ChromaDB can handle lists of floats.
            final_chunks_for_db = json.load(f_in) 
        print(f"Loaded {len(final_chunks_for_db)} chunks with embeddings.")
        # Ensure embedding model is available for queries later (cached in embed.py)
        get_embedding_model(model_name=embedding_model_to_use)


    # --- Step 2: Initialize Vector Store and Add/Verify Chunks ---
    print("\nInitializing and populating vector store (ChromaDB)...")
    # This will get or create the collection. The collection object is cached in vectordb.py
    chroma_collection = initialize_vector_store(
        collection_name=vector_db_collection_name, 
        db_path=vector_db_persistent_path
    )
    
    # Check if data needs to be added (simple check based on count)
    # A more robust check would involve IDs or checksums if content could change frequently.
    current_db_item_count = chroma_collection.count()
    if current_db_item_count != len(final_chunks_for_db):
        print(f"Vector store count ({current_db_item_count}) differs from loaded chunks ({len(final_chunks_for_db)}).")
        print("Attempting to add/update chunks in vector store...")
        # add_chunks_to_vector_store uses upsert, so it's safe to call even if some items exist.
        add_chunks_to_vector_store(final_chunks_for_db) 
    else:
        print(f"Vector store collection '{vector_db_collection_name}' ({current_db_item_count} items) appears up-to-date.")


    # --- Step 3: Pre-load Self-Contained LLM ---
    print("\nInitializing self-contained LLM (Llama.cpp)...")
    try:
        # This loads the LLM and caches the instance in llm.py
        get_llm_instance(
            model_path=llm_model_file_path, 
            n_gpu_layers=llm_gpu_layers_config, 
            n_ctx=llm_context_window_config
        )
        print("Self-contained LLM is ready.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize LLM from '{llm_model_file_path}'. Chatbot cannot function.")
        print(f"Error details: {e}")
        exit()

    # --- Step 4: Interactive Q&A Loop ---
    print("\n--- ATC RAG Chatbot (Self-Contained & Modular) Ready ---")
    print(f"Model: {os.path.basename(llm_model_file_path)}, Document: {os.path.basename(pdf_document_path)}")
    print("Type 'exit' or 'quit' to end.")
    
    try:
        while True:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting chatbot.")
                break
            if not user_query:
                continue

            print("Processing your query...")
            # answer_query_with_rag is imported from rag.py
            final_answer, retrieved_context_for_citation = answer_query_with_rag(
                query=user_query,
                llm_model_path=llm_model_file_path, # Passed for LLM init within RAG function if needed
                embedding_model_name_for_query=embedding_model_to_use,
                num_retrieved_chunks=3, 
                llm_max_new_tokens=750, # Increased slightly
                llm_context_window_size=llm_context_window_config,
                llm_gpu_layers_to_use=llm_gpu_layers_config
            )
            
            print("\nLLM Answer:")
            print(final_answer)

            # Optional: Display sources from the retrieved context
            if retrieved_context_for_citation and retrieved_context_for_citation.get('metadatas') and retrieved_context_for_citation['metadatas'][0]:
                print("\nSources considered for context:")
                seen_sources_display = set()
                for i, meta in enumerate(retrieved_context_for_citation['metadatas'][0]):
                    doc_id = meta.get('document_id', 'N/A')
                    page = meta.get('source_page', 'N/A')
                    # Create a unique key for display to avoid redundant listing of same page from multiple chunks
                    source_display_key = (doc_id, page) 
                    if source_display_key not in seen_sources_display:
                        content_snippet = retrieved_context_for_citation['documents'][0][i][:100].replace('\n', ' ') + "..."
                        print(f"  - From: Document '{doc_id}', Page: {page} (Content snippet: \"{content_snippet}\")")
                        seen_sources_display.add(source_display_key)
            else:
                print("(No specific context chunks were retrieved or an issue occurred during retrieval)")

    except KeyboardInterrupt:
        print("\nExiting chatbot due to user interruption.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the chat session: {e}")
    finally:
        print("Chatbot session ended.")
