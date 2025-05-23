# rag.py
# Contains the RAG pipeline logic: prompt construction and answer generation.
# Last updated: 2025-05-22

from vectordb import retrieve_relevant_chunks # For getting context
from llm import get_llm_response_self_contained # For generating the answer

def construct_rag_prompt(user_query, retrieved_chunks_result_from_db):
    """
    Constructs the final prompt string for the LLM, including retrieved context.
    """
    context_str = "No relevant context found in the documents." # Default if no context

    if retrieved_chunks_result_from_db and retrieved_chunks_result_from_db.get('documents') and retrieved_chunks_result_from_db['documents'][0]:
        context_items = []
        for i, doc_content in enumerate(retrieved_chunks_result_from_db['documents'][0]):
            metadata = retrieved_chunks_result_from_db['metadatas'][0][i] if retrieved_chunks_result_from_db.get('metadatas') else {}
            source_info = f"Source Document: '{metadata.get('document_id', 'N/A')}', Page: {metadata.get('source_page', 'N/A')}, Type: {metadata.get('type', 'text')}"
            context_items.append(f"Context Chunk {i+1} ({source_info}):\n{doc_content}\n---")
        if context_items:
            context_str = "\n".join(context_items)
    
    # This is the prompt that will be sent to get_llm_response_self_contained (as prompt_text)
    # The LLM function will then wrap this with model-specific chat tokens.
    final_prompt_to_llm = f"""You are a specialized AI assistant for an Air Traffic Control company.
Your task is to answer questions based *solely* on the provided context from technical documents.
If the context does not contain the information needed to answer the question, clearly state that.
Do not use any external knowledge or make assumptions beyond what is in the context.
Be concise and precise in your answers. If possible, mention the source page or document.

Provided Context from Technical Documents:
{context_str}

User Question: {user_query}

Based *only* on the context above, what is the answer to the user's question?
Answer:"""
    return final_prompt_to_llm

def answer_query_with_rag(
    query, 
    llm_model_path, 
    embedding_model_name_for_query='all-MiniLM-L6-v2',
    num_retrieved_chunks=3, 
    llm_max_new_tokens=500,
    llm_context_window_size=4096, 
    llm_gpu_layers_to_use=-1
):
    """
    The main RAG pipeline function.
    1. Retrieves relevant chunks from the vector DB.
    2. Constructs a prompt with the query and context.
    3. Gets an answer from the LLM.
    """
    # print(f"\nProcessing RAG query: '{query}'")
    
    # 1. Retrieve relevant chunks
    # (retrieve_relevant_chunks uses the embedding model specified or its default)
    retrieved_context = retrieve_relevant_chunks(
        query_text=query, 
        n_results=num_retrieved_chunks,
        embedding_model_name=embedding_model_name_for_query 
    )

    # 2. Construct the prompt for the LLM
    # This prompt_for_llm is what will be the 'user' part of the chat with the LLM
    prompt_for_llm = construct_rag_prompt(query, retrieved_context)
    
    # 3. Get response from the self-contained LLM
    answer = get_llm_response_self_contained(
        prompt_text=prompt_for_llm, # This is the fully formed query + context
        llm_model_path_for_init=llm_model_path,
        max_new_tokens=llm_max_new_tokens,
        current_n_ctx=llm_context_window_size,
        current_n_gpu_layers=llm_gpu_layers_to_use
    )
    
    return answer, retrieved_context # Return answer and the context used (for citation/inspection)
