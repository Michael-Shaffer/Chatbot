# chunk.py
# Contains functions for splitting extracted content into smaller chunks.
# Last updated: 2025-05-22

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_extracted_data(extracted_data_list, chunk_size=1000, chunk_overlap=150):
    """
    Chunks text content from the extracted data.
    Table and image content are treated as whole chunks for now.
    """
    final_chunks_with_metadata = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Adds 'start_index' to metadata
    )

    for item_id, item in enumerate(extracted_data_list):
        item_type = item.get("type", "unknown")
        content = item.get("content", "")
        # Preserve all original metadata from the item
        metadata = {k: v for k, v in item.items() if k != "content"}
        metadata["original_item_id"] = item_id # Add an ID for the original item

        if item_type == "text":
            if content:
                text_chunks = text_splitter.split_text(content)
                for chunk_index, chunk_text in enumerate(text_chunks):
                    chunk_metadata = metadata.copy() # Start with item's metadata
                    # Add chunk-specific metadata (start_index is added by splitter)
                    chunk_metadata["chunk_id_in_item"] = chunk_index 
                    final_chunks_with_metadata.append({
                        "page_content": chunk_text, # LangChain convention for content
                        "metadata": chunk_metadata
                    })
        elif item_type == "table" or item_type == "image":
            # Treat tables (as Markdown) and image descriptions as single chunks.
            # You might want to split very large tables further.
            if content:
                final_chunks_with_metadata.append({
                    "page_content": content,
                    "metadata": metadata 
                })
        else:
            # print(f"Warning: Unknown item type '{item_type}' for item ID {item_id}. Skipping.")
            pass
            
    return final_chunks_with_metadata
