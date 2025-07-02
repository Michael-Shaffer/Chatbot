import os
import re
from dataclasses import dataclass, asdict
from typing import List, Literal, Dict, Any
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class RAGChunk:
    """A structured representation of a content chunk for our RAG pipeline."""
    chunk_id: int
    source_document: str
    content_type: Literal["text", "table", "image"]
    content: str
    metadata: Dict[str, Any]

def caption_image_with_vlm(image_path: str, processor, model, device: str) -> str:
    """Generates a descriptive caption for an image using the Llava VLM."""
    print(f"  -> Generating caption for {image_path} with Llava...")
    try:
        raw_image = Image.open(image_path).convert('RGB')
        prompt = "USER: <image>\nDescribe this technical diagram from an FAA manual in detail. Include all text, symbols, and their spatial relationships to explain the rule being illustrated.\nASSISTANT:"
        inputs = processor(prompt, raw_image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=200)
        caption = processor.decode(output[0], skip_special_tokens=True)
        assistant_response = caption.split("ASSISTANT:")[1].strip()
        print(f"  -> Generated Caption: '{assistant_response}'")
        return assistant_response
    except Exception as e:
        print(f"  -> Error captioning image {image_path}: {e}")
        return "Could not generate caption."

def create_smart_chunks(nougat_output_path: str, image_dir: str, vlm_processor, vlm_model, device: str) -> List[RAGChunk]:
    """Main function to load, parse, and chunk the document content."""
    print(f"Starting smart chunking for '{nougat_output_path}'...")
    
    try:
        with open(nougat_output_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except FileNotFoundError:
        print(f"Error: Nougat output file not found at '{nougat_output_path}'")
        return []

    table_pattern = re.compile(r"(\\begin{tabular}.*?\\end{tabular})", re.DOTALL)
    tables = table_pattern.findall(raw_content)
    text_without_tables = table_pattern.sub("[TABLE_PLACEHOLDER]", raw_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    text_chunks = text_splitter.split_text(text_without_tables)
    
    all_chunks = []
    chunk_id_counter = 0
    source_doc_name = os.path.basename(nougat_output_path)

    for text_chunk in text_chunks:
        all_chunks.append(RAGChunk(chunk_id_counter, source_doc_name, "text", text_chunk, {}))
        chunk_id_counter += 1
        
    for i, table_content in enumerate(tables):
        all_chunks.append(RAGChunk(chunk_id_counter, source_doc_name, "table", table_content, {"table_number": i + 1}))
        chunk_id_counter += 1

    image_files = [f for f in os.listdir(image_dir) if f.startswith('page') and f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in sorted(image_files):
        image_path = os.path.join(image_dir, image_file)
        image_caption = caption_image_with_vlm(image_path, vlm_processor, vlm_model, device)
        all_chunks.append(RAGChunk(chunk_id_counter, source_doc_name, "image", image_caption, {"image_path": image_path}))
        chunk_id_counter += 1

    print(f"Smart chunking complete. Generated {len(all_chunks)} chunks.")
    return all_chunks

