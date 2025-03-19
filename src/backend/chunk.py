from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image
import pytesseract
from PIL import Image as PILImage
import pandas as pd

def process_document(file_path):
    """Process document and extract text, tables, and figures"""
    elements = partition_pdf(file_path)
    processed_content = {
        'text': [],
        'tables': [],
        'figures': []
    }
    for element in elements:
        if isinstance(element, Table):
            # Convert table to pandas DataFrame and then to string
            table_data = pd.DataFrame(element.metadata.get('text_as_html'))
            processed_content['tables'].append(str(table_data))
        elif isinstance(element, Image):
            # Extract text from images using OCR
            try:
                image = PILImage.open(element.metadata.get('image_path'))
                image_text = pytesseract.image_to_string(image)
                processed_content['figures'].append({
                    'caption': element.metadata.get('caption', ''),
                    'text': image_text
                })
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            processed_content['text'].append(str(element))
    return processed_content


def load_pdf(DATA_PATH) -> list[Document]:
    processed_content = process_document(DATA_PATH)
    docs = []

    for text in processed_content['text']:
        docs.append(Document(page_content=text))

    for table in processed_content['tables']:
        docs.append(Document(page_content=f"[TABLE] {table}"))
    
    for figure in processed_content['figures']:
        figure_text = f"[FIGURE] Caption: {figure['caption']}\nContent: {figure['text']}"
        docs.append(Document(page_content=figure_text))
    
    return docs

def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
        is_separator_regex=False,
        strip_whitespace=True
    )
    
    chunks = []
    for doc in docs:
        if doc.page_content.startswith("[TABLE]") or doc.page_content.startswith("[FIGURE]"):
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))
    
    return chunks 
