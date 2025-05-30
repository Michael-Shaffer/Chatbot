# extract1.py
# Contains functions for extracting text, tables, and image references from PDFs.
# Last updated: 2025-05-22

import os
import io
import json
import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
from PIL import Image

def extract_tables_from_pdf(path_to_pdf, path_to_output_dir_for_pdf_assets) -> list:
    """
    Extracts tables from each page of a PDF and saves them as CSV and Markdown files.
    Returns a list of dictionaries, each representing a table.
    """
    tables_csv_output_dir = os.path.join(path_to_output_dir_for_pdf_assets, "tables")
    os.makedirs(tables_csv_output_dir, exist_ok=True)
    
    extracted_tables_info = []
    try:
        with pdfplumber.open(path_to_pdf) as pdf:
            for page_id, page in enumerate(pdf.pages):
                page_number = page_id + 1 # 1-based page number
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "explicit_vertical_lines": page.curves + page.edges,
                    "explicit_horizontal_lines": page.curves + page.edges,
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                    "edge_min_length": 10,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                    "text_tolerance": 5,
                    "intersection_tolerance": 5,
                })

                if not tables:
                    continue

                for table_id, table_data in enumerate(tables):
                    if not table_data or len(table_data) == 0:
                        continue

                    valid_table_data = [row for row in table_data if row is not None and any(cell is not None for cell in row)]
                    if len(valid_table_data) < 1:
                        continue
                    
                    header = valid_table_data[0]
                    data_rows_for_df = valid_table_data[1:]

                    if not header or all(h is None or str(h).strip() == "" for h in header):
                        # print(f"Warning: Page {page_number}, Table {table_id} has an invalid/empty header. Skipping.")
                        continue

                    df = pd.DataFrame(data_rows_for_df, columns=header)

                    cleaned_columns = []
                    column_counts = {}
                    for idx, col_name in enumerate(df.columns):
                        name = str(col_name).strip().replace('\n', ' ') if col_name else f"Unnamed_Col_{idx}"
                        if name in column_counts:
                            column_counts[name] += 1
                            name = f"{name}_{column_counts[name]}"
                        else:
                            column_counts[name] = 0
                        cleaned_columns.append(name)
                    df.columns = cleaned_columns
                    
                    for col_name in df.columns:
                        if df[col_name].dtype == 'object':
                            df[col_name] = df[col_name].astype(str).str.replace('\n', ' ', regex=False).str.strip()
                    
                    if df.empty and not any(str(h).strip() for h in header if h is not None):
                        # print(f"Warning: Page {page_number}, Table {table_id} is effectively empty. Skipping.")
                        continue

                    # Save CSV file
                    csv_file_name = f"page_{page_number}_table_{table_id}.csv"
                    csv_file_path = os.path.join(tables_csv_output_dir, csv_file_name)
                    df.to_csv(csv_file_path, index=False, encoding='utf-8')
                    
                    # Save Markdown file
                    md_file_name = f"page_{page_number}_table_{table_id}.md"
                    md_file_path = os.path.join(tables_csv_output_dir, md_file_name)
                    markdown_content = df.to_markdown(index=False)
                    with open(md_file_path, 'w', encoding='utf-8') as md_file:
                        md_file.write(markdown_content)
                    
                    extracted_tables_info.append({
                        "type": "table",
                        "source_page": page_number,
                        "table_id": table_id,
                        "csv_file_path": csv_file_path,
                        "md_file_path": md_file_path,
                        "content_description": f"Table data from page {page_number}, table {table_id}. Columns: {', '.join(df.columns.tolist())}",
                        "content": markdown_content # Serialize table to Markdown for RAG
                    })
    except Exception as e:
        print(f"Error extracting tables from {path_to_pdf}: {e}")
    return extracted_tables_info

def extract_images_from_pdf(path_to_pdf, path_to_output_dir_for_pdf_assets) -> list:
    """
    Extracts images from each page of a PDF and saves them.
    Returns a list of dictionaries, each representing an image.
    """
    images_output_dir = os.path.join(path_to_output_dir_for_pdf_assets, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    extracted_images_info = []
    try:
        doc = fitz.open(path_to_pdf)
        for page_index in range(len(doc)):
            page_number = page_index + 1 # 1-based page number
            page_obj = doc.load_page(page_index)
            image_list = page_obj.get_images(full=True)

            if not image_list:
                continue

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                if image_ext.lower() == 'jpx': image_ext = 'jp2'
                elif image_ext.lower() == 'jb2': image_ext = 'png'

                img_filename_base = f"page_{page_number}_img_{img_index}"
                image_filename = "" # Initialize
                
                common_formats = ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'tiff', 'jp2']
                if image_ext.lower() in common_formats:
                    image_filename = os.path.join(images_output_dir, f"{img_filename_base}.{image_ext}")
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                else:
                    try:
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        image_filename = os.path.join(images_output_dir, f"{img_filename_base}_converted.png")
                        pil_image.save(image_filename, "PNG")
                    except Exception as e:
                        # print(f"Could not convert image page {page_number} img {img_index} (ext {image_ext}): {e}")
                        continue 
                
                if image_filename: # Only append if image was successfully saved/converted
                    extracted_images_info.append({
                        "type": "image",
                        "source_page": page_number,
                        "image_id": img_index,
                        "file_path": image_filename,
                        "content": f"Reference to an image: Image {img_index} from page {page_number}. Path: {image_filename}"
                    })
        doc.close()
    except Exception as e:
        print(f"Error extracting images from {path_to_pdf}: {e}")
    return extracted_images_info

def extract_text_from_pdf(path_to_pdf) -> list:
    """
    Extracts text from each page of a PDF.
    Returns a list of dictionaries, each containing text and page number.
    """
    extracted_text_info = []
    try:
        doc = fitz.open(path_to_pdf)
        for page_index in range(len(doc)):
            page_number = page_index + 1 # 1-based page number
            page_obj = doc.load_page(page_index)
            text = page_obj.get_text("text")
            if text and text.strip():
                extracted_text_info.append({
                    "type": "text",
                    "source_page": page_number,
                    "content": text.strip()
                })
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {path_to_pdf}: {e}")
    return extracted_text_info

def process_pdf_for_rag(pdf_path, output_base_dir="rag_data"):
    """
    Processes a single PDF: extracts text, tables, and images.
    Saves assets and returns a consolidated list of extracted content items.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
    # Create a specific directory for this PDF's assets (tables, images)
    assets_dir_for_this_pdf = os.path.join(output_base_dir, pdf_filename_base, "assets")
    os.makedirs(assets_dir_for_this_pdf, exist_ok=True)
    
    print(f"Processing {pdf_path}...")
    all_extracted_data = []
    
    texts = extract_text_from_pdf(pdf_path)
    for text_data in texts:
        text_data["document_id"] = pdf_filename_base
        all_extracted_data.append(text_data)
    print(f"  Extracted {len(texts)} text segments.")

    tables_info = extract_tables_from_pdf(pdf_path, assets_dir_for_this_pdf)
    for table_data in tables_info:
        table_data["document_id"] = pdf_filename_base
        all_extracted_data.append(table_data)
    print(f"  Extracted {len(tables_info)} tables.")

    images_info = extract_images_from_pdf(pdf_path, assets_dir_for_this_pdf)
    for image_data in images_info:
        image_data["document_id"] = pdf_filename_base
        all_extracted_data.append(image_data)
    print(f"  Extracted {len(images_info)} image references.")
    
    # No JSON saving here, just return the list of dicts.
    # The main script will handle saving the final chunks with embeddings.
    return all_extracted_data

def main():
    """
    Main function for testing the PDF extraction functionality.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tables, text, and images from PDF files")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--output-dir", "-o", default="rag_data", 
                       help="Output directory for extracted data (default: rag_data)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found!")
        return
    
    print(f"Starting extraction from: {args.pdf_path}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    # Process the PDF
    extracted_data = process_pdf_for_rag(args.pdf_path, args.output_dir)
    
    print("-" * 50)
    print(f"Extraction complete! Total items extracted: {len(extracted_data)}")
    
    # Print summary
    text_count = sum(1 for item in extracted_data if item['type'] == 'text')
    table_count = sum(1 for item in extracted_data if item['type'] == 'table')
    image_count = sum(1 for item in extracted_data if item['type'] == 'image')
    
    print(f"  - Text segments: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images: {image_count}")
    
    if table_count > 0:
        print("\nTable details:")
        for item in extracted_data:
            if item['type'] == 'table':
                print(f"  Page {item['source_page']}, Table {item['table_id']}:")
                print(f"    CSV: {item['csv_file_path']}")
                print(f"    Markdown: {item['md_file_path']}")
    
    print(f"\nAll files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
