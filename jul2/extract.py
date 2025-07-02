import fitz
import os

def extract_images_from_pdf(pdf_path: str, output_dir: str):
    """
    Extracts all images from a PDF and saves them to a specified directory.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    print(f"Processing '{pdf_path}' for image extraction...")

    image_count = 0
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page{page_num+1}_img{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_count += 1
            print(f"  -> Saved {image_path}")

    print(f"Image extraction complete. Found and saved {image_count} images.")
    doc.close()
    return image_count
