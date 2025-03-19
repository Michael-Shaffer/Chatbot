from document_chunker import load_pdf, split_docs
from pathlib import Path

DATA_PATH = "/Users/michaelshaffer/Work/Projects/Chat/Chatbot/docs/atc.pdf"

def main():
    file_extension = Path(DATA_PATH).suffix
    if file_extension == '.pdf':
        print(load_pdf(DATA_PATH))

if __name__ == "__main__":
    main()
