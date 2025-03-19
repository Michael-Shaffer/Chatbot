import pymupdf4llm

DATA_PATH = "../../data/data.md"

md_text = pymupdf4llm.to_markdown(DATA_PATH)
output = open(DATA_PATH, "w")
output.write(md_text)
output.close()