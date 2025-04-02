from docx import Document
import sys

def convert_docx_to_txt(path):
    # Load the document
    doc = Document(path)
    
    # Extract text from the document
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # Join all text into a single string
    return '\n'.join(full_text)

# Path to your Word document
docx_path = sys.argv[1]
txt_content = convert_docx_to_txt(docx_path)
print(txt_content)
