import fitz  # This is the import name for PyMuPDF

def create_thumbnails(pdf_path, output_folder):
    # Open the provided PDF file
    doc = fitz.open(pdf_path)
    
    # Iterate through each page
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)  # Load the current page
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))  # Reduce the size to 50%
        
        # Create an output path for the thumbnail
        output_path = f"{output_folder}/thumbnail_{page_number + 1}.png"
        
        # Save the pixmap as a PNG
        pix.save(output_path)
        print(f"Thumbnail saved as: {output_path}")
    
    # Close the document
    doc.close()

# Example usage
if __name__ == "__main__":
    import os
    pdf_file = "/t/a.pdf"
    output_dir = "thumbnails"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    create_thumbnails(pdf_file, output_dir)
