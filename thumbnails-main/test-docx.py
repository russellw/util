from docx import Document
from PIL import Image, ImageDraw, ImageFont

def create_thumbnail(docx_path, thumbnail_path, width=200, height=100):
    # Load the .docx file
    doc = Document(docx_path)
    full_text = []
    
    # Extract text from the first few paragraphs
    for para in doc.paragraphs[:5]:  # Adjust the number of paragraphs as needed
        full_text.append(para.text)
    text = ' '.join(full_text)

    # Create a blank image with white background
    img = Image.new('RGB', (width, height), color = 'white')
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Load a default font

    # Draw the text onto the image
    d.text((10,10), text, fill=(0,0,0), font=font)

    # Save the image
    img.save(thumbnail_path)

# Example usage:
create_thumbnail('/t/a.docx', 'thumbnail.png')
