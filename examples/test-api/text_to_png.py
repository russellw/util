import textwrap

from PIL import Image, ImageDraw, ImageFont


def text_to_png(input_text, output_filename, font_path="arial.ttf", font_size=18):
    # Set the image width and height
    img_width, img_height = 800, 1000
    # Create a new image with white background
    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    # Initialize the drawing context
    draw = ImageDraw.Draw(image)

    # Try loading the specified font, or use a default if it fails
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Could not load font {font_path}, using default font.")
        font = ImageFont.load_default()

    # Initialize Y-coordinate to start drawing text
    y_text = 50

    # Prepare the text to fit the image width
    wrapped_text = textwrap.fill(input_text, width=70)
    # Draw the text on the image
    draw.text((50, 50), wrapped_text, fill=(0, 0, 0), font=font)

    # Save the image
    image.save(output_filename)


# Example usage:
# Read text from a file
with open("sample.txt", "r") as file:
    text_content = file.read()

# Convert the first page of text to PNG
text_to_png(text_content, "output.png")
