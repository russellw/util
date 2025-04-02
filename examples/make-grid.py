from PIL import Image, ImageDraw

def create_custom_grid(image_size, unit_size, outline_color, line_color, background_color, column_sizes, output_file):
    # Create a blank image with the background color
    img = Image.new('RGB', (image_size, image_size), background_color)
    draw = ImageDraw.Draw(img)

    # Set the thickness for the blue lines
    line_thickness = 2

    # Draw the horizontal blue line across the top
    draw.line((0, 0, image_size, 0), fill=line_color, width=line_thickness)

    # Calculate the x-position of each column and draw squares
    x_position = 0
    for i, column_size in enumerate(column_sizes):
        square_size = column_size * unit_size
        num_squares = image_size // square_size

        # Draw the squares in the current column
        for j in range(num_squares):
            top_left_x = x_position
            top_left_y = j * square_size + line_thickness  # Adjust for the horizontal line thickness
            bottom_right_x = top_left_x + square_size
            bottom_right_y = top_left_y + square_size
            draw.rectangle(
                [top_left_x, top_left_y, bottom_right_x, bottom_right_y],
                outline=outline_color
            )

        # Move to the next column position
        x_position += square_size

        # Draw a vertical blue line between columns of different sizes, except after the last column
        if i < len(column_sizes) - 1 and i%2==0:
            draw.line((x_position, 0, x_position, image_size), fill=line_color, width=line_thickness)

    # Save the image as a PNG file
    img.save(output_file)

# Parameters
image_size = 10000           # Size of the image (width and height)
unit_size = 100              # Size of the smallest square (in pixels)
gray=180
outline_color = (gray, gray, gray)  # Outline color (light gray) (R, G, B)
line_color = (0, 0, 140)      # Line color (blue) (R, G, B)
background_color = (255, 255, 255)  # Background color (white) (R, G, B)
column_sizes = [1, 1, 10, 10, 100, 100]  # Column sizes in units
output_file = 'grid.png'  # Output file name

# Create and save the custom grid image
create_custom_grid(image_size, unit_size, outline_color, line_color, background_color, column_sizes, output_file)
