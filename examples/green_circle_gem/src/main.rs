use image::{Rgb, RgbImage, ImageBuffer};
use imageproc::drawing::draw_filled_circle_mut;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // --- Configuration ---
    let img_width: u32 = 400;
    let img_height: u32 = 300;
    let background_color = Rgb([255u8, 255u8, 255u8]); // White background
    let circle_color = Rgb([0u8, 255u8, 0u8]);        // Green circle (R, G, B)
    let center_x = (img_width / 2) as i32;           // Center X coordinate
    let center_y = (img_height / 2) as i32;          // Center Y coordinate
    let radius = 50i32;                              // Circle radius
    let output_filename = "green_circle_output.png";

    // --- Image Creation ---
    // Create a new ImageBuffer with the specified dimensions and background color.
    // `ImageBuffer::from_pixel` fills the entire buffer with the provided pixel value.
    println!("Creating a {}x{} image buffer...", img_width, img_height);
    let mut img: RgbImage = ImageBuffer::from_pixel(img_width, img_height, background_color);

    // --- Drawing ---
    // Draw a filled circle onto the image buffer.
    // `draw_filled_circle_mut` modifies the image buffer in place (_mut).
    // Coordinates for drawing functions in imageproc are typically i32.
    println!("Drawing a green circle at ({}, {}) with radius {}...", center_x, center_y, radius);
    draw_filled_circle_mut(&mut img, (center_x, center_y), radius, circle_color);

    // --- Saving ---
    // Save the image buffer to a file.
    // The image format is inferred from the file extension (.png in this case).
    println!("Saving image to {}...", output_filename);
    img.save(output_filename)?; // The '?' propagates any potential saving errors

    println!("Successfully created and saved {}", output_filename);

    Ok(()) // Indicate success
}