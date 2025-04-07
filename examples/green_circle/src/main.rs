use image::{ImageBuffer, Rgb};
use imageproc::drawing::draw_filled_circle;

fn main() {
    // Create a new RGB image with white background
    let width = 800;
    let height = 600;
    let mut img = ImageBuffer::from_fn(width, height, |_, _| {
        Rgb([255, 255, 255]) // White background
    });
    
    // Parameters for the circle
    let center_x = width / 2;
    let center_y = height / 2;
    let radius = 150;
    let green_color = Rgb([0, 255, 0]); // Green color (RGB)
    
    // Draw a filled green circle
    draw_filled_circle(&mut img, (center_x as i32, center_y as i32), radius, green_color);
    
    // Save the image to a file
    match img.save("green_circle.png") {
        Ok(_) => println!("Image saved successfully as 'green_circle.png'"),
        Err(e) => eprintln!("Failed to save image: {}", e),
    }
}