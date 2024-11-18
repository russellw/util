use image::{ImageFormat};

fn main() {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: {} <background_image> <overlay_image> <output_image> <x> <y>", args[0]);
        std::process::exit(1);
    }

    let background_image_path = &args[1];
    let overlay_image_path = &args[2];
    let output_image_path = &args[3];
    let x: u32 = args[4].parse().expect("X must be a positive integer");
    let y: u32 = args[5].parse().expect("Y must be a positive integer");

    // Load the background image
    let mut background_image = image::open(background_image_path)
        .expect("Failed to open background image")
        .into_rgba8();

    // Load the overlay image
    let overlay_image = image::open(overlay_image_path)
        .expect("Failed to open overlay image")
        .into_rgba8();

    // Get dimensions of both images
    let (bg_width, bg_height) = background_image.dimensions();
    let (overlay_width, overlay_height) = overlay_image.dimensions();

    // Ensure the overlay doesn't go out of bounds
    if x + overlay_width > bg_width || y + overlay_height > bg_height {
        eprintln!("Overlay image exceeds bounds of the background image at the given coordinates.");
        std::process::exit(1);
    }

    // Paste the overlay image onto the background image
    for j in 0..overlay_height {
        for i in 0..overlay_width {
            let pixel = overlay_image.get_pixel(i, j);
            // Blend only if the pixel is not completely transparent
            if pixel[3] > 0 {
                background_image.put_pixel(x + i, y + j, *pixel); // Fix: Dereference `pixel`
            }
        }
    }

    // Save the output image
    background_image
        .save_with_format(output_image_path, ImageFormat::Png)
        .expect("Failed to save output image");
}
