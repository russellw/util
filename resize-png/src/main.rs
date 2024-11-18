use std::env;
use std::fs::File;
use std::process;
use image::{GenericImageView, ImageOutputFormat};
use image::imageops::FilterType;

fn main() {
    // Collect command-line arguments
    let args: Vec<String> = env::args().collect();

    // Ensure correct number of arguments
    if args.len() != 4 {
        eprintln!("Usage: resize_png <input_file> <output_file> <scale_factor>");
        process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let scale_factor: f32 = match args[3].parse() {
        Ok(val) => val,
        Err(_) => {
            eprintln!("Error: Scale factor must be a valid floating-point number.");
            process::exit(1);
        }
    };

    // Load the input image
    let img = match image::open(input_file) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error loading image '{}': {}", input_file, e);
            process::exit(1);
        }
    };

    // Calculate new dimensions
    let (width, height) = img.dimensions();
    let new_width = (width as f32 * scale_factor) as u32;
    let new_height = (height as f32 * scale_factor) as u32;

    // Resize the image
    let resized_img = img.resize_exact(new_width, new_height, FilterType::Lanczos3);

    // Save the resized image to the output file
    let output_file_handle = match File::create(output_file) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating output file '{:?}': {}", output_file, e);
            process::exit(1);
        }
    };

    if let Err(e) = resized_img.write_to(&mut std::io::BufWriter::new(output_file_handle), ImageOutputFormat::Png) {
        eprintln!("Error writing to output file '{:?}': {}", output_file, e);
        process::exit(1);
    }

    println!("Image resized successfully: {} -> {}", input_file, output_file);
}
