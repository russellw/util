use image::{ImageBuffer, Rgba};
use std::env;
use std::process;

fn parse_color(color_str: &str) -> Result<Rgba<u8>, &'static str> {
    let color_str = color_str.trim_start_matches('#'); // Allow optional # prefix.
    if color_str.len() != 6 {
        return Err("Color must be a 6-character hex string (e.g., #RRGGBB)");
    }
    let r = u8::from_str_radix(&color_str[0..2], 16).map_err(|_| "Invalid red value")?;
    let g = u8::from_str_radix(&color_str[2..4], 16).map_err(|_| "Invalid green value")?;
    let b = u8::from_str_radix(&color_str[4..6], 16).map_err(|_| "Invalid blue value")?;
    Ok(Rgba([r, g, b, 255])) // Default alpha to 255 (opaque)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!(
            "Usage: {} <output_file> <width> <height> <color>",
            args[0]
        );
        eprintln!("Example: {} output.png 800 600 #FF5733", args[0]);
        process::exit(1);
    }

    let output_file = &args[1];
    let width: u32 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Invalid width value");
        process::exit(1);
    });
    let height: u32 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Invalid height value");
        process::exit(1);
    });
    let color = parse_color(&args[4]).unwrap_or_else(|err| {
        eprintln!("Error parsing color: {}", err);
        process::exit(1);
    });

    // Create a blank image buffer
    let img = ImageBuffer::from_pixel(width, height, color);

    // Save the image as PNG
    match img.save(output_file) {
        Ok(_) => println!("Image saved to {}", output_file),
        Err(e) => {
            eprintln!("Error saving image: {}", e);
            process::exit(1);
        }
    }
}
