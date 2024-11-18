use resvg::usvg::{Options, TreeParsing};
use resvg::tiny_skia::{Pixmap, Transform};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input.svg> <width> [height]", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let width: u32 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Invalid width: {}", args[2]);
        std::process::exit(1);
    });

    let height: u32 = if args.len() > 3 {
        args[3].parse().unwrap_or_else(|_| {
            eprintln!("Invalid height: {}", args[3]);
            std::process::exit(1);
        })
    } else {
        width
    };

    let output_file = Path::new(input_file)
        .with_extension("png")
        .to_str()
        .unwrap()
        .to_string();

    // Load SVG file
    let svg_data = fs::read(input_file).unwrap_or_else(|err| {
        eprintln!("Failed to read file {}: {}", input_file, err);
        std::process::exit(1);
    });

    // Parse the SVG
    let options = Options::default();
    let tree = resvg::usvg::Tree::from_data(&svg_data, &options).unwrap_or_else(|err| {
        eprintln!("Failed to parse SVG: {}", err);
        std::process::exit(1);
    });

    // Create Pixmap
    let mut pixmap = Pixmap::new(width, height).unwrap_or_else(|| {
        eprintln!("Failed to create Pixmap with dimensions {}x{}", width, height);
        std::process::exit(1);
    });

    // Render SVG to Pixmap
    resvg::render(
        &tree,
        resvg::FitTo::Size(width, height),
        Transform::default(),
        pixmap.as_mut(),
    )
    .ok_or_else(|| {
        eprintln!("Failed to render SVG");
        std::process::exit(1);
    })
    .unwrap();

    // Save Pixmap as PNG
    pixmap.save_png(&output_file).unwrap_or_else(|err| {
        eprintln!("Failed to save PNG {}: {}", output_file, err);
        std::process::exit(1);
    });

    println!("Saved PNG to {}", output_file);
}
