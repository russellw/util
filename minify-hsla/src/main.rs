use palette::{Hsla, LinSrgb};
use std::{env, fs, process};

fn main() {
    // Read the command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: minify-hsla <input-file>");
        process::exit(1);
    }

    let input_file = &args[1];
    let overwrite = args.contains(&String::from("-w"));

    // Read the input file content
    let content = fs::read_to_string(input_file).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", input_file, err);
        process::exit(1);
    });

    // Process the content
    let updated_content = content.replace(
        r"hsla\((\d+),\s*(\d+)%\s*,\s*(\d+)%\s*,\s*([\d\.]+)\)",
        |caps: &regex::Captures| {
            let h: f32 = caps[1].parse().unwrap_or(0.0);
            let s: f32 = caps[2].parse().unwrap_or(0.0);
            let l: f32 = caps[3].parse().unwrap_or(0.0);
            let a: f32 = caps[4].parse().unwrap_or(1.0);

            // Convert HSLA to RGBA
            let hsla = Hsla::new(h, s / 100.0, l / 100.0, a);
            let rgba: LinSrgb = hsla.into();

            // Convert RGBA to hex format
            let hex = format!(
                "#{:02x}{:02x}{:02x}{:02x}",
                (rgba.red * 255.0) as u8,
                (rgba.green * 255.0) as u8,
                (rgba.blue * 255.0) as u8,
                (rgba.alpha * 255.0) as u8
            );
            hex
        },
    );

    // Output the updated content
    if overwrite {
        fs::write(input_file, updated_content).unwrap_or_else(|err| {
            eprintln!("Error writing to file {}: {}", input_file, err);
            process::exit(1);
        });
    } else {
        println!("{}", updated_content);
    }
}
