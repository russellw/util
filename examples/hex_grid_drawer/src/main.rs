use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_polygon_mut, Point};
use std::f32::consts::PI;

// --- Configuration ---
const IMAGE_WIDTH: u32 = 800;
const IMAGE_HEIGHT: u32 = 600;
const GRID_COLS: u32 = 15; // Number of columns in the grid
const GRID_ROWS: u32 = 10; // Number of rows in the grid
const HEX_SIZE: f32 = 30.0; // Distance from center to a vertex
const LINE_COLOR: Rgb<u8> = Rgb([0u8, 0u8, 0u8]); // Black lines
const BACKGROUND_COLOR: Rgb<u8> = Rgb([255u8, 255u8, 255u8]); // White background
const OUTPUT_FILENAME: &str = "hexagonal_grid.png";
// --- End Configuration ---

// Helper function to calculate the vertices of a single hexagon
fn calculate_hex_vertices(center_x: f32, center_y: f32, size: f32) -> Vec<Point<i32>> {
    let mut vertices = Vec::with_capacity(6);
    for i in 0..6 {
        // Angle in radians. Add PI / 6.0 to start with a flat top,
        // or PI / 2.0 for pointy top. We'll use pointy top here.
        let angle_rad = PI / 3.0 * (i as f32) + PI / 2.0; // Start from the top point
        let x = center_x + size * angle_rad.cos();
        let y = center_y + size * angle_rad.sin();
        vertices.push(Point::new(x.round() as i32, y.round() as i32));
    }
    vertices
}

fn main() {
    println!("Generating hexagonal grid...");

    // Create a new white image buffer
    let mut img = RgbImage::from_pixel(IMAGE_WIDTH, IMAGE_HEIGHT, BACKGROUND_COLOR);

    // Calculate hex grid geometry (pointy-top orientation)
    let hex_width = 3.0f32.sqrt() * HEX_SIZE;
    let hex_height = 2.0 * HEX_SIZE;
    let vert_spacing = hex_height * 3.0 / 4.0; // Vertical distance between rows
    let horiz_spacing = hex_width;             // Horizontal distance between columns

    // Draw the grid
    for row in 0..GRID_ROWS {
        for col in 0..GRID_COLS {
            // Calculate the center of the current hexagon
            // Add an offset for every other row to make them interlock
            let x_offset = if row % 2 == 1 { horiz_spacing / 2.0 } else { 0.0 };
            let center_x = (col as f32 * horiz_spacing) + x_offset + hex_width / 2.0; // Add margin/padding if needed
            let center_y = (row as f32 * vert_spacing) + hex_height / 2.0; // Add margin/padding if needed

            // Ensure the hexagon is somewhat within the image bounds before drawing
            // (This is a basic check; more precise clipping could be added)
            if center_x + HEX_SIZE > 0.0 && center_x - HEX_SIZE < IMAGE_WIDTH as f32 &&
               center_y + HEX_SIZE > 0.0 && center_y - HEX_SIZE < IMAGE_HEIGHT as f32 {

                // Calculate the vertices for this hexagon
                let hex_points = calculate_hex_vertices(center_x, center_y, HEX_SIZE);

                // Draw the hexagon outline (polygon)
                // imageproc handles drawing lines between the points and closing the shape.
                draw_polygon_mut(&mut img, &hex_points, LINE_COLOR);

                /* // Alternative: Draw individual lines if you need more control
                for i in 0..6 {
                    let p1 = hex_points[i];
                    let p2 = hex_points[(i + 1) % 6]; // Wrap around for the last line
                    imageproc::drawing::draw_line_segment_mut(&mut img, (p1.x as f32, p1.y as f32), (p2.x as f32, p2.y as f32), LINE_COLOR);
                }
                */
            }
        }
    }

    // Save the image as a PNG file
    println!("Saving image to {}...", OUTPUT_FILENAME);
    match img.save(OUTPUT_FILENAME) {
        Ok(_) => println!("Successfully saved image!"),
        Err(e) => eprintln!("Error saving image: {}", e),
    }
}