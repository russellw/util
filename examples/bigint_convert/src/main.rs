use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;

fn main() {
    // Create a truly massive integer
    let mut massive = BigInt::from(-1);
    for _ in 0..2000 {
        massive *= 2u32;
    }
    
    // Try to convert to f64
    //let float_val = massive.to_f64().unwrap_or(f64::INFINITY);
    let float_val = massive.to_f64().unwrap();
    println!("Result: {}", float_val); // Will be infinity
}