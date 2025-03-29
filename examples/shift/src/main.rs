fn main() {
    println!("Demonstrating bit shifting with negative amounts in Rust (i64)");
    println!("------------------------------------------------------------");

    // Test values
    let values = [1i64, -1i64, 42i64, -42i64];
    let shifts = [1i64, 2i64, -1i64, -2i64, 63i64, -63i64];
    
    for &value in &values {
        println!("\nStarting value: {} (binary: {:064b})", value, value);
        
        // Try left shifts
        for &shift in &shifts {
            match value.checked_shl(shift as u32) {
                Some(result) => println!("{} << {} = {} (binary: {:064b})", value, shift, result, result),
                None => println!("{} << {} = PANIC/ERROR (Rust prevents this at compile time)", value, shift),
            }
        }
        
        // Try right shifts
        for &shift in &shifts {
            match value.checked_shr(shift as u32) {
                Some(result) => println!("{} >> {} = {} (binary: {:064b})", value, shift, result, result),
                None => println!("{} >> {} = PANIC/ERROR (Rust prevents this at compile time)", value, shift),
            }
        }
    }
    
    // Now let's try to actually use negative shift values to show what happens
    println!("\n\nAttempting direct bit shifts with negative values:");
    println!("-----------------------------------------------");
    println!("Note: This will cause a compile-time error in Rust");
    
    // The following code will not compile, but it's here to demonstrate
    // what would trigger an error:
    
    // let value = 42i64;
    // let negative_shift = -2i64;
    // let result = value << negative_shift; // Compile error
    // let result = value >> negative_shift; // Compile error
    
    // What happens internally in Rust with negative shifts
    println!("\nRust behavior explanation:");
    println!("1. In Rust, bit shift operators (<<, >>) require the right operand to be unsigned.");
    println!("2. When you use an i64 as the shift amount, Rust tries to cast it to u32.");
    println!("3. Negative numbers can't be represented as unsigned values, causing a panic in debug mode.");
    println!("4. In release mode, the behavior is undefined according to the Rust specification.");
    println!("5. This is why Rust provides checked_shl and checked_shr methods to safely handle these cases.");
    
    // Demonstrate what happens with type casting
    println!("\nWhat actually happens during bit shifts with negative numbers:");
    
    for &shift in &[-1i64, -2i64] {
        let bits = shift.to_le_bytes();
        let as_u32 = u32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]);
        println!("When casting {} to u32, we get: {} (due to two's complement)", shift, as_u32);
        for &value in &[42i64] {
            // This is what would happen if Rust allowed the operation
            println!("{} << {} (interpreted as {}) = {:?}", 
                value, shift, as_u32, value.checked_shl(as_u32));
            println!("{} >> {} (interpreted as {}) = {:?}", 
                value, shift, as_u32, value.checked_shr(as_u32));
        }
    }
}