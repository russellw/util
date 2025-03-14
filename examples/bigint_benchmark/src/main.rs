use std::env;
use std::time::Instant;
use num_bigint::BigUint;
use num_traits::{One, Zero};

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    
    // Check if we have exactly one argument (plus the program name)
    if args.len() != 2 {
        eprintln!("Usage: {} <big_integer>", args[0]);
        std::process::exit(1);
    }
    
    // Parse the argument as a BigUint
    let n = match args[1].parse::<BigUint>() {
        Ok(num) => num,
        Err(_) => {
            eprintln!("Error: Could not parse '{}' as a positive integer", args[1]);
            std::process::exit(1);
        }
    };
    
    // Start timing
    let start = Instant::now();
    
    // Initialize sum to zero
    let mut sum = BigUint::zero();
    
    // Initialize counter to one
    let mut counter = BigUint::one();
    
    // Iterate from 1 to n, adding each number to the sum
    while counter <= n {
        sum = sum + &counter;
        counter = counter + BigUint::one();
    }
    
    // Calculate elapsed time
    let duration = start.elapsed();
    
    // Print the result
    println!("Sum of integers from 1 to {} is: {}", n, sum);
    println!("Time taken: {:?}", duration);
}