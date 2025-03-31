use std::collections::HashMap;
use std::time::{Duration, Instant};
use rand::{Rng, thread_rng};
use rand::distributions::Alphanumeric;

// Utility function to generate random strings
fn random_string(length: usize) -> String {
    let rng = thread_rng();
    rng.sample_iter(&Alphanumeric)
        .take(length)
        .map(char::from)
        .collect()
}

// Function to measure insertion performance
fn benchmark_insertion(num_entries: usize, key_length: usize) -> Duration {
    let mut map: HashMap<String, i32> = HashMap::with_capacity(num_entries);
    let start = Instant::now();
    
    for i in 0..num_entries {
        let key = random_string(key_length);
        map.insert(key, i as i32);
    }
    
    start.elapsed()
}

// Function to measure lookup performance
fn benchmark_lookup(map: &HashMap<String, i32>, keys: &[String]) -> Duration {
    let start = Instant::now();
    
    for key in keys {
        let _ = map.get(key);
    }
    
    start.elapsed()
}

// Function to measure removal performance
fn benchmark_removal(mut map: HashMap<String, i32>, keys: &[String]) -> Duration {
    let start = Instant::now();
    
    for key in keys {
        let _ = map.remove(key);
    }
    
    start.elapsed()
}

fn main() {
    // Benchmark parameters
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let key_lengths = [10, 20, 50, 100];
    
    // Print header
    println!("HashMap Performance Benchmark");
    println!("{:-<60}", "");
    
    // Run benchmarks for different map sizes
    for &size in &sizes {
        println!("\nBenchmarking with {} entries:", size);
        println!("{:-<60}", "");
        
        for &key_length in &key_lengths {
            println!("\nKey length: {} characters", key_length);
            
            // Insertion benchmark
            let insertion_time = benchmark_insertion(size, key_length);
            println!("Insertion time: {:?} ({:.2} ns/op)", 
                     insertion_time, 
                     insertion_time.as_nanos() as f64 / size as f64);
            
            // Prepare for lookup and removal benchmarks
            let mut map: HashMap<String, i32> = HashMap::with_capacity(size);
            let mut keys = Vec::with_capacity(size);
            
            // Insert entries and collect keys
            for i in 0..size {
                let key = random_string(key_length);
                keys.push(key.clone());
                map.insert(key, i as i32);
            }
            
            // Lookup benchmark
            let lookup_time = benchmark_lookup(&map, &keys);
            println!("Lookup time: {:?} ({:.2} ns/op)", 
                     lookup_time, 
                     lookup_time.as_nanos() as f64 / size as f64);
            
            // Removal benchmark
            let removal_time = benchmark_removal(map, &keys);
            println!("Removal time: {:?} ({:.2} ns/op)", 
                     removal_time, 
                     removal_time.as_nanos() as f64 / size as f64);
        }
    }
}