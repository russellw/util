// src/bin/binary2.rs
use multibin_project::{shared_function, SharedStruct};

fn main() {
    println!("This is binary2");
    println!("{}", shared_function());
    
    let my_struct = SharedStruct::new(20);
    my_struct.print_value();
}