// src/bin/binary1.rs
use multibin_project::{shared_function, SharedStruct};

fn main() {
    println!("This is binary1");
    println!("{}", shared_function());
    
    let my_struct = SharedStruct::new(10);
    my_struct.print_value();
}