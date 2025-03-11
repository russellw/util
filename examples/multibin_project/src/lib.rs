// src/lib.rs
pub fn shared_function() -> &'static str {
    "Hello from the shared library!"
}

pub struct SharedStruct {
    pub value: i32,
}

impl SharedStruct {
    pub fn new(value: i32) -> Self {
        Self { value }
    }
    
    pub fn print_value(&self) {
        println!("Value: {}", self.value);
    }
}