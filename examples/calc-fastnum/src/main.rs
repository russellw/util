use std::io::{self, Write};
use fastnum::{dec256, D256};
use fastnum::decimal::Context;
//use std::str::FromStr;

enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Sin,
    Cos,
    Tan,
    Ln,
    Log10,
    Exp,
    Sqrt,
    Invalid,
}

fn main() {
    println!("D256 Desk Calculator");
    println!("Enter calculations in the format: <number> <operator> <number>");
    println!("Supported operators: + - * / ^");
    println!("Transcendental functions: sin cos tan ln log10 exp sqrt");
    println!("Function usage: <function> <number>");
    println!("Type 'quit' or 'exit' to end the program");
    println!("---------------------------------------");

    let mut memory: Option<D256> = None;

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        
        let input = input.trim();
        if input.eq_ignore_ascii_case("q") || input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        // Handle memory recall with 'ans'
        let input = if input.contains("ans") && memory.is_some() {
            let mem_str = memory.as_ref().unwrap().to_string();
            input.replace("ans", &mem_str)
        } else {
            input.to_string()
        };

        let value = process_input(&input);
                println!("= {}", value);
                memory = Some(value);
    }
}

fn calculate_transcendental(operation: Operation, value: D256) -> D256 {
    match operation {
        Operation::Sin => value.sin(),
        Operation::Cos => value.cos(),
        Operation::Tan => value.tan(),
        Operation::Ln => {
            if value <= dec256!(0.0) {
                panic!("Logarithm of non-positive number is undefined");
            }
            value.ln()
        },
        Operation::Log10 => {
            if value <= dec256!(0.0) {
                panic!("Logarithm of non-positive number is undefined");
            }
            value.log10()
        },
        Operation::Exp => value.exp(),
        Operation::Sqrt => {
            if value < dec256!(0.0) {
                panic!("Square root of negative number is undefined in real domain");
            }
            value.sqrt()
        },
        _ => panic!("Invalid transcendental operation"),
    }
}

fn process_input(input: &str) -> D256 {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    
    // Check for function operations (sin, cos, etc.)
    if tokens.len() == 2 {
        let operation = match tokens[0] {
            "sin" => Operation::Sin,
            "cos" => Operation::Cos,
            "tan" => Operation::Tan,
            "ln" => Operation::Ln,
            "log10" => Operation::Log10,
            "exp" => Operation::Exp,
            "sqrt" => Operation::Sqrt,
            _ => Operation::Invalid,
        };
        
        if let Operation::Invalid = operation {
            panic!("Unknown function or invalid input format");
        }
        
        let value = D256::from_str(tokens[1],Context::default()).unwrap();
            
        return calculate_transcendental(operation, value);
    }
    
    // Regular binary operations
    if tokens.len() != 3 {
        panic!("Input must be in the format: <number> <operator> <number> or <function> <number>");
    }
    
    let left = D256::from_str(tokens[0],Context::default()).unwrap();
    
    let right = D256::from_str(tokens[2],Context::default()).unwrap();
    
    let operation = match tokens[1] {
        "+" => Operation::Add,
        "-" => Operation::Subtract,
        "*" => Operation::Multiply,
        "/" => Operation::Divide,
        "^" => Operation::Power,
        _ => Operation::Invalid,
    };
    
    match operation {
        Operation::Add => left + right,
        Operation::Subtract => left - right,
        Operation::Multiply => left * right,
        Operation::Divide => {
            if right.is_zero() {
                panic!("Division by zero")
            } else {
                // Set precision for division
                left.clone() / right.clone()
            }
        }
        Operation::Power => {
            // For power operations with D256, we need to convert to u64
            // This is a limitation as D256 doesn't support direct power operations
            if let Ok(exp) = right.to_string().parse::<u64>() {
                let mut result = D256::from(1);
                let base = left.clone();
                
                for _ in 0..exp {
                    result = result * base.clone();
                }
                
                result
            } else {
                panic!("Exponent must be a positive integer for power operations")
            }
        }
        Operation::Invalid => panic!("Unknown operator"),
        // These operations shouldn't be reached here, but handle them for exhaustiveness
        Operation::Sin | Operation::Cos | Operation::Tan | 
        Operation::Ln | Operation::Log10 | Operation::Exp | Operation::Sqrt => {
            panic!("Transcendental functions should be used with the format: <function> <number>")
        },
    }
}