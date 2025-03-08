use std::io::{self, Write};
use fastnum::{dec256, D256};
use fastnum::decimal::Context;
use std::str::FromStr;

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
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
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

        let result = process_input(&input);
        match result {
            Ok(value) => {
                println!("= {}", value);
                memory = Some(value);
            }
            Err(msg) => println!("Error: {}", msg),
        }
    }
}

fn calculate_transcendental(operation: Operation, value: D256) -> Result<D256, String> {
    // Convert D256 to f64 for transcendental operations
    let float_val = value.to_string().parse::<f64>()
        .map_err(|_| "Could not convert to floating point for transcendental operation".to_string())?;
    
    let result = match operation {
        Operation::Sin => float_val.sin(),
        Operation::Cos => float_val.cos(),
        Operation::Tan => float_val.tan(),
        Operation::Ln => {
            if float_val <= 0.0 {
                return Err("Logarithm of non-positive number is undefined".to_string());
            }
            float_val.ln()
        },
        Operation::Log10 => {
            if float_val <= 0.0 {
                return Err("Logarithm of non-positive number is undefined".to_string());
            }
            float_val.log10()
        },
        Operation::Exp => float_val.exp(),
        Operation::Sqrt => {
            if float_val < 0.0 {
                return Err("Square root of negative number is undefined in real domain".to_string());
            }
            float_val.sqrt()
        },
        _ => return Err("Invalid transcendental operation".to_string()),
    };
    
    // Convert back to D256
    D256::from_str(&result.to_string())
        .map_err(|_| "Error converting result back to D256".to_string())
}

fn process_input(input: &str) -> Result<D256, String> {
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
            return Err("Unknown function or invalid input format".to_string());
        }
        
        let value = D256::from_str(tokens[1])
            .map_err(|_| format!("Invalid number: {}", tokens[1]))?;
            
        return calculate_transcendental(operation, value);
    }
    
    // Regular binary operations
    if tokens.len() != 3 {
        return Err("Input must be in the format: <number> <operator> <number> or <function> <number>".to_string());
    }
    
    let left = D256::from_str(tokens[0],Context::default())
        .map_err(|_| format!("Invalid first number: {}", tokens[0]))?;
    
    let right = D256::from_str(tokens[2])
        .map_err(|_| format!("Invalid second number: {}", tokens[2]))?;
    
    let operation = match tokens[1] {
        "+" => Operation::Add,
        "-" => Operation::Subtract,
        "*" => Operation::Multiply,
        "/" => Operation::Divide,
        "^" => Operation::Power,
        _ => Operation::Invalid,
    };
    
    match operation {
        Operation::Add => Ok(left + right),
        Operation::Subtract => Ok(left - right),
        Operation::Multiply => Ok(left * right),
        Operation::Divide => {
            if right.is_zero() {
                Err("Division by zero".to_string())
            } else {
                // Set precision for division
                Ok(left.clone() / right.clone())
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
                
                Ok(result)
            } else {
                Err("Exponent must be a positive integer for power operations".to_string())
            }
        }
        Operation::Invalid => Err(format!("Unknown operator: {}", tokens[1])),
        // These operations shouldn't be reached here, but handle them for exhaustiveness
        Operation::Sin | Operation::Cos | Operation::Tan | 
        Operation::Ln | Operation::Log10 | Operation::Exp | Operation::Sqrt => {
            Err("Transcendental functions should be used with the format: <function> <number>".to_string())
        },
    }
}