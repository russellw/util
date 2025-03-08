use std::io::{self, Write};
use bigdecimal::BigDecimal;
use bigdecimal::Zero;
use std::str::FromStr;

enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Invalid,
}

fn main() {
    println!("BigDecimal Desk Calculator");
    println!("Enter calculations in the format: <number> <operator> <number>");
    println!("Supported operators: + - * / ^");
    println!("Type 'quit' or 'exit' to end the program");
    println!("---------------------------------------");

    let mut memory: Option<BigDecimal> = None;

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

fn process_input(input: &str) -> Result<BigDecimal, String> {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    
    if tokens.len() != 3 {
        return Err("Input must be in the format: <number> <operator> <number>".to_string());
    }
    
    let left = BigDecimal::from_str(tokens[0])
        .map_err(|_| format!("Invalid first number: {}", tokens[0]))?;
    
    let right = BigDecimal::from_str(tokens[2])
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
            // For power operations with BigDecimal, we need to convert to u64
            // This is a limitation as BigDecimal doesn't support direct power operations
            if let Ok(exp) = right.to_string().parse::<u64>() {
                let mut result = BigDecimal::from(1);
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
    }
}