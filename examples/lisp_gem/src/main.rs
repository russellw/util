use std::io::{self, Write};
use std::collections::HashMap;

#[derive(Debug, Clone)]
enum LispValue {
    Number(f64),
    Symbol(String),
    List(Vec<LispValue>),
}

#[derive(Debug)]
enum LispError {
    InvalidInput,
    UndefinedSymbol(String),
    TypeMismatch,
    ArityMismatch,
    DivideByZero,
}

type LispResult = Result<LispValue, LispError>;

fn evaluate(value: &LispValue, env: &mut HashMap<String, LispValue>) -> LispResult {
    match value {
        LispValue::Number(_) => Ok(value.clone()),
        LispValue::Symbol(s) => env.get(s).cloned().ok_or(LispError::UndefinedSymbol(s.clone())),
        LispValue::List(list) => {
            if list.is_empty() {
                return Ok(LispValue::List(Vec::new()));
            }

            let head = evaluate(&list[0], env)?;
            let tail = &list[1..];

            match head {
                LispValue::Symbol(operator) => match operator.as_str() {
                    "+" => eval_arithmetic(tail, env, |a, b| a + b),
                    "-" => eval_arithmetic(tail, env, |a, b| a - b),
                    "*" => eval_arithmetic(tail, env, |a, b| a * b),
                    "/" => eval_arithmetic(tail, env, |a, b| {
                        if b == 0.0 {
                            Err(LispError::DivideByZero)
                        } else {
                            Ok(a / b)
                        }
                    }),
                    "define" => eval_define(tail, env),
                    _ => Err(LispError::UndefinedSymbol(operator)),
                },
                _ => Err(LispError::TypeMismatch),
            }
        }
    }
}

fn eval_arithmetic(
    args: &[LispValue],
    env: &mut HashMap<String, LispValue>,
    op: fn(f64, f64) -> Result<f64, LispError>,
) -> LispResult {
    if args.len() != 2 {
        return Err(LispError::ArityMismatch);
    }
    let a = evaluate(&args[0], env)?;
    let b = evaluate(&args[1], env)?;
    if let (LispValue::Number(num1), LispValue::Number(num2)) = (&a, &b) {
        op(*num1, *num2).map(LispValue::Number)
    } else {
        Err(LispError::TypeMismatch)
    }
}

fn eval_define(args: &[LispValue], env: &mut HashMap<String, LispValue>) -> LispResult {
    if args.len() != 2 {
        return Err(LispError::ArityMismatch);
    }
    if let LispValue::Symbol(name) = &args[0] {
        let value = evaluate(&args[1], env)?;
        env.insert(name.clone(), value);
        Ok(value)
    } else {
        Err(LispError::TypeMismatch)
    }
}

fn parse(input: &str) -> LispResult {
    let mut tokens = input.replace("(", " ( ").replace(")", " ) ").split_whitespace().collect::<Vec<&str>>();
    parse_list(&mut tokens)
}

fn parse_list(tokens: &mut Vec<&str>) -> LispResult {
    match tokens.first() {
        Some(&"(") => {
            tokens.remove(0); // Consume '('
            let mut list = Vec::new();
            while !tokens.is_empty() && tokens.first() != Some(&")") {
                list.push(parse_value(tokens)?);
            }
            if tokens.first() == Some(&")") {
                tokens.remove(0); // Consume ')'
                Ok(LispValue::List(list))
            } else {
                Err(LispError::InvalidInput)
            }
        }
        _ => parse_value(tokens),
    }
}

fn parse_value(tokens: &mut Vec<&str>) -> LispResult {
    if let Some(&token) = tokens.first() {
        tokens.remove(0);
        if let Ok(num) = token.parse::<f64>() {
            Ok(LispValue::Number(num))
        } else {
            Ok(LispValue::Symbol(token.to_string()))
        }
    } else {
        Err(LispError::InvalidInput)
    }
}

fn main() {
    let mut env: HashMap<String, LispValue> = HashMap::new();

    println!("Simple Lisp Interpreter in Rust");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "exit" {
            break;
        }

        match parse(input) {
            Ok(ast) => match evaluate(&ast, &mut env) {
                Ok(result) => println!("=> {:?}", result),
                Err(e) => eprintln!("Error: {:?}", e),
            },
            Err(e) => eprintln!("Parse Error: {:?}", e),
        }
    }
}