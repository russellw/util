use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};
use std::rc::Rc;
use fastnum::{dec256, D256};
use fastnum::decimal::Context;

// Define error types separately from values
#[derive(Clone, Debug, PartialEq)]
pub enum InterpreterError {
    DivisionByZero,
    TypeError(String),
    IndexOutOfBounds,
    Overflow,
    UndefinedVariable(String),
    SyntaxError(String),
    // Add other error types as needed
}

// Value doesn't include errors
#[derive(Clone, Debug)]
pub enum Value {
    Number(D256),
    String(Rc<String>),
    Boolean(bool),
    Array(Rc<HashMap<usize, Value>>),
    Null,
}

// Type alias for convenience
pub type EvalResult = Result<Value, InterpreterError>;

impl Value {
    // Create helpers remain the same
    pub fn number(n: D256) -> Self {
        Value::Number(n)
    }

    pub fn string<S: Into<String>>(s: S) -> Self {
        Value::String(Rc::new(s.into()))
    }

    // Type checking methods remain the same
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }
    
    // As do conversion methods
    pub fn as_number(&self) -> Option<D256> {
        match self {
            Value::Number(n) => Some(*n),
            Value::String(s) => s.parse::<D256>().ok(),
            Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    
    // Adding the as_string method that was missing
    pub fn as_string(&self) -> String {
        match self {
            Value::String(s) => s.to_string(),
            Value::Number(n) => n.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Array(_) => "[Array]".to_string(),
            Value::Null => "".to_string(),
        }
    }
}

// For expression evaluation using Result
impl Add for Value {
    type Output = EvalResult;

    fn add(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => {
                Ok(Value::Number(a + b))
            },
            (Value::String(a), Value::String(b)) => {
                let mut new_string = a.to_string();
                new_string.push_str(b);
                Ok(Value::String(Rc::new(new_string)))
            },
            // Other cases and coercion
            _ => {
                // Try numeric addition with coercion
                if let (Some(a), Some(b)) = (self.as_number(), other.as_number()) {
                    Ok(Value::Number(a + b))
                } else {
                    // Fall back to string concatenation
                    let mut result = self.as_string();
                    result.push_str(&other.as_string());
                    Ok(Value::String(Rc::new(result)))
                }
            }
        }
    }
}

// Division would properly handle errors
impl Div for Value {
    type Output = EvalResult;
    
    fn div(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => {
                if *b == 0.0 {
                    Err(InterpreterError::DivisionByZero)
                } else {
                    Ok(Value::Number(a / b))
                }
            },
            _ => {
                // Try numeric division with coercion
                if let (Some(a), Some(b)) = (self.as_number(), other.as_number()) {
                    if b == 0.0 {
                        Err(InterpreterError::DivisionByZero)
                    } else {
                        Ok(Value::Number(a / b))
                    }
                } else {
                    Err(InterpreterError::TypeError(
                        "Cannot divide these types".to_string()
                    ))
                }
            }
        }
    }
}

// Adding a simple main function to make the compiler happy
fn main() {
    println!("BASIC Interpreter Value Type Example");
    
    // Example usage:
    let num_val = Value::number(42.0);
    let str_val = Value::string("Hello, BASIC!");
    
    println!("Number: {:?}", num_val);
    println!("String: {:?}", str_val);
    
    // Example of error handling with division
    let ten = Value::number(10.0);
    let zero = Value::number(0.0);
    
    match ten/zero {
        Ok(result) => println!("Result: {:?}", result),
        Err(error) => println!("Error: {:?}", error)
    }
}