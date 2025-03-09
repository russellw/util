use fastnum::decimal::Context;
use fastnum::{dec256, D256};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

// Value doesn't include errors
#[derive(Clone, Debug)]
pub enum Value {
    Number(D256),
    String(Rc<String>),
    Array(Rc<HashMap<usize, Value>>),
}

// Type alias for convenience
pub type EvalResult = Result<Value, String>;

const NO_TRAPS: Context = Context::default().without_traps();

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
            _ => None,
        }
    }

    // Adding the as_string method that was missing
    pub fn as_string(&self) -> String {
        match self {
            Value::String(s) => s.to_string(),
            Value::Number(n) => n.to_string(),
            Value::Array(_) => "[Array]".to_string(),
        }
    }
}

// For expression evaluation using Result
impl Add for Value {
    type Output = EvalResult;

    fn add(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.clone() + b.clone())),
            (Value::String(a), Value::String(b)) => {
                let mut new_string = a.to_string();
                new_string.push_str(b);
                Ok(Value::String(Rc::new(new_string)))
            }
            // Other cases and coercion
            _ => {
                // Try numeric addition with coercion
                if let (Some(a), Some(b)) = (self.as_number(), other.as_number()) {
                    Ok(Value::Number(a.clone() + b.clone()))
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
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.clone() / b.clone())),
            _ => {
                // Try numeric division with coercion
                if let (Some(a), Some(b)) = (self.as_number(), other.as_number()) {
                    Ok(Value::Number(a.clone() / b.clone()))
                } else {
                    Err("Cannot divide these types".to_string())
                }
            }
        }
    }
}

// Adding a simple main function to make the compiler happy
fn main() {
    println!("BASIC Interpreter Value Type Example");

    // Example usage:
    let num_val = Value::number(dec256!(42.0).with_ctx(NO_TRAPS));
    let str_val = Value::string("Hello, BASIC!");

    println!("Number: {:?}", num_val);
    println!("String: {:?}", str_val);

    // Example of error handling with division
    let ten = Value::number(dec256!(10.0).with_ctx(NO_TRAPS));
    let zero = Value::number(dec256!(0.0).with_ctx(NO_TRAPS));

    match ten / zero {
        Ok(result) => println!("Result: {:?}", result),
        Err(error) => println!("Error: {:?}", error),
    }
}
