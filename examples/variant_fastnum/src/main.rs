use fastnum::decimal::Context;
use fastnum::{dec256, D256};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Value {
    Number(D256),
    String(Rc<String>),
}

pub type EvalResult = Result<Value, String>;

const NO_TRAPS: Context = Context::default().without_traps();

impl Value {
    pub fn number(n: D256) -> Self {
        Value::Number(n)
    }

    pub fn string<S: Into<String>>(s: S) -> Self {
        Value::String(Rc::new(s.into()))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    pub fn as_number(&self) -> Option<D256> {
        match self {
            Value::Number(n) => Some(*n),
            Value::String(s) => s.parse::<D256>().ok(),
            _ => None,
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            Value::String(s) => s.to_string(),
            Value::Number(n) => n.to_string(),
        }
    }
}

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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(a) => write!(f, "{}", a),
            Value::String(s) => write!(f, "{}", s),
        }
    }
}

fn main() {
    println!("BASIC Interpreter Value Type Example");

    let num_val = Value::number(dec256!(42.0).with_ctx(NO_TRAPS));
    let str_val = Value::string("Hello, BASIC!");

    println!("Number: {:?}", num_val);
    println!("Number: {}", num_val);
    println!("String: {:?}", str_val);

    let ten = Value::number(dec256!(10.0).with_ctx(NO_TRAPS));
    let zero = Value::number(dec256!(0.0).with_ctx(NO_TRAPS));

    match ten / zero {
        Ok(result) => println!("Result: {:?}", result),
        Err(error) => println!("Error: {:?}", error),
    }
}
