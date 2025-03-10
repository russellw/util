use fastnum::decimal::Context;
use fastnum::{dec256, D256};
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
    pub fn number(a: D256) -> Self {
        Value::Number(a)
    }

    pub fn string<S: Into<String>>(s: S) -> Self {
        Value::String(Rc::new(s.into()))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    pub fn as_number(&self) -> Option<D256> {
        match self {
            Value::Number(a) => Some(*a),
            Value::String(s) => s.parse::<D256>().ok(),
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            Value::String(s) => s.to_string(),
            Value::Number(a) => a.to_string(),
        }
    }
}

impl Add for Value {
    type Output = EvalResult;

    fn add(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.clone() + b.clone())),
            _ => {
                    let mut result = self.as_string();
                    result.push_str(&other.as_string());
                    Ok(Value::String(Rc::new(result)))
                }
            }
        }
    }

impl Div for Value {
    type Output = EvalResult;

    fn div(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.clone() / b.clone())),
            _ => Err("/: expected numbers".to_string()),
        }
    }
}

impl Mul for Value {
    type Output = EvalResult;

    fn mul(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.clone() * b.clone())),
            _ => Err("*: expected numbers".to_string()),
        }
    }
}

impl Sub for Value {
    type Output = EvalResult;

    fn sub(self, other: Value) -> EvalResult {
        match (&self, &other) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.clone() - b.clone())),
            _ => Err("-: expected numbers".to_string()),
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
    let num_val = Value::number(dec256!(42).with_ctx(NO_TRAPS));
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
