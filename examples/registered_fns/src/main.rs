use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    /// Floating-point value
    Float(f64),
    /// String value
    Str(Rc<String>),
    /// Function value that takes a Val and returns a Result with Val or error
    Func(Rc<dyn Fn(Val) -> Result<Val, String>>),
}

impl Val {
    /// Creates a new string value from any type that can be converted to a String.
    ///
    /// # Arguments
    ///
    /// * `s` - A value that can be converted into a String
    ///
    /// # Returns
    ///
    /// A Val::Str containing the string value
    pub fn string<S: Into<String>>(s: S) -> Self {
        Val::Str(Rc::new(s.into()))
    }

    /// Creates a new function value.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes a Val and returns a Result with Val or error
    ///
    /// # Returns
    ///
    /// A Val::Func containing the function
    pub fn func<F>(f: F) -> Self
    where
        F: Fn(Val) -> Result<Val, String> + 'static,
    {
        Val::Func(Rc::new(f))
    }

    /// Attempts to convert the value to an f64.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Val::Float(a) => Some(*a),
            Val::Str(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Applies the function to an argument if this value is a function.
    ///
    /// # Arguments
    ///
    /// * `arg` - The argument to pass to the function
    ///
    /// # Returns
    ///
    /// * `Ok(Val)` - The successful result of applying the function to the argument
    /// * `Err(String)` - If the function returned an error
    /// * `Err("Not a function")` - If this value is not a function
    pub fn apply(&self, arg: Val) -> Result<Val, String> {
        match self {
            Val::Func(f) => f(arg),
            _ => Err("Not a function".to_string()),
        }
    }
}

pub fn eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        // TODO: is this needed?
        (Val::Func(a), Val::Func(b)) => Rc::ptr_eq(a, b),
        _ => a == b,
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Float(a) => write!(f, "{}", a),
            Val::Str(s) => write!(f, "{}", s),
            Val::Func(_) => write!(f, "<fn>"),
        }
    }
}

fn main() {
    println!("Hello, world!");
}
