use std::fmt;
use std::rc::Rc;

// Remove PartialEq and Debug from the derive attributes for the Val enum
// We'll implement them manually
#[derive(Clone)]
pub enum Val {
    /// Floating-point value
    Float(f64),
    /// String value
    Str(Rc<String>),
    /// Function value that takes a Val and returns a Result with Val or error
    Func(Rc<dyn Fn(Val) -> Result<Val, String>>),
}

/// Trait for anything that can be converted into a Val::Func
pub trait IntoValFunction {
    fn into_val_function(self) -> Val;
}

/// Trait that all callable value functions must implement
pub trait ValCallable: fmt::Debug {
    /// Call the function with a list of arguments
    fn call(&self, args: &[Val]) -> Result<Val, String>;

    /// Return the expected number of arguments
    fn arity(&self) -> usize;
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

    /// Creates a new function value from any type that can be converted to a Val::Func.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that can be converted into a Val::Func
    ///
    /// # Returns
    ///
    /// A Val::Func containing the function
    pub fn func<F: IntoValFunction>(f: F) -> Self {
        f.into_val_function()
    }

    /// Applies the function to arguments if this value is a function.
    ///
    /// # Arguments
    ///
    /// * `args` - The arguments to pass to the function
    ///
    /// # Returns
    ///
    /// * `Ok(Val)` - The successful result of applying the function
    /// * `Err(String)` - If the function returned an error
    /// * `Err("Not a function")` - If this value is not a function
    pub fn apply(&self, args: &[Val]) -> Result<Val, String> {
        match self {
            Val::Func(f) => f.call(args),
            _ => Err("Not a function".to_string()),
        }
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

// Implement PartialEq manually for Val
impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Val::Float(a), Val::Float(b)) => a == b,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Func(a), Val::Func(b)) => Rc::ptr_eq(a, b), // Compare function pointers
            _ => false,
        }
    }
}

// Implement Debug manually for Val
impl fmt::Debug for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Float(a) => write!(f, "Float({:?})", a),
            Val::Str(s) => write!(f, "Str({:?})", s),
            Val::Func(_) => write!(f, "Func(<function>)"),
        }
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

// Implementation for a function with 1 parameter
impl<F> IntoValFunction for F
where
    F: Fn(Val) -> Result<Val, String> + 'static + fmt::Debug,
{
    fn into_val_function(self) -> Val {
        Val::Func(Rc::new(ValFunction1(self)))
    }
}

// Implementation for a function with 2 parameters
impl<F> IntoValFunction for F
where
    F: Fn(Val, Val) -> Result<Val, String> + 'static + fmt::Debug,
{
    fn into_val_function(self) -> Val {
        Val::Func(Rc::new(ValFunction2(self)))
    }
}

// Wrapper for a 1-parameter function
#[derive(Debug)]
struct ValFunction1<F>(F);

impl<F> ValCallable for ValFunction1<F>
where
    F: Fn(Val) -> Result<Val, String> + 'static + fmt::Debug,
{
    fn call(&self, args: &[Val]) -> Result<Val, String> {
        if args.len() != 1 {
            return Err(format!("Expected 1 argument, got {}", args.len()));
        }

        (self.0)(args[0].clone())
    }

    fn arity(&self) -> usize {
        1
    }
}

// Wrapper for a 2-parameter function
#[derive(Debug)]
struct ValFunction2<F>(F);

impl<F> ValCallable for ValFunction2<F>
where
    F: Fn(Val, Val) -> Result<Val, String> + 'static + fmt::Debug,
{
    fn call(&self, args: &[Val]) -> Result<Val, String> {
        if args.len() != 2 {
            return Err(format!("Expected 2 arguments, got {}", args.len()));
        }

        (self.0)(args[0].clone(), args[1].clone())
    }

    fn arity(&self) -> usize {
        2
    }
}

fn main() {
    let sqrt = Val::func(|x: Val| -> Result<Val, String> {
        match x {
            Val::Float(n) => Ok(Val::Float(n.sqrt())),
            _ => Err("Expected float".to_string()),
        }
    });
    let r = sqrt.apply(&[Val::Float(16.0)]).unwrap();
    println!("{:?}", r);
}
