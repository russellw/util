use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

// The core LispVal enum represents all possible Lisp values
#[derive(Clone)]
pub enum LispVal {
    Symbol(String),
    Number(f64),
    String(String),
    Bool(bool),
    List(Vec<LispVal>),
    Function(LispFunc),
    Nil,
}

// A LispFunc contains a function pointer and a closure environment
#[derive(Clone)]
pub struct LispFunc {
    params: Vec<String>,           // Parameter names
    body: Rc<LispVal>,             // Function body expression
    env: Rc<RefCell<Environment>>, // Captured lexical environment
    is_macro: bool,                // Whether this is a macro
}

// Environment implements lexical scoping
pub struct Environment {
    vars: HashMap<String, LispVal>,  // Current scope's bindings
    outer: Option<Rc<RefCell<Environment>>>, // Parent (outer) environment
}

impl Environment {
    // Create a new empty environment
    pub fn new() -> Self {
        Environment {
            vars: HashMap::new(),
            outer: None,
        }
    }
    
    // Create an environment with a given parent
    pub fn with_outer(outer: Rc<RefCell<Environment>>) -> Self {
        Environment {
            vars: HashMap::new(),
            outer: Some(outer),
        }
    }
    
    // Look up a variable in the environment, respecting lexical scope
    pub fn get(&self, key: &str) -> Option<LispVal> {
        match self.vars.get(key) {
            Some(val) => Some(val.clone()),
            None => match &self.outer {
                Some(outer) => outer.borrow().get(key),
                None => None,
            },
        }
    }
    
    // Set a variable in the current environment
    pub fn set(&mut self, key: String, val: LispVal) {
        self.vars.insert(key, val);
    }
    
    // Create a new environment for function calls with given bindings
    pub fn extend(&self, params: Vec<String>, args: Vec<LispVal>) -> Environment {
        let mut new_env = Environment::with_outer(Rc::new(RefCell::new(self.clone())));
        
        for (param, arg) in params.into_iter().zip(args.into_iter()) {
            new_env.set(param, arg);
        }
        
        new_env
    }
}

// Clone implementation for Environment
impl Clone for Environment {
    fn clone(&self) -> Self {
        let outer = match &self.outer {
            Some(env) => Some(Rc::clone(env)),
            None => None,
        };
        
        Environment {
            vars: self.vars.clone(),
            outer,
        }
    }
}

// Display implementation for LispVal
impl fmt::Display for LispVal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LispVal::Symbol(s) => write!(f, "{}", s),
            LispVal::Number(n) => write!(f, "{}", n),
            LispVal::String(s) => write!(f, "\"{}\"", s),
            LispVal::Bool(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            LispVal::List(list) => {
                write!(f, "(")?;
                for (i, val) in list.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, ")")
            },
            LispVal::Function(_) => write!(f, "#<function>"),
            LispVal::Nil => write!(f, "nil"),
        }
    }
}

// The core evaluation function (simplified)
pub fn eval(expr: &LispVal, env: Rc<RefCell<Environment>>) -> Result<LispVal, String> {
    match expr {
        // Symbol lookup - follows lexical scoping chain
        LispVal::Symbol(sym) => {
            env.borrow().get(sym).ok_or_else(|| format!("Undefined symbol: {}", sym))
        },
        
        // Self-evaluating expressions
        LispVal::Number(_) | LispVal::String(_) | LispVal::Bool(_) | LispVal::Nil => {
            Ok(expr.clone())
        },
        
        // List evaluation
        LispVal::List(list) => {
            if list.is_empty() {
                return Ok(LispVal::Nil);
            }
            
            // Special forms would be handled here (if, define, lambda, etc.)
            // For this example, we'll just show a simple function call
            
            let first = &list[0];
            let rest = &list[1..];
            
            // Evaluate the function position
            let func = eval(first, Rc::clone(&env))?;
            
            match func {
                LispVal::Function(func) => {
                    // Evaluate arguments (unless it's a macro)
                    let args = if func.is_macro {
                        rest.to_vec()
                    } else {
                        let mut evaluated_args = Vec::new();
                        for arg in rest {
                            evaluated_args.push(eval(arg, Rc::clone(&env))?);
                        }
                        evaluated_args
                    };
                    
                    // Create a new environment with the function's closed-over environment as parent
                    let new_env = Environment::with_outer(Rc::clone(&func.env));
                    let mut call_env = new_env;
                    
                    // Bind parameters to arguments in the new environment
                    for (param, arg) in func.params.iter().zip(args.iter()) {
                        call_env.set(param.clone(), arg.clone());
                    }
                    
                    // Evaluate the function body in the new environment
                    eval(&func.body, Rc::new(RefCell::new(call_env)))
                },
                _ => Err(format!("Not a function: {}", first)),
            }
        },
        
        // Function values evaluate to themselves
        LispVal::Function(_) => Ok(expr.clone()),
    }
}

// Parse and create a function definition
pub fn make_lambda(params: Vec<String>, body: LispVal, env: Rc<RefCell<Environment>>, is_macro: bool) -> LispVal {
    LispVal::Function(LispFunc {
        params,
        body: Rc::new(body),
        env: Rc::clone(&env),
        is_macro,
    })
}