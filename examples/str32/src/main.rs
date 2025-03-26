#[derive(PartialEq)]
pub struct Str32 {
    v: Box<[char]>,
}

impl Str32 {
    // Constructor for a new Str32 from a string slice
    pub fn new(s: &str) -> Self {
        let chars: Vec<char> = s.chars().collect();
        Self {
            v: chars.into_boxed_slice(),
        }
    }

    // Constructor from a String
    pub fn from_string(s: String) -> Self {
        let chars: Vec<char> = s.chars().collect();
        Self {
            v: chars.into_boxed_slice(),
        }
    }

    // Constructor from a single character
    pub fn from_char(c: char) -> Self {
        Self {
            v: vec![c].into_boxed_slice(),
        }
    }

    // Constructor from a Vec<char>
    pub fn from_vec(chars: Vec<char>) -> Self {
        Self {
            v: chars.into_boxed_slice(),
        }
    }

    // Returns the length of the string in characters
    pub fn len(&self) -> usize {
        self.v.len()
    }

    // Checks if the string is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Accesses a character at a specific index, returning an error if out of bounds
    pub fn at(&self, i: usize) -> Result<char, String> {
        if i < self.len() {
            Ok(self.v[i])
        } else {
            Err("Index out of range".to_string())
        }
    }

    // Converts the Str32 back to a regular String
    pub fn to_string(&self) -> String {
        self.v.iter().collect()
    }

    // Creates a substring from a range of indices
    pub fn substr(&self, i: usize, j: usize) -> Self {
        let slice = &self.v[i..j];
        let chars: Vec<char> = slice.to_vec();
        Self {
            v: chars.into_boxed_slice(),
        }
    }

    // Concatenates two Str32 instances
    pub fn add(&self, b: &Self) -> Self {
        let mut result: Vec<char> = self.v.to_vec();
        result.extend(b.v.iter());
        Self {
            v: result.into_boxed_slice(),
        }
    }

    // Returns a new Str32 with all uppercase characters
    pub fn upper(&self) -> Self {
        let r: Vec<char> = self
            .v
            .iter()
            .map(|c| c.to_uppercase().next().unwrap_or(*c))
            .collect();

        Self {
            v: r.into_boxed_slice(),
        }
    }

    // Returns a new Str32 with all lowercase characters
    pub fn lower(&self) -> Self {
        let r: Vec<char> = self
            .v
            .iter()
            .map(|c| c.to_lowercase().next().unwrap_or(*c))
            .collect();

        Self {
            v: r.into_boxed_slice(),
        }
    }
}

// Implementing Display trait to allow printing with {}
impl std::fmt::Display for Str32 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// Implementing Debug trait to allow printing with {:?}
impl std::fmt::Debug for Str32 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Str32({:?})", self.to_string())
    }
}

fn main() {
    // Creating new Str32 instances
    let s1 = Str32::new("Hello, world!");
    let s2 = Str32::new("Rust is awesome");
    let empty = Str32::new("");

    // Basic information
    println!("s1: {}", s1);
    println!("s2: {}", s2);
    println!("s1 length: {}", s1.len());
    println!("Is empty string empty? {}", empty.is_empty());
    println!("Is s1 empty? {}", s1.is_empty());

    // Character access
    println!("\nAccessing characters:");
    match s1.at(0) {
        Ok(c) => println!("First character of s1: {}", c),
        Err(e) => println!("Error: {}", e),
    }

    match s1.at(100) {
        Ok(c) => println!("Character at index 100: {}", c),
        Err(e) => println!("Error: {}", e),
    }

    // String manipulations
    println!("\nString manipulations:");
    println!("s1 uppercase: {}", s1.upper());
    println!("s1 lowercase: {}", s1.lower());

    // Substrings
    println!("\nSubstrings:");
    println!("s1[0..5]: {}", s1.substr(0, 5));

    // Concatenation
    println!("\nConcatenation:");
    let s3 = s1.add(&s2);
    println!("s1 + s2: {}", s3);

    // Equality check (using derived PartialEq)
    println!("\nEquality check:");
    let s1_copy = Str32::new("Hello, world!");
    println!("s1 == s1_copy: {}", s1 == s1_copy);
    println!("s1 == s2: {}", s1 == s2);

    // From String
    let string_str = Str32::from_string(String::from("From a String"));
    println!("From String: {}", string_str);

    // From char
    let char_str = Str32::from_char('A');
    println!("From char 'A': {}", char_str);

    // From Vec<char>
    let vec_chars = vec!['R', 'u', 's', 't'];
    let vec_str = Str32::from_vec(vec_chars);
    println!("From Vec<char>: {}", vec_str);
}
