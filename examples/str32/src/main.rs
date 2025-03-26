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
    pub fn substr(&self, start: usize, end: usize) -> Result<Self, String> {
        if start > end {
            return Err("Start index must be less than or equal to end index".to_string());
        }
        if end > self.len() {
            return Err("End index out of range".to_string());
        }

        let slice = &self.v[start..end];
        let chars: Vec<char> = slice.to_vec();

        Ok(Self {
            v: chars.into_boxed_slice(),
        })
    }

    // Concatenates two Str32 instances
    pub fn concat(&self, other: &Self) -> Self {
        let mut result: Vec<char> = self.v.to_vec();
        result.extend(other.v.iter());

        Self {
            v: result.into_boxed_slice(),
        }
    }

    // Checks if the string contains a specific character
    pub fn contains(&self, c: char) -> bool {
        self.v.contains(&c)
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
    match s1.substr(0, 5) {
        Ok(sub) => println!("s1[0..5]: {}", sub),
        Err(e) => println!("Error: {}", e),
    }

    match s1.substr(7, 5) {
        Ok(sub) => println!("s1[7..5]: {}", sub),
        Err(e) => println!("Error: {}", e),
    }

    // Concatenation
    println!("\nConcatenation:");
    let s3 = s1.concat(&s2);
    println!("s1 + s2: {}", s3);

    // Character search
    println!("\nCharacter search:");
    println!("s1 contains 'o': {}", s1.contains('o'));
    println!("s1 contains 'z': {}", s1.contains('z'));

    // Equality check (using derived PartialEq)
    println!("\nEquality check:");
    let s1_copy = Str32::new("Hello, world!");
    println!("s1 == s1_copy: {}", s1 == s1_copy);
    println!("s1 == s2: {}", s1 == s2);
}
