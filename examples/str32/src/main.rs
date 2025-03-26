#[derive(PartialEq)]
pub struct Str32 {
    v: Box<[char]>,
}

impl Str32 {
    pub fn new(s: &str) -> Self {
        let chars: Vec<char> = s.chars().collect();
        Self {
            v: chars.into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn at(&self, i: usize) -> Result<char, String> {
        if i < self.len() {
            Ok(self.v[i])
        } else {
            Err("Index out of range".to_string())
        }
    }
}

fn main() {
    println!("Hello, world!");
}
