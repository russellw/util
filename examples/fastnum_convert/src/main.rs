use fastnum::dec256;
use num_traits::ToPrimitive;

fn main() {
    let a = dec256!(10);
    let b = a.to_usize();
    println!("{:?}", b);
}