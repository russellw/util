use fastnum::{dec256};

fn main() {
    let a = dec256!(10);
    let b = usize::try_from(a);
    println!("{:?}", b);

    let a = dec256!(-10);
    let b = usize::try_from(a);
    println!("{:?}", b);
}