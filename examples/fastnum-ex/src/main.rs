use fastnum::{udec256, UD256};

fn main() {
    let a = udec256!(0.1);
    let b = udec256!(0.2);

    assert_eq!(a + b, udec256!(0.3));

    println!("Hello, world!");
    println!("{}", a);
}
