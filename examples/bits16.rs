use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("Usage: {} <num1> <num2>", args[0]);
        return;
    }

    let a: u16 = args[1].parse().expect("First argument must be a number");
    let b: u16 = args[2].parse().expect("Second argument must be a number");
    let c = a.wrapping_add(b);
    println!("{} + {} = {}", a, b, c);
}