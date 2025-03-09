use fastnum::decimal::Context;
use fastnum::udec256;

fn main() {
    let allow_divzero = Context::default().without_traps();

    let a = udec256!(1.5).with_ctx(allow_divzero);
    let b = udec256!(0.0).with_ctx(allow_divzero);

    let c = a / b;

    dbg!(c);
	
	let three=a+a;
    let q = three / b;
    dbg!(q);

    println!("{:?}", udec256!(1.5));
    println!("{:?}", a);
}