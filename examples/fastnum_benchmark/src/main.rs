use fastnum::decimal::Context;
use fastnum::{D256, dec256};
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;

fn main() {
	let mut t=dec256!(0);
	for i in 0..10000 {
			t+=D256::from_i32(i).unwrap().sin();
    }
    println!("{}",t);
}
