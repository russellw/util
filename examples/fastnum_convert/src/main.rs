use fastnum::{D256, dec256};
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;

fn main() {
    let a = dec256!(10);
    let b = a.to_u128();
    println!("{:?}", b);

    let n = 340_282_366_920_938_463_463_374_607_431_768_211_455u128;
    let z = D256::from_u128(n).unwrap();
    println!("{}", z);
}
