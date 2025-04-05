'use strict';

function f(){
}

console.log(f())

var a,m;
for (a in 'abc') {
console.log(a)
}

for (a of 'abc') {
console.log(a)
}

for (a of [30,40,50]) {
console.log(a)
}

m=new Map([ ['a',1], ['b',2] ])
console.log(m)

for (a of m) {
console.log(a)
}

console.log(665)
return 5
console.log(666)
