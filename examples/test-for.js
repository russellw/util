'use strict';

var i,x;

var a=['a','b','c']
for ([i,x] of a.entries()){
	console.log(i);
	console.log(x);
}

var  m=new Map([ ['a','x'], ['b','y'] ]);
for ([i,x] of m.entries()){
	console.log(i);
	console.log(x);
}
