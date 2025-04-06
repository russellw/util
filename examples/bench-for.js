const largeArray = Array(10000000).fill().map((_, i) => i);

// For loop - fastest
console.time('for');
for (let i = 0; i < largeArray.length; i++) {
  const value = largeArray[i];
  // Do something with i and value
}
console.timeEnd('for');

// forEach
console.time('forEach');
largeArray.forEach((value, i) => {
  // Do something with i and value
});
console.timeEnd('forEach');

// for...of with entries()
console.time('for...of');
for (const [i, value] of largeArray.entries()) {
  // Do something with i and value
}
console.timeEnd('for...of');

// map - slowest
console.time('map');
largeArray.map((value, i) => {
  // Do something with i and value
  return value;
});
console.timeEnd('map');
