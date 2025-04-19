// src/math.js

// Named exports
export const PI = 3.14159;

export function add(a, b) {
  return a + b;
}

export function multiply(a, b) {
  return a * b;
}

// Default export
const calculator = {
  square: (x) => x * x,
  cube: (x) => x * x * x
};

export default calculator;