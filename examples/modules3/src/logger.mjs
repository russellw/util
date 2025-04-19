// src/logger.js
// This module imports from math.js

import { add, PI } from './math.mjs';

class Logger {
  constructor(prefix = '') {
    this.prefix = prefix;
  }
  
  log(message) {
    console.log(`${this.prefix}[LOG] ${message}`);
  }
  
  error(message) {
    console.error(`${this.prefix}[ERROR] ${message}`);
  }
}

// Using imported functions from math.js
export function logMathOperation() {
  const result = add(5, 7);
  console.log(`5 + 7 = ${result}`);
  console.log(`PI value: ${PI}`);
}

// Default export
export default Logger;
