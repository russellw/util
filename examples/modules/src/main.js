// src/main.js
// This imports from both modules

// Import default export from logger.js
import Logger from './logger.js';

// Import named exports
import { logMathOperation } from './logger.js';
import { multiply } from './math.js';

// Import both default and named exports from math.js
import calculator, { add } from './math.js';

// Create a logger instance
const logger = new Logger('MAIN: ');

// Use the imported modules
logger.log('Application started');

// Use function from logger module
logMathOperation();

// Use functions from math module
logger.log(`4 × 6 = ${multiply(4, 6)}`);
logger.log(`10 + 20 = ${add(10, 20)}`);

// Use the default export from math module
logger.log(`5² = ${calculator.square(5)}`);
logger.log(`3³ = ${calculator.cube(3)}`);

logger.log('Application finished');