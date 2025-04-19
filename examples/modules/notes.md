I'd be happy to show you how ES modules work when creating your own modules and importing between them. Let's create a simple Node.js program with multiple files that demonstrates module importing and exporting.

Let me walk you through a simple example of how to create and use your own ES modules in Node.js. I'll explain the key concepts as we go.

First, to use ES modules in Node.js, you'll need to:
1. Set `"type": "module"` in your package.json, or
2. Use the .mjs file extension for your module files

Here's how you can set up a project with multiple modules:

## Creating a Basic Project with ES Modules

Let's imagine we're building a small application with:
- A math module with utility functions
- A logger module that uses the math module
- A main entry point that uses both

### 1. Project Structure

```
project/
├── package.json
├── src/
│   ├── main.js        (Entry point)
│   ├── math.js        (Math utilities)
│   └── logger.js      (Logging utilities)
```

### 2. package.json

```json
{
  "name": "es-modules-demo",
  "version": "1.0.0",
  "type": "module",
  "main": "src/main.js"
}
```

### 3. The Math Module (math.js)

```javascript
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
```

### 4. The Logger Module (logger.js)

```javascript
// src/logger.js
// This module imports from math.js

import { add, PI } from './math.js';

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
```

### 5. Main Entry Point (main.js)

```javascript
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
```

## Running the Program

To run this program:
1. Create the files with the content above
2. Run: `node src/main.js`

## Key Concepts Demonstrated

1. **Named exports** (`export function add()`, `export const PI`)
2. **Default exports** (`export default calculator`)
3. **Importing named exports** (`import { add, PI } from './math.js'`)
4. **Importing default exports** (`import Logger from './logger.js'`)
5. **Importing both types** (`import calculator, { add } from './math.js'`)
6. **File extensions** (note that in ES modules, the `.js` extension is required)
7. **Module reuse** (math.js is imported by both logger.js and main.js)

Would you like me to explain any particular aspect of ES modules in more detail?