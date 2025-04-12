/**
 * Benchmark comparing different string concatenation methods in Node.js
 * 
 * This script tests four methods of string concatenation:
 * 1. Using += operator
 * 2. Using array.join()
 * 3. Using array.join() with pre-allocated array
 * 4. Using string literal (template strings)
 * 
 * Run with: node string-concatenation-benchmark.js
 */

function benchmarkStringConcatenation(iterations, stringSize) {
  console.log(`\nBenchmarking ${iterations.toLocaleString()} iterations with strings of size ~${stringSize} characters\n`);
  
  // Method 1: += operator
  const testPlusEquals = () => {
    let result = '';
    for (let i = 0; i < iterations; i++) {
      result += 'a';
    }
    return result;
  };

  // Method 2: Array join
  const testArrayJoin = () => {
    const arr = [];
    for (let i = 0; i < iterations; i++) {
      arr.push('a');
    }
    return arr.join('');
  };

  // Method 3: Pre-allocated array + join
  const testPreallocatedArrayJoin = () => {
    const arr = new Array(iterations);
    for (let i = 0; i < iterations; i++) {
      arr[i] = 'a';
    }
    return arr.join('');
  };

  // Method 4: Template literals (similar to += but different syntax)
  const testTemplateLiterals = () => {
    let result = '';
    for (let i = 0; i < iterations; i++) {
      result = `${result}a`;
    }
    return result;
  };

  // Function to run a benchmark
  const runBenchmark = (fn, name) => {
    // Warm up the JIT compiler
    fn();

    // Measure
    const start = process.hrtime.bigint();
    const result = fn();
    const end = process.hrtime.bigint();
    
    const timeInMs = Number(end - start) / 1_000_000;
    console.log(`${name.padEnd(25)} | Time: ${timeInMs.toFixed(2).padStart(10)} ms | Length: ${result.length}`);
    
    return timeInMs;
  };

  // Run benchmarks
  const results = {
    "+=": runBenchmark(testPlusEquals, "+= operator"),
    "array.join()": runBenchmark(testArrayJoin, "array.join()"),
    "preallocated array.join()": runBenchmark(testPreallocatedArrayJoin, "preallocated array.join()"),
    "template literals": runBenchmark(testTemplateLiterals, "template literals"),
  };

  // Find fastest method
  const fastest = Object.entries(results).reduce(
    (min, curr) => curr[1] < min[1] ? curr : min,
    ["", Infinity]
  );

  console.log(`\nFastest method: ${fastest[0]} (${fastest[1].toFixed(2)} ms)`);

  // Calculate relative performance
  console.log("\nRelative performance (lower is better):");
  Object.entries(results).forEach(([method, time]) => {
    const relative = time / fastest[1];
    console.log(`${method.padEnd(25)} | ${relative.toFixed(2)}x slower than fastest`);
  });
}

// Run benchmarks with different sizes
console.log("===== STRING CONCATENATION BENCHMARK =====");
console.log("Testing V8 optimization for string concatenation");

// Test with small strings (10K iterations)
benchmarkStringConcatenation(10_000, 10_000);

// Test with medium strings (100K iterations)
benchmarkStringConcatenation(100_000, 100_000);

// Test with large strings (1M iterations)
benchmarkStringConcatenation(1_000_000, 1_000_000);

// Optional: Test with very large strings (10M iterations)
// Uncomment if you have enough memory and want to stress test
// benchmarkStringConcatenation(10_000_000, 10_000_000);

console.log("\nNOTE: Results may vary based on Node.js version and V8 version");
console.log("Current Node.js version:", process.version);
console.log("For accurate results, run multiple times and check for consistency");