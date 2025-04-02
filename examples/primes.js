/**
 * Node.js program to print all prime numbers less than 100
 */

/**
 * Function to check if a number is prime
 * @param {number} num - The number to check
 * @returns {boolean} - True if the number is prime, false otherwise
 */
function isPrime(num) {
  // 1 is not a prime number
  if (num <= 1) return false;
  
  // 2 and 3 are prime numbers
  if (num <= 3) return true;
  
  // If number is divisible by 2 or 3, it's not prime
  if (num % 2 === 0 || num % 3 === 0) return false;
  
  // Check if number is divisible by any number of form 6k±1
  // up to the square root of the number
  for (let i = 5; i * i <= num; i += 6) {
    if (num % i === 0 || num % (i + 2) === 0) {
      return false;
    }
  }
  
  return true;
}

/**
 * Function to print all prime numbers less than a given limit
 * @param {number} limit - The upper limit
 */
function printPrimes(limit) {
  console.log(`Prime numbers less than ${limit}:`);
  
  const primes = [];
  
  // Check each number up to the limit
  for (let i = 2; i < limit; i++) {
    if (isPrime(i)) {
      primes.push(i);
    }
  }
  
  // Print the result
  console.log(primes.join(', '));
  console.log(`Total: ${primes.length} prime numbers`);
}

// Execute the function with limit 100
printPrimes(100);