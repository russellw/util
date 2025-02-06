// Compile with: clang -O3 -S -emit-llvm program.c
#include <stdio.h>

// Force inline this function
__attribute__((always_inline)) inline int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Constant expression that will be evaluated at compile time
constexpr int MAGIC_NUMBER = 42 * 314159;

// Function that uses compile-time constants
static inline int compute_value(void) {
    return MAGIC_NUMBER / 42;
}

// Function that will be inlined at call site
static inline int multiply_by_constant(int x) {
    return x * MAGIC_NUMBER;
}

int main() {
    // This will be computed at compile time
    int static_result = factorial(5);
    printf("Factorial(5) = %d\n", static_result);
    
    // This will inline multiply_by_constant
    int dynamic_result = multiply_by_constant(10);
    printf("10 * MAGIC_NUMBER = %d\n", dynamic_result);
    
    // This will be replaced with the constant 314159
    printf("Computed value = %d\n", compute_value());
    
    return 0;
}
