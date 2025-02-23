#include <stdio.h>
#include <stdint.h>

// Define a type that can hold 19 bits (using a larger underlying type)
typedef struct {
    uint32_t value : 19;  // Bit field of 19 bits
} int19_t;

// Helper functions for operations
int19_t add19(int19_t a, int19_t b) {
    int19_t result = {.value = (a.value + b.value) & ((1U << 19) - 1)};
    return result;
}

int19_t multiply19(int19_t a, int19_t b) {
    int19_t result = {.value = (a.value * b.value) & ((1U << 19) - 1)};
    return result;
}

void print19(int19_t x) {
    printf("Value (19-bit): %u\n", x.value);
}

int main() {
    // Test values near the 19-bit limit (2^19 - 1 = 524287)
    int19_t a = {.value = 500000};
    int19_t b = {.value = 50000};
    
    printf("a = ");
    print19(a);
    printf("b = ");
    print19(b);
    
    // Test addition with overflow
    int19_t sum = add19(a, b);
    printf("a + b = ");
    print19(sum);
    
    // Test multiplication with overflow
    int19_t product = multiply19(a, b);
    printf("a * b = ");
    print19(product);
    
    // Demonstrate 19-bit overflow
    int19_t max = {.value = (1U << 19) - 1};
    int19_t one = {.value = 1};
    
    printf("\nMax 19-bit value: ");
    print19(max);
    
    printf("Max + 1 = ");
    print19(add19(max, one));
    
    return 0;
}