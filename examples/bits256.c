typedef unsigned __int128 uint128_t;
typedef unsigned __attribute__((vector_size(32))) int uint256_t;  // 256 bits = 32 bytes

#include <stdio.h>

void print_256(uint256_t x) {
    // Access as an array of uint128_t
    uint128_t* parts = (uint128_t*)&x;
    
    // Print high bits first
    unsigned long long high_high = (unsigned long long)(parts[1] >> 64);
    unsigned long long high_low = (unsigned long long)parts[1];
    unsigned long long low_high = (unsigned long long)(parts[0] >> 64);
    unsigned long long low_low = (unsigned long long)parts[0];
    
    printf("0x%016llx %016llx %016llx %016llx\n", 
           high_high, high_low, low_high, low_low);
}

int main() {
    // Initialize values
    uint256_t a = {0};
    uint256_t b = {0};
    
    // Set high bit in a (2^255)
    uint128_t* a_parts = (uint128_t*)&a;
    a_parts[1] = ((uint128_t)1) << 127;
    
    // Set b to some value
    uint128_t* b_parts = (uint128_t*)&b;
    b_parts[0] = 1;  // Low part = 1
    b_parts[1] = ((uint128_t)1) << 64;  // High part = 2^64
    
    printf("a = ");
    print_256(a);
    printf("b = ");
    print_256(b);
    
    // Add them
    uint256_t sum = a + b;
    printf("sum = ");
    print_256(sum);
    
    return 0;
}