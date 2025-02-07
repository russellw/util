#include <iostream>
#include <iomanip>

// Define 128-bit integer types using Clang's built-in support
using int128_t = __int128_t;
using uint128_t = __uint128_t;

// Helper function to print 128-bit integers
void print128(int128_t n) {
    if (n < 0) {
        std::cout << "-";
        n = -n;
    }
    if (n > 9) {
        print128(n / 10);
    }
    std::cout << (int)(n % 10);
}

void printHex128(uint128_t n) {
    std::cout << "0x" << std::hex;
    for (int i = 0; i < 32; i++) {
        std::cout << ((n >> (124 - 4 * i)) & 0xF);
    }
    std::cout << std::dec;
}

int main() {
    // Initialize 128-bit integers
    int128_t a = (int128_t)1 << 100;  // 2^100
    int128_t b = ((int128_t)1 << 99);  // 2^99
    
    std::cout << "Demonstrating 128-bit arithmetic:\n\n";
    
    // Addition
    std::cout << "2^100 = ";
    print128(a);
    std::cout << "\n2^99 = ";
    print128(b);
    std::cout << "\n\nAddition (2^100 + 2^99) = ";
    print128(a + b);
    
    // Multiplication
    int128_t mul = a * b;
    std::cout << "\n\nMultiplication (2^100 * 2^99) = ";
    print128(mul);
    
    // Division
    int128_t div = a / b;
    std::cout << "\n\nDivision (2^100 / 2^99) = ";
    print128(div);
    
    // Bitwise operations
    uint128_t x = (uint128_t)1 << 127;  // Highest bit set
    uint128_t y = ((uint128_t)1 << 126) | 1;  // Second highest bit and lowest bit set
    
    std::cout << "\n\nBitwise operations:";
    std::cout << "\nX = ";
    printHex128(x);
    std::cout << "\nY = ";
    printHex128(y);
    std::cout << "\nX & Y = ";
    printHex128(x & y);
    std::cout << "\nX | Y = ";
    printHex128(x | y);
    std::cout << "\nX ^ Y = ";
    printHex128(x ^ y);
    
    return 0;
}
