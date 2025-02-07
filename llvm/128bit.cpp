#include <iostream>
#include <iomanip>

using int128_t = __int128_t;
using uint128_t = __uint128_t;

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
    const char hex_chars[] = "0123456789ABCDEF";
    std::cout << "0x";
    bool leading_zeros = true;
    
    for (int i = 15; i >= 0; i--) {
        uint8_t byte = (n >> (i * 8)) & 0xFF;
        uint8_t high = (byte >> 4) & 0xF;
        uint8_t low = byte & 0xF;
        
        if (high != 0 || !leading_zeros) {
            std::cout << hex_chars[high];
            leading_zeros = false;
        }
        if (low != 0 || !leading_zeros) {
            std::cout << hex_chars[low];
            leading_zeros = false;
        }
    }
    if (leading_zeros) std::cout << "0";
}

int main() {
    int128_t a = (int128_t)1 << 100;
    int128_t b = ((int128_t)1 << 99);
    
    std::cout << "2^100 = ";
    print128(a);
    std::cout << "\n2^99 = ";
    print128(b);
    std::cout << "\n\nAddition (2^100 + 2^99) = ";
    print128(a + b);
    
    int128_t mul = a * b;
    std::cout << "\n\nMultiplication (2^100 * 2^99) = ";
    print128(mul);
    
    int128_t div = a / b;
    std::cout << "\n\nDivision (2^100 / 2^99) = ";
    print128(div);
    
    uint128_t x = (uint128_t)1 << 127;
    uint128_t y = ((uint128_t)1 << 126) | 1;
    
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
    std::cout << "\n";
    
    return 0;
}
