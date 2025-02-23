#include <stdio.h>
#include <stdint.h>

int main() {
    uint16_t a = 50000;
    uint16_t b = 50000;
    uint16_t result = a + b;
    
    // Print the values and calculation
    printf("First number:  %u (0x%04X)\n", a, a);
    printf("Second number: %u (0x%04X)\n", b, b);
    printf("Sum:          %u (0x%04X)\n", result, result);
    
    // Show the full computation including the overflow bits
    uint32_t full_result = (uint32_t)a + (uint32_t)b;
    printf("\nFull 32-bit result before truncation: %u (0x%08X)\n", full_result, full_result);
    
    // Show binary representation
    printf("\nBinary representation:\n");
    printf("First number:  0b");
    for(int i = 15; i >= 0; i--) {
        printf("%d", (a >> i) & 1);
    }
    printf("\nSecond number: 0b");
    for(int i = 15; i >= 0; i--) {
        printf("%d", (b >> i) & 1);
    }
    printf("\nResult:        0b");
    for(int i = 15; i >= 0; i--) {
        printf("%d", (result >> i) & 1);
    }
    printf("\n");
    
    return 0;
}