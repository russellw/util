#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global variables to demonstrate different storage classes
static int static_var = 42;
volatile int volatile_var = 100;

// Structure to demonstrate aggregate types
struct Point {
    int x;
    int y;
    char label[20];
};

// Union to demonstrate type punning
union DataConverter {
    int i;
    float f;
    char bytes[4];
};

// Function pointer type
typedef int (*Operation)(int, int);

// Inline function to demonstrate function inlining
static inline int max1(int a, int b) {
    return (a > b) ? a : b;
}

// Recursive function to demonstrate call graph
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Function to demonstrate switch table generation
int switch_demo(int x) {
    switch (x) {
        case 0: return 100;
        case 1: return 200;
        case 2: return 300;
        case 3: return 400;
        case 4: return 500;
        default: return -1;
    }
}

// Function to demonstrate loop optimizations
void loop_demo(int* arr, int size) {
    // Loop unrolling candidate
    for (int i = 0; i < size; i++) {
        arr[i] = arr[i] * 2;
    }
    
    // Nested loops for loop interchange
    int matrix[10][10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            matrix[i][j] = i + j;
        }
    }
}

// Function to demonstrate SIMD vectorization
void vector_add(float* a, float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to demonstrate exception handling
void* allocate_or_die(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

// Binary operation functions for function pointer demo
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

int main() {
    // Local variables with different alignments
    char c = 'A';
    short s = 123;
    int i = 456;
    long l = 789L;
    float f = 3.14f;
    double d = 2.71828;
    
    // Array operations
    int arr[10];
    for (int i = 0; i < 10; i++) {
        arr[i] = i * i;
    }
    
    // Demonstrate structure operations
    struct Point p1 = {.x = 10, .y = 20, .label = "Point 1"};
    struct Point p2;
    memcpy(&p2, &p1, sizeof(struct Point));
    
    // Demonstrate union type punning
    union DataConverter conv;
    conv.f = 3.14f;
    printf("Float as int: %d\n", conv.i);
    
    // Function pointer array
    Operation ops[] = {add, subtract, multiply};
    int x = 10, y = 5;
    for (int i = 0; i < 3; i++) {
        printf("Operation %d result: %d\n", i, ops[i](x, y));
    }
    
    // Dynamic memory allocation
    int* dynamic_arr = (int*)allocate_or_die(sizeof(int) * 100);
    free(dynamic_arr);
    
    // Vector operations
    float va[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vb[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float vc[4];
    vector_add(va, vb, vc, 4);
    
    // Demonstrate recursive function
    printf("Fibonacci(10) = %d\n", fibonacci(10));
    
    // Demonstrate switch
    for (int i = 0; i < 6; i++) {
        printf("switch_demo(%d) = %d\n", i, switch_demo(i));
    }
    
    // Demonstrate loop optimizations
    loop_demo(arr, 10);
    
    return 0;
}