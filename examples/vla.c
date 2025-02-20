#include <stdio.h>
#include <stdlib.h>

// This function uses a VLA which will cause dynamic stack allocation
void process_dynamic_array(int size) {
    // This VLA declaration will cause the compiler to emit dynamic alloca
    int dynamic_array[size];
    
    // Initialize the array
    for (int i = 0; i < size; i++) {
        dynamic_array[i] = i * 2;
    }
    
    // Use the array to demonstrate it's not optimized away
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += dynamic_array[i];
    }
    
    printf("Sum of array elements: %d\n", sum);
}

// This function demonstrates nested VLAs
void process_2d_dynamic_array(int rows, int cols) {
    // 2D VLA will cause multiple dynamic allocations
    int matrix[rows][cols];
    
    // Initialize the 2D array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = i + j;
        }
    }
    
    // Use the array to prevent optimization
    int sum = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += matrix[i][j];
        }
    }
    
    printf("Sum of matrix elements: %d\n", sum);
}

int main(int argc, char *argv[]) {
    // Get size from command line to prevent compile-time optimization
    if (argc != 2) {
        printf("Usage: %s <size>\n", argv[0]);
        return 1;
    }
    
    int size = atoi(argv[1]);
    if (size <= 0) {
        printf("Please provide a positive number\n");
        return 1;
    }
    
    // Call functions with runtime-determined sizes
    process_dynamic_array(size);
    process_2d_dynamic_array(size, size + 1);
    
    return 0;
}