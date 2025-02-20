#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>

// This function forces dynamic alloca by using a variable size
void process_dynamic_array(int size) {
    // alloca allocates 'size' bytes on the stack
    int *array = (int *)alloca(size * sizeof(int));
    
    // Initialize the array with some values
    for (int i = 0; i < size; i++) {
        array[i] = i * 2;
    }
    
    // Use the array to force the compiler to actually allocate it
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    
    printf("Sum of array elements: %d\n", sum);
}

int main(int argc, char *argv[]) {
    // Get size from command line or use default
    int size = (argc > 1) ? atoi(argv[1]) : 5;
    
    // This will force clang to use dynamic alloca
    // since the size isn't known at compile time
    process_dynamic_array(size);
    
    return 0;
}