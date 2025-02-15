#include <stdio.h>

int main(void) {
    void *labels[] = { &&label1, &&label2, &&label3 };
    
    // Test going to each label in sequence
    for(int i = 0; i < 3; i++) {
        printf("Going to label %d\n", i + 1);
        goto *labels[i];
    }

    printf("This line should never be reached\n");
    return 1;

label1:
    printf("At label 1\n");
    return 0;
label2:
    printf("At label 2\n");
    return 0;
label3:
    printf("At label 3\n");
    return 0;
}