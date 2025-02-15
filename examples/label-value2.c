#include <stdio.h>

int main(void) {
    void *labels[] = { &&label1, &&label2, &&label3 };
    
    for(int i = 0; i < 3; i++) {
        printf("Going to label %d\n", i + 1);
        goto *labels[i];
    continue_loop:
        continue;
    }

    return 0;

label1:
    printf("At label 1\n");
    goto continue_loop;
label2:
    printf("At label 2\n");
    goto continue_loop;
label3:
    printf("At label 3\n");
    goto continue_loop;
}