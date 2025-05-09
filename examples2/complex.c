#include <stdio.h>      /* Standard Library of Input and Output */
#include <complex.h>    /* Standard Library of Complex Numbers */

int main() {

    double _Complex z1 = 1.0 + 3.0 * I;
    double _Complex z2 = 1.0 - 4.0 * I;

    printf("Working with complex numbers:\n\v");

    printf("Starting values: Z1 = %.2f + %.2fi\tZ2 = %.2f %+.2fi\n", creal(z1), cimag(z1), creal(z2), cimag(z2));

    double _Complex sum = z1 + z2;
    printf("The sum: Z1 + Z2 = %.2f %+.2fi\n", creal(sum), cimag(sum));

    double _Complex difference = z1 - z2;
    printf("The difference: Z1 - Z2 = %.2f %+.2fi\n", creal(difference), cimag(difference));

    double _Complex product = z1 * z2;
    printf("The product: Z1 x Z2 = %.2f %+.2fi\n", creal(product), cimag(product));

    double _Complex quotient = z1 / z2;
    printf("The quotient: Z1 / Z2 = %.2f %+.2fi\n", creal(quotient), cimag(quotient));

    double _Complex conjugate = conj(z1);
    printf("The conjugate of Z1 = %.2f %+.2fi\n", creal(conjugate), cimag(conjugate));

    return 0;
}
