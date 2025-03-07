#include <iostream>
#include <iomanip>
#include <boost/multiprecision/cpp_dec_float.hpp>

// Define a decimal type with 50 digits of precision
using cpp_dec_float_50 = boost::multiprecision::cpp_dec_float_50;
using boost::multiprecision::cpp_dec_float;

// Useful aliases
template <unsigned Digits10>
using decimal = boost::multiprecision::number<cpp_dec_float<Digits10>>;

int main() {
    std::cout << "Boost Arbitrary Precision Decimal Arithmetic Demo\n";
    std::cout << "------------------------------------------------\n\n";
    
    // Set output formatting
    std::cout << std::setprecision(50) << std::scientific;
    
    // Create variables with 50 digits of precision
    cpp_dec_float_50 a("1.23456789012345678901234567890123456789012345678901");
    cpp_dec_float_50 b("9.87654321098765432109876543210987654321098765432109");
    
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n\n";
    
    // Basic arithmetic operations
    std::cout << "Basic Arithmetic Operations:\n";
    std::cout << "a + b = " << (a + b) << "\n";
    std::cout << "a - b = " << (a - b) << "\n";
    std::cout << "a * b = " << (a * b) << "\n";
    std::cout << "a / b = " << (a / b) << "\n\n";
    
    // Computing pi to high precision
    std::cout << "Computing π to high precision using the Chudnovsky algorithm:\n";
    
    // A simplified implementation of the Chudnovsky algorithm for π calculation
    // π = (426880 * sqrt(10005)) / (13591409 + sum(k=1 to inf) of M(k))
    // where M(k) = (6k)! * (13591409 + 545140134*k) / ((3k)! * (k!)^3 * (-640320)^3k)
    
    decimal<100> pi = 0;
    decimal<100> C = 426880 * sqrt(decimal<100>(10005));
    decimal<100> sum = 13591409;
    decimal<100> term = 13591409;
    
    // Compute a few terms of the series
    for (int k = 1; k <= 5; ++k) {
        // Calculate (6k)! / ((3k)! * (k!)^3)
        decimal<100> factorial_term = 1;
        for (int j = 3*k+1; j <= 6*k; ++j) {
            factorial_term *= j;
        }
        for (int j = 1; j <= k; ++j) {
            factorial_term /= (j*j*j);
        }
        
        // Calculate (-640320)^3k
        decimal<100> power_term = 1;
        decimal<100> base = -640320;
        for (int j = 1; j <= 3*k; ++j) {
            power_term *= base;
        }
        
        // Put it all together
        term = factorial_term * (13591409 + 545140134*k) / power_term;
        
        // Alternating series, so odd terms are negative
        if (k % 2 == 1) {
            sum -= term;
        } else {
            sum += term;
        }
    }
    
    pi = C / sum;
    
    std::cout << "π ≈ " << std::fixed << pi << "\n\n";
    
    // Higher precision examples
    std::cout << "Working with 1000 Digits of Precision:\n";
    using decimal_1000 = decimal<1000>;
    
    decimal_1000 e = 1;
    decimal_1000 factorial = 1;
    
    // Calculate e using the series: e = sum(1/n!)
    for (int i = 1; i <= 100; ++i) {
        factorial *= i;
        e += decimal_1000(1) / factorial;
    }
    
    std::cout << "e ≈ " << std::fixed << std::setprecision(50) << e << "\n";
    std::cout << "(Full 1000 digits not shown for clarity)\n\n";
    
    // Demonstrating exact decimal representation (unlike floating point)
    std::cout << "Exact Decimal Representation:\n";
    cpp_dec_float_50 exact_decimal("0.1");
    double floating_point = 0.1;
    
    std::cout << "0.1 as a double: " << std::setprecision(17) << floating_point << "\n";
    std::cout << "0.1 with arbitrary precision: " << exact_decimal << "\n\n";
    
    // Large number arithmetic without overflow
    std::cout << "Large Number Arithmetic:\n";
    cpp_dec_float_50 large1("1e40");  // 10^40
    cpp_dec_float_50 large2("1e39");  // 10^39
    
    std::cout << "10^40 + 10^39 = " << (large1 + large2) << "\n";
    std::cout << "10^40 * 10^39 = " << (large1 * large2) << "\n";
    
    return 0;
}