#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>
#include <string>

using boost::multiprecision::uint128_t;

uint128_t calculate_factorial(unsigned int n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    
    uint128_t result = 1;
    for (unsigned int i = 2; i <= n; ++i) {
        result *= i;
        
        // Check for overflow
        if (result == 0) {
            throw std::overflow_error("Factorial result too large for 128-bit integer");
        }
    }
    
    return result;
}

void print_usage() {
    std::cout << "Usage: factorial <number>\n";
    std::cout << "Calculates the factorial of a given number using 128-bit arithmetic\n";
    std::cout << "Maximum supported input is 34 (larger values will overflow)\n";
}

int main(int argc, char* argv[]) {
    try {
        // Check command line arguments
        if (argc != 2) {
            print_usage();
            return 1;
        }

        // Parse input number
        int input;
        try {
            input = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid input. Please provide a valid number.\n";
            return 1;
        }

        // Validate input range
        if (input < 0) {
            std::cerr << "Error: Cannot calculate factorial of negative numbers.\n";
            return 1;
        }
        if (input > 34) {
            std::cerr << "Error: Input too large. Maximum supported value is 34.\n";
            return 1;
        }

        // Calculate and print the factorial
        uint128_t result = calculate_factorial(input);
        std::cout << input << "! = " << result << std::endl;

        return 0;
    }
    catch (const std::overflow_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
}