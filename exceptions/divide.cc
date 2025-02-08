#include <iostream>
#include <stdexcept>

double divide(double numerator, double denominator) {
    if (denominator == 0) {
        throw std::runtime_error("Division by zero is not allowed");
    }
    return numerator / denominator;
}

int main() {
    try {
        // Test normal division
        std::cout << "10 / 2 = " << divide(10, 2) << std::endl;
        
        // Test division by zero
        std::cout << "10 / 0 = " << divide(10, 0) << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "Caught runtime error: " << e.what() << std::endl;
    }
    
    std::cout << "Program continues after error handling" << std::endl;
    return 0;
}
