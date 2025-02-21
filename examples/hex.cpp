#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <sstream>
#include <string>

using boost::multiprecision::cpp_int;

int main() {
    // Method 1: Using stringstream
    std::stringstream ss1("0x123ABC");
    cpp_int value1;
    ss1 >> std::hex >> value1;
    std::cout << "Method 1 (stringstream):\n";
    std::cout << "0x123ABC = " << std::dec << value1 << "\n";
    std::cout << "In hex: 0x" << std::hex << value1 << "\n\n";

    // Method 2: Alternative stringstream
    cpp_int value2;
    std::stringstream ss2;
    ss2 << std::hex << "DEF456";  // without 0x prefix
    ss2 >> value2;
    std::cout << "Method 2 (alternative stringstream):\n";
    std::cout << "0xDEF456 = " << std::dec << value2 << "\n";
    std::cout << "In hex: 0x" << std::hex << value2 << "\n\n";

    // Demonstrating with some large values
    std::stringstream ss3("0xFFFFFFFFFFFFFFFF");
    cpp_int large1;
    ss3 >> std::hex >> large1;
    std::cout << "Large value 1:\n";
    std::cout << "0xFFFFFFFFFFFFFFFF = " << std::dec << large1 << "\n";

    std::stringstream ss4("0x123456789ABCDEF0");
    cpp_int large2;
    ss4 >> std::hex >> large2;
    std::cout << "\nLarge value 2:\n";
    std::cout << "0x123456789ABCDEF0 = " << std::dec << large2 << "\n";

    return 0;
}