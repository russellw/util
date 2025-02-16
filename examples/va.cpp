#include <iostream>
#include <initializer_list>
#include <string>

// Function that takes any number of strings and prints them with numbers
void print_numbered(std::initializer_list<std::string> items) {
    int i = 1;
    for (const auto& item : items) {
        std::cout << i << ". " << item << '\n';
        i++;
    }
}

// Function that sums any number of integers
int sum(std::initializer_list<int> numbers) {
    int total = 0;
    for (int num : numbers) {
        total += num;
    }
    return total;
}

int main() {
    // Example with strings
    std::cout << "Shopping list:\n";
    print_numbered({"Apples", "Bananas", "Carrots", "Bread"});
    
    std::cout << "\nTo-do list:\n";
    print_numbered({"Wake up", "Drink coffee"});
    
    // Example with numbers
    int result1 = sum({1, 2, 3, 4, 5});
    std::cout << "\nSum of numbers 1-5: " << result1 << '\n';
    
    int result2 = sum({10, 20, 30});
    std::cout << "Sum of 10, 20, 30: " << result2 << '\n';
    
    // Can also pass zero arguments
    int result3 = sum({});
    std::cout << "Sum of empty list: " << result3 << '\n';
    
    return 0;
}