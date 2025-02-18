#include <iostream>
#include <string>
#include <variant>
#include <unordered_map>

// Define our variant type that will be used as a key
using KeyVariant = std::variant<int, std::string, double>;

int main() {
    // Create an unordered_map with our variant as the key type
    // No need for custom hash or equality - std::variant handles this automatically
    std::unordered_map<KeyVariant, std::string> variant_map;
    
    // Insert different types of keys
    variant_map[42] = "integer key";
    variant_map[3.14] = "double key";
    variant_map[std::string("hello")] = "string key";
    
    // Test accessing elements
    std::cout << "Value for int key 42: " << variant_map[42] << std::endl;
    std::cout << "Value for double key 3.14: " << variant_map[3.14] << std::endl;
    std::cout << "Value for string key 'hello': " << variant_map[std::string("hello")] << std::endl;
    
    // Test finding elements
    auto it = variant_map.find(42);
    if (it != variant_map.end()) {
        std::cout << "\nFound value for key 42: " << it->second << std::endl;
    }
    
    // Test iteration
    std::cout << "\nAll key-value pairs:" << std::endl;
    for (const auto& [key, value] : variant_map) {
        std::cout << "Value: " << value << " for key type index: " << key.index() << std::endl;
    }
    
    // Test type checking and value extraction
    KeyVariant test_key = 42;
    if (std::holds_alternative<int>(test_key)) {
        std::cout << "\nKey holds an int with value: " << std::get<int>(test_key) << std::endl;
    }
    
    // Test modification
    variant_map[42] = "modified integer key";
    std::cout << "\nModified value for key 42: " << variant_map[42] << std::endl;
    
    // Test removal
    variant_map.erase(3.14);
    std::cout << "\nSize after removing 3.14: " << variant_map.size() << std::endl;
    
    return 0;
}