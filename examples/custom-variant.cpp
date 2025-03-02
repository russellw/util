#include <string>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <variant>
#include <iostream>

class Ref {
private:
    const int index1;
    const size_t num1 = 0;
    const std::string str1;

public:
    Ref(size_t num): index1(0), num1(num) {}
    Ref(const std::string& str): index1(1), str1(str) {}
    
    int index() const { return index1; }
    
    size_t num() const {
        if (index1 != 0) throw std::bad_variant_access();
        return num1;
    }
    
    std::string str() const {
        if (index1 != 1) throw std::bad_variant_access();
        return str1;
    }
    
    // To allow std::get to return references
    const size_t& get_num_ref() const {
        if (index1 != 0) throw std::bad_variant_access();
        return num1;
    }
    
    const std::string& get_str_ref() const {
        if (index1 != 1) throw std::bad_variant_access();
        return str1;
    }
};

// Specializations need to be in the std namespace
namespace std {
    // Define variant_size for our Ref class
    template<>
    struct variant_size<Ref> : std::integral_constant<size_t, 2> {};
    
    // Define variant_alternative for our Ref class
    template<size_t I>
    struct variant_alternative<I, Ref> {
        static_assert(I < 2, "Index out of bounds for Ref variant");
        using type = typename std::conditional<I == 0, size_t, std::string>::type;
    };
    
    // Implement std::get by index
    template<std::size_t I>
    constexpr auto& get(const Ref& ref) {
        if constexpr (I == 0) {
            return ref.get_num_ref();
        } else if constexpr (I == 1) {
            return ref.get_str_ref();
        } else {
            static_assert(I < 2, "Index out of bounds for Ref variant");
        }
    }
    
    // Implement std::get by type
    template<typename T>
    constexpr const T& get(const Ref& ref) {
        if constexpr (std::is_same_v<T, size_t>) {
            return get<0>(ref);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return get<1>(ref);
        } else {
            static_assert(std::is_same_v<T, size_t> || std::is_same_v<T, std::string>, 
                          "Type not in Ref variant");
            // This is never reached due to static_assert, but needed for compilation
            if constexpr (std::is_same_v<T, size_t>) {
                return ref.get_num_ref();
            } else {
                return ref.get_str_ref();
            }
        }
    }
    
    // Implement holds_alternative
    template<typename T>
    constexpr bool holds_alternative(const Ref& ref) noexcept {
        if constexpr (std::is_same_v<T, size_t>) {
            return ref.index() == 0;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return ref.index() == 1;
        } else {
            return false;
        }
    }
}

// Example usage
int main() {
    Ref v1(42);
    Ref v2(std::string("hello"));
    
    // Using std::get<index>
    std::cout << "v1 contains: " << std::get<0>(v1) << std::endl;
    std::cout << "v2 contains: " << std::get<1>(v2) << std::endl;
    
    // Using std::get<type>
    std::cout << "v1 contains: " << std::get<size_t>(v1) << std::endl;
    std::cout << "v2 contains: " << std::get<std::string>(v2) << std::endl;
    
    // Using std::holds_alternative
    std::cout << "v1 holds size_t: " << std::boolalpha << std::holds_alternative<size_t>(v1) << std::endl;
    std::cout << "v2 holds string: " << std::holds_alternative<std::string>(v2) << std::endl;
    
    // This would throw std::bad_variant_access
    try {
        std::cout << std::get<std::string>(v1) << std::endl;
    } catch (const std::bad_variant_access&) {
        std::cout << "Caught bad_variant_access as expected" << std::endl;
    }
    
    return 0;
}