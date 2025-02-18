#include <assert.h>

#include <string>
#include <variant>
#include <optional>

// GlobalRef represents a reference to an LLVM global value (function or global variable)
// It can store either a string name or an integer index
class GlobalRef {
public:
    // Constructors for different reference types
    explicit GlobalRef(const std::string& name) : value_(name) {}
    explicit GlobalRef(int64_t index) : value_(index) {}

    // Type checking
    bool isName() const { return std::holds_alternative<std::string>(value_); }
    bool isIndex() const { return std::holds_alternative<int64_t>(value_); }

    // Accessors with type safety
    std::optional<std::string> getName() const {
        if (const std::string* name = std::get_if<std::string>(&value_)) {
            return *name;
        }
        return std::nullopt;
    }

    std::optional<int64_t> getIndex() const {
        if (const int64_t* index = std::get_if<int64_t>(&value_)) {
            return *index;
        }
        return std::nullopt;
    }

    // Comparison operators
    bool operator==(const GlobalRef& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const GlobalRef& other) const {
        return !(*this == other);
    }

    // Hash support for use in unordered containers
    struct Hash {
        size_t operator()(const GlobalRef& ref) const {
            return std::visit([](const auto& v) {
                return std::hash<std::decay_t<decltype(v)>>{}(v);
            }, ref.value_);
        }
    };

private:
    std::variant<std::string, int64_t> value_;
};

// Example usage:
void example() {
    // Create references
    GlobalRef nameRef("main");
    GlobalRef indexRef(42);

    // Type checking
    assert(nameRef.isName());
    assert(indexRef.isIndex());

    // Safe access
    if (auto name = nameRef.getName()) {
        // Use *name
    }

    if (auto index = indexRef.getIndex()) {
        // Use *index
    }
}