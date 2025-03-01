#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

// Forward declarations
namespace custom {

// Basic types for our LLVM IR implementation
class Type;
class Value;
class Function;
class GlobalVariable;
class BasicBlock;
class Instruction;
class Module;

// Type system
class Type {
public:
    enum TypeID {
        VoidTyID,
        IntegerTyID,
        FunctionTyID,
        PointerTyID,
        // Add other types as needed
    };
    
    Type(TypeID id) : id(id) {}
    virtual ~Type() = default;
    
    TypeID getTypeID() const { return id; }
    bool isVoidTy() const { return id == VoidTyID; }
    bool isIntegerTy() const { return id == IntegerTyID; }
    bool isFunctionTy() const { return id == FunctionTyID; }
    bool isPointerTy() const { return id == PointerTyID; }
    
    virtual bool isEqual(const Type* other) const {
        return id == other->id;
    }
    
private:
    TypeID id;
};

// Base class for all values in LLVM IR
class Value {
public:
    Value(Type* type, const std::string& name = "") : type(type), name(name) {}
    virtual ~Value() = default;
    
    Type* getType() const { return type; }
    const std::string& getName() const { return name; }
    void setName(const std::string& newName) { name = newName; }
    
private:
    Type* type;
    std::string name;
};

// Function declaration/definition
class Function : public Value {
public:
    Function(Type* returnType, const std::string& name, 
             std::vector<Type*> paramTypes)
        : Value(returnType, name), paramTypes(std::move(paramTypes)), 
          isDeclaration(true) {}
    
    bool isDeclarationOnly() const { return isDeclaration; }
    void setIsDefinition() { isDeclaration = false; }
    
    void addBasicBlock(std::unique_ptr<BasicBlock> BB) {
        basicBlocks.push_back(std::move(BB));
        isDeclaration = false;
    }
    
    const std::vector<Type*>& getParamTypes() const { return paramTypes; }
    
    bool isSignatureEqual(const Function* other) const {
        if (!type->isEqual(other->getType()))
            return false;
            
        const auto& otherParams = other->getParamTypes();
        if (paramTypes.size() != otherParams.size())
            return false;
            
        for (size_t i = 0; i < paramTypes.size(); i++) {
            if (!paramTypes[i]->isEqual(otherParams[i]))
                return false;
        }
        
        return true;
    }
    
private:
    std::vector<Type*> paramTypes;
    std::vector<std::unique_ptr<BasicBlock>> basicBlocks;
    bool isDeclaration;
};

// Global variable
class GlobalVariable : public Value {
public:
    GlobalVariable(Type* type, bool isConstant, const std::string& name)
        : Value(type, name), isConstant(isConstant), 
          initializer(nullptr) {}
    
    bool isConst() const { return isConstant; }
    Value* getInitializer() const { return initializer; }
    void setInitializer(Value* init) { initializer = init; }
    
private:
    bool isConstant;
    Value* initializer;
};

// Basic block (container for instructions)
class BasicBlock : public Value {
public:
    BasicBlock(const std::string& name, Function* parent)
        : Value(nullptr, name), parent(parent) {}
    
    void addInstruction(std::unique_ptr<Instruction> inst) {
        instructions.push_back(std::move(inst));
    }
    
    Function* getParent() const { return parent; }
    
private:
    Function* parent;
    std::vector<std::unique_ptr<Instruction>> instructions;
};

// Base class for all instructions
class Instruction : public Value {
public:
    Instruction(Type* type, const std::string& name, BasicBlock* parent)
        : Value(type, name), parent(parent) {}
    
    BasicBlock* getParent() const { return parent; }
    
private:
    BasicBlock* parent;
};

// Module (container for functions and global variables)
class Module {
public:
    Module(const std::string& name) : name(name) {}
    
    const std::string& getName() const { return name; }
    
    void addFunction(std::unique_ptr<Function> func) {
        functions[func->getName()] = std::move(func);
    }
    
    void addGlobalVariable(std::unique_ptr<GlobalVariable> global) {
        globals[global->getName()] = std::move(global);
    }
    
    Function* getFunction(const std::string& name) const {
        auto it = functions.find(name);
        return (it != functions.end()) ? it->second.get() : nullptr;
    }
    
    GlobalVariable* getGlobalVariable(const std::string& name) const {
        auto it = globals.find(name);
        return (it != globals.end()) ? it->second.get() : nullptr;
    }
    
    const std::unordered_map<std::string, std::unique_ptr<Function>>& getFunctions() const {
        return functions;
    }
    
    const std::unordered_map<std::string, std::unique_ptr<GlobalVariable>>& getGlobals() const {
        return globals;
    }
    
private:
    std::string name;
    std::unordered_map<std::string, std::unique_ptr<Function>> functions;
    std::unordered_map<std::string, std::unique_ptr<GlobalVariable>> globals;
};

// Linker error reporting
class LinkerError {
public:
    enum ErrorType {
        SymbolConflict,
        TypeMismatch,
        Other
    };
    
    LinkerError(ErrorType type, const std::string& message)
        : type(type), message(message) {}
    
    ErrorType getType() const { return type; }
    const std::string& getMessage() const { return message; }
    
private:
    ErrorType type;
    std::string message;
};

// Main linker function
std::unique_ptr<Module> linkModules(
    std::vector<std::unique_ptr<Module>>& modules,
    std::vector<LinkerError>& errors) {
    
    if (modules.empty()) {
        errors.emplace_back(LinkerError::Other, "No modules to link");
        return nullptr;
    }
    
    // Use the first module as our destination
    std::unique_ptr<Module> destModule = std::move(modules[0]);
    
    // Process each module
    for (size_t i = 1; i < modules.size(); i++) {
        Module* srcModule = modules[i].get();
        if (!srcModule) continue;
        
        // Link global variables
        for (const auto& [name, globalVar] : srcModule->getGlobals()) {
            GlobalVariable* destGlobal = destModule->getGlobalVariable(name);
            
            if (!destGlobal) {
                // Global doesn't exist in destination module, add it
                auto newGlobal = std::make_unique<GlobalVariable>(
                    globalVar->getType(),
                    globalVar->isConst(),
                    globalVar->getName()
                );
                
                if (globalVar->getInitializer()) {
                    // Note: In a real implementation, we would need to clone/map the initializer value
                    newGlobal->setInitializer(globalVar->getInitializer());
                }
                
                destModule->addGlobalVariable(std::move(newGlobal));
            } else {
                // Global already exists, check for conflicts
                if (!destGlobal->getType()->isEqual(globalVar->getType())) {
                    errors.emplace_back(
                        LinkerError::TypeMismatch,
                        "Global variable '" + name + "' has conflicting types"
                    );
                    continue;
                }
                
                // If the destination is a declaration and source has initializer, update it
                if (!destGlobal->getInitializer() && globalVar->getInitializer()) {
                    // Note: In a real implementation, we would need to clone/map the initializer value
                    destGlobal->setInitializer(globalVar->getInitializer());
                }
                // If both have initializers, it's a potential conflict (external linkage rules apply)
                else if (destGlobal->getInitializer() && globalVar->getInitializer()) {
                    // In a full implementation, we would check linkage types and external visibility
                    // For simplicity, we're just warning about the potential conflict
                    errors.emplace_back(
                        LinkerError::SymbolConflict,
                        "Global variable '" + name + "' has multiple initializers"
                    );
                }
            }
        }
        
        // Link functions
        for (const auto& [name, function] : srcModule->getFunctions()) {
            Function* destFunc = destModule->getFunction(name);
            
            if (!destFunc) {
                // Function doesn't exist in destination module, add it
                auto newFunc = std::make_unique<Function>(
                    function->getType(),
                    function->getName(),
                    function->getParamTypes()
                );
                
                if (!function->isDeclarationOnly()) {
                    // Note: In a real implementation, we would need to clone all basic blocks and instructions
                    newFunc->setIsDefinition();
                }
                
                destModule->addFunction(std::move(newFunc));
            } else {
                // Function already exists, check for conflicts
                if (!destFunc->isSignatureEqual(function.get())) {
                    errors.emplace_back(
                        LinkerError::TypeMismatch,
                        "Function '" + name + "' has conflicting signatures"
                    );
                    continue;
                }
                
                // If dest is declaration and source is definition, replace with definition
                if (destFunc->isDeclarationOnly() && !function->isDeclarationOnly()) {
                    // In a real implementation, we would need to clone all basic blocks and instructions
                    // Here we just mark it as a definition
                    destFunc->setIsDefinition();
                }
                // If both are definitions, it's a conflict (unless they are marked as weak, etc.)
                else if (!destFunc->isDeclarationOnly() && !function->isDeclarationOnly()) {
                    // In a full implementation, we would check linkage types
                    // For simplicity, we're just warning about the conflict
                    errors.emplace_back(
                        LinkerError::SymbolConflict,
                        "Function '" + name + "' has multiple definitions"
                    );
                }
            }
        }
    }
    
    return destModule;
}

// Helper function to print module information
void printModuleInfo(const Module* module) {
    if (!module) {
        std::cout << "Null module" << std::endl;
        return;
    }
    
    std::cout << "Module: " << module->getName() << std::endl;
    
    std::cout << "Global Variables:" << std::endl;
    for (const auto& [name, global] : module->getGlobals()) {
        std::cout << "  " << name;
        if (global->isConst()) std::cout << " (constant)";
        if (global->getInitializer()) std::cout << " (initialized)";
        std::cout << std::endl;
    }
    
    std::cout << "Functions:" << std::endl;
    for (const auto& [name, function] : module->getFunctions()) {
        std::cout << "  " << name;
        if (function->isDeclarationOnly()) std::cout << " (declaration)";
        else std::cout << " (definition)";
        std::cout << std::endl;
    }
}

} // namespace custom

// Example usage
void exampleUsage() {
    using namespace custom;
    
    // Create some basic types for our example
    auto voidTy = std::make_unique<Type>(Type::VoidTyID);
    auto int32Ty = std::make_unique<Type>(Type::IntegerTyID);
    auto funcTy = std::make_unique<Type>(Type::FunctionTyID);
    auto ptrTy = std::make_unique<Type>(Type::PointerTyID);
    
    // Create modules to link
    auto module1 = std::make_unique<Module>("module1");
    auto module2 = std::make_unique<Module>("module2");
    
    // Add a global variable to module 1
    auto global1 = std::make_unique<GlobalVariable>(int32Ty.get(), false, "globalVar");
    module1->addGlobalVariable(std::move(global1));
    
    // Add a function declaration to module 1
    std::vector<Type*> paramTypes1 = {int32Ty.get(), int32Ty.get()};
    auto func1 = std::make_unique<Function>(int32Ty.get(), "add", paramTypes1);
    module1->addFunction(std::move(func1));
    
    // Add a function definition to module 2
    std::vector<Type*> paramTypes2 = {int32Ty.get(), int32Ty.get()};
    auto func2 = std::make_unique<Function>(int32Ty.get(), "add", paramTypes2);
    func2->setIsDefinition();
    module2->addFunction(std::move(func2));
    
    // Add another unique function to module 2
    std::vector<Type*> paramTypes3 = {int32Ty.get()};
    auto func3 = std::make_unique<Function>(voidTy.get(), "print", paramTypes3);
    func3->setIsDefinition();
    module2->addFunction(std::move(func3));
    
    // Print info about original modules
    std::cout << "Original modules:" << std::endl;
    printModuleInfo(module1.get());
    std::cout << std::endl;
    printModuleInfo(module2.get());
    std::cout << std::endl;
    
    // Link the modules
    std::vector<std::unique_ptr<Module>> modules;
    modules.push_back(std::move(module1));
    modules.push_back(std::move(module2));
    
    std::vector<LinkerError> errors;
    auto linkedModule = linkModules(modules, errors);
    
    // Print any errors
    if (!errors.empty()) {
        std::cout << "Linking errors:" << std::endl;
        for (const auto& error : errors) {
            std::cout << "  " << error.getMessage() << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Print info about linked module
    std::cout << "Linked module:" << std::endl;
    printModuleInfo(linkedModule.get());
}

int main() {
    exampleUsage();
    return 0;
}