// Forward declarations
class BasicBlock;
class Function;
class Module;
class Value;
class Type;

// Represents a type in the IR
class Type {
public:
    enum TypeID {
        VoidTyID,
        IntegerTyID,
        FloatTyID,
        DoubleTyID,
        PointerTyID,
        ArrayTyID,
        FunctionTyID
    };

    TypeID getTypeID() const { return typeID; }
    virtual ~Type() = default;

protected:
    Type(TypeID id) : typeID(id) {}

private:
    const TypeID typeID;
};

// Represents a value that can be used in instructions
class Value {
public:
    virtual ~Value() = default;
    Type* getType() const { return type; }
    const std::string& getName() const { return name; }

protected:
    Value(Type* ty, const std::string& name = "") 
        : type(ty), name(name) {}

private:
    Type* type;
    std::string name;
};

// Represents a variable that can be assigned multiple times
class Variable : public Value {
public:
    Variable(Type* ty, const std::string& name) 
        : Value(ty, name) {}

    // Track the current defining instruction
    void setDefiningInstruction(Instruction* inst) {
        currentDef = inst;
    }

    Instruction* getDefiningInstruction() const {
        return currentDef;
    }

private:
    Instruction* currentDef = nullptr;
};

// Base class for all instructions
class Instruction : public Value {
public:
    enum OpCode {
        Add, Sub, Mul, Div,
        Load, Store,
        Branch, Call,
        Ret
        // Add more as needed
    };

    Instruction(OpCode op, Type* ty, const std::vector<Value*>& ops = {})
        : Value(ty), opcode(op), operands(ops) {}

    OpCode getOpcode() const { return opcode; }
    
    const std::vector<Value*>& getOperands() const { return operands; }
    
    // For instructions that define a variable
    void setDefinedVariable(Variable* var) {
        definedVar = var;
        if (var) {
            var->setDefiningInstruction(this);
        }
    }

    Variable* getDefinedVariable() const { return definedVar; }

    BasicBlock* getParent() const { return parent; }
    void setParent(BasicBlock* BB) { parent = BB; }

private:
    OpCode opcode;
    std::vector<Value*> operands;
    Variable* definedVar = nullptr;
    BasicBlock* parent = nullptr;
};

// Represents a basic block in the control flow graph
class BasicBlock {
public:
    using InstList = std::list<std::unique_ptr<Instruction>>;
    
    BasicBlock(const std::string& name = "") : name(name) {}

    const std::string& getName() const { return name; }
    
    // Instruction management
    void addInstruction(std::unique_ptr<Instruction> inst) {
        inst->setParent(this);
        instructions.push_back(std::move(inst));
    }

    // CFG management
    void addSuccessor(BasicBlock* succ) {
        successors.push_back(succ);
        succ->predecessors.push_back(this);
    }

    const std::vector<BasicBlock*>& getSuccessors() const { return successors; }
    const std::vector<BasicBlock*>& getPredecessors() const { return predecessors; }

    // Iterator access to instructions
    InstList::iterator begin() { return instructions.begin(); }
    InstList::iterator end() { return instructions.end(); }
    InstList::const_iterator begin() const { return instructions.begin(); }
    InstList::const_iterator end() const { return instructions.end(); }

private:
    std::string name;
    InstList instructions;
    std::vector<BasicBlock*> successors;
    std::vector<BasicBlock*> predecessors;
    Function* parent = nullptr;
};

// Represents a function in the module
class Function {
public:
    Function(Type* returnTy, const std::string& name,
            const std::vector<Variable*>& args = {})
        : returnType(returnTy), name(name), arguments(args) {}

    void addBasicBlock(std::unique_ptr<BasicBlock> BB) {
        basicBlocks.push_back(std::move(BB));
    }

    // Variable management
    Variable* createVariable(Type* ty, const std::string& name) {
        auto var = std::make_unique<Variable>(ty, name);
        auto varPtr = var.get();
        variables.push_back(std::move(var));
        return varPtr;
    }

    const std::vector<Variable*>& getArguments() const { return arguments; }
    Type* getReturnType() const { return returnType; }
    const std::string& getName() const { return name; }

private:
    Type* returnType;
    std::string name;
    std::vector<Variable*> arguments;
    std::vector<std::unique_ptr<Variable>> variables;
    std::vector<std::unique_ptr<BasicBlock>> basicBlocks;
};

// Top-level container for the entire program
class Module {
public:
    Module(const std::string& name = "") : name(name) {}

    void addFunction(std::unique_ptr<Function> F) {
        functions.push_back(std::move(F));
    }

    Function* getFunction(const std::string& name) {
        auto it = std::find_if(functions.begin(), functions.end(),
            [&](const auto& F) { return F->getName() == name; });
        return it != functions.end() ? it->get() : nullptr;
    }

private:
    std::string name;
    std::vector<std::unique_ptr<Function>> functions;
};
