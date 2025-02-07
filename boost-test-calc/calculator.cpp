#include "calculator.hpp"
#include <sstream>
#include <stack>

double Calculator::evaluate(const std::string& expr) {
    std::istringstream iss(expr);
    std::stack<double> values;
    std::stack<char> ops;
    
    char ch;
    double num;
    
    while (iss >> ch) {
        if (std::isdigit(ch)) {
            iss.putback(ch);
            iss >> num;
            values.push(num);
        } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/') {
            ops.push(ch);
        }
    }
    
    while (!ops.empty()) {
        double b = values.top(); values.pop();
        double a = values.top(); values.pop();
        char op = ops.top(); ops.pop();
        
        switch (op) {
            case '+': values.push(add(a, b)); break;
            case '-': values.push(subtract(a, b)); break;
            case '*': values.push(multiply(a, b)); break;
            case '/': values.push(divide(a, b)); break;
        }
    }
    
    return values.top();
}
