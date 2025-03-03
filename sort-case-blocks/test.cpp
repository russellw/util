#include <iostream>
#include <string>

// Simple function with a basic switch statement
void basicSwitch(int value) {
	switch (value) {
	case 3:
		std::cout << "Three" << std::endl;
		break;
	case 1:
		std::cout << "One" << std::endl;
		break;
	case 2:
		std::cout << "Two" << std::endl;
		break;
	default:
		std::cout << "Other value" << std::endl;
		break;
	}
}

// Function with a more complex switch that has multi-cases and braced blocks
void complexSwitch(char op, int a, int b) {
	switch (op) {
	case 'z':
		// Single line case
		std::cout << "No operation" << std::endl;
		break;
	case '*':
	case 'x':
	case 'X': {
		// Case with braces
		int result = a * b;
		std::cout << "Multiplication: " << result << std::endl;
		break;
	}
	case '/':
		if (b == 0) {
			std::cout << "Error: Division by zero" << std::endl;
			return;
		}
		std::cout << "Division: " << (a / b) << std::endl;
		break;
	case '+':
		// Simple addition
		std::cout << "Addition: " << (a + b) << std::endl;
		break;
	case '-':
	case '_': {
		// Another case with braces
		int result = a - b;
		std::cout << "Subtraction: " << result << std::endl;
		
		// Nested switch
		switch (result) {
		case 0:
			std::cout << "Result is zero" << std::endl;
			break;
		case 42:
			std::cout << "Result is the answer" << std::endl;
			break;
		case -1:
			std::cout << "Result is negative one" << std::endl;
			break;
		}
		
		break;
	}
	default:
		std::cout << "Unknown operator" << std::endl;
		break;
	}
}

// Function with a switch containing empty cases
void emptySwitch(int status) {
	switch (status) {
	case 200:
	case 201:
	case 202:
		std::cout << "Success" << std::endl;
		break;
	case 404:
		std::cout << "Not found" << std::endl;
		break;
	case 500:
		std::cout << "Server error" << std::endl;
		break;
	case 400:
	case 401:
	case 403:
		std::cout << "Client error" << std::endl;
		break;
	}
}

int main() {
	basicSwitch(2);
	complexSwitch('+', 10, 20);
	emptySwitch(404);
	
	// Another inline switch
	switch (42) {
	case 1:
		std::cout << "Not the answer" << std::endl;
		break;
	case 42:
		std::cout << "The answer" << std::endl;
		break;
	}
	
	return 0;
}