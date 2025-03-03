#include <iostream>

int main() {
	int value = 2;
	
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
	
	return 0;
}