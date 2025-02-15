#include <iostream>

class Complex {
private:
	double real;
	double imag;

public:
	Complex(double r = 0, double i = 0): real(r), imag(i) {
	}

	// Friend declaration for operator
	friend std::ostream& operator<<(std::ostream& os, const Complex& c);

	// Getters if needed
	double getReal() const {
		return real;
	}
	double getImag() const {
		return imag;
	}
};

// Implementation of operator<< (must be outside the class)
std::ostream& operator<<(std::ostream& os, const Complex& c) {
	os << c.real;
	if (c.imag >= 0) {
		os << "+";
	}
	os << c.imag << "i";
	return os;
}

int main() {
	Complex c1(3.0, 4.0);
	Complex c2(2.5, -1.5);

	std::cout << "c1 = " << c1 << std::endl;
	std::cout << "c2 = " << c2 << std::endl;

	return 0;
}
