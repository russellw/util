// cl /std:c++17 /EHsc variant.cpp
#include <iostream>
#include <variant>
#include <vector>
#include <string>
#include <cmath>

// Helper template for overloading lambdas
template<class... Ts> struct overloaded : Ts... { 
    using Ts::operator()...; 
};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// Shape classes
class Circle {
    double radius;
public:
    Circle(double r) : radius(r) {}
    double getRadius() const { return radius; }
};

class Rectangle {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double getWidth() const { return width; }
    double getHeight() const { return height; }
};

class Triangle {
    double base, height;
public:
    Triangle(double b, double h) : base(b), height(h) {}
    double getBase() const { return base; }
    double getHeight() const { return height; }
};

// Define the variant type that can hold any shape
using Shape = std::variant<Circle, Rectangle, Triangle>;

int main() {
    // Create a vector of different shapes
    std::vector<Shape> shapes = {
        Circle(5),
        Rectangle(4, 6),
        Triangle(3, 7),
        Circle(3)
    };

    // Calculate and print area for each shape
    auto calculateArea = overloaded {
        [](const Circle& c) { 
            return 3.14 * c.getRadius() * c.getRadius(); 
        },
        [](const Rectangle& r) { 
            return r.getWidth() * r.getHeight(); 
        },
        [](const Triangle& t) { 
            return 0.5 * t.getBase() * t.getHeight(); 
        }
    };

    // Print description of each shape
    auto printDescription = overloaded {
        [](const Circle& c) {
            std::cout << "Circle with radius " << c.getRadius();
        },
        [](const Rectangle& r) {
            std::cout << "Rectangle " << r.getWidth() << "x" << r.getHeight();
        },
        [](const Triangle& t) {
            std::cout << "Triangle with base " << t.getBase() 
                      << " and height " << t.getHeight();
        }
    };

    // Process all shapes
    for (const auto& shape : shapes) {
        std::cout << "Shape: ";
        std::visit(printDescription, shape);
        double area = std::visit(calculateArea, shape);
        std::cout << " has area: " << area << std::endl;
    }

    // Demonstrate counting shapes using a visitor
    size_t circleCount = 0;
    auto countCircles = overloaded {
        [&circleCount](const Circle&) { ++circleCount; },
        [](const auto&) { /* ignore other shapes */ }
    };

    for (const auto& shape : shapes) {
        std::visit(countCircles, shape);
    }
    
    std::cout << "\nFound " << circleCount << " circles" << std::endl;

    return 0;
}