#include <boost/container_hash/hash.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

// Example custom type
struct Point {
    int x, y;
    
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// Hash specialization for Point using Boost
namespace std {
    template<>
    struct hash<Point> {
        std::size_t operator()(const Point& p) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, p.x);
            boost::hash_combine(seed, p.y);
            return seed;
        }
    };
}

int main() {
    // Demonstrating Boost's hash_combine
    std::size_t seed = 0;
    boost::hash_combine(seed, 42);
    boost::hash_combine(seed, std::string("hello"));
    std::cout << "Combined hash: " << seed << std::endl;
    
    // Using with Point in unordered_map
    std::unordered_map<Point, std::string> point_map;
    point_map[{1, 2}] = "Point(1,2)";
    point_map[{3, 4}] = "Point(3,4)";
    
    for (const auto& [point, value] : point_map) {
        std::cout << "Point(" << point.x << "," << point.y << ") hash: " 
                  << std::hash<Point>{}(point) << std::endl;
    }
    
    return 0;
}