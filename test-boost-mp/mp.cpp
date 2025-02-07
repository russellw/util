#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>

int main() {
    using namespace boost::multiprecision;
    cpp_int big = 1;
    for(int i = 1; i <= 100; ++i) big *= i;
    std::cout << "100! = " << big << std::endl;
    return 0;
}
