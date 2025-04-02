#include <cstddef>
#include <iostream>
#include <string_view>

using namespace std::literals::string_view_literals;
using std::cout;

int main() {
    for (const auto color: {"red"sv, "green"sv, "blue"sv}) {
        cout << color << '\n';
    }

    return EXIT_SUCCESS;
}
