#include <iostream>
#include <utility>

template <class... Args>
void foo(Args&&...args) {
    (..., (std::cout << std::forward<Args>(args) << ' '));
}

int main() {
    foo("red", "green", "blue");
}


#include <string_view>
#include <iostream>

template<std::size_t N>
void foo(const std::string_view(&strings)[N])
{
    for (const auto string : strings)
    {
        std::cout << string << "\n";
    }
}

int main()
{
    foo({ "red","green","blue" });
    return 0;
}


#include <iostream>
#include <string>
#include <initializer_list>

void foo (std::initializer_list <std::string> il)
{
    for (const auto &s : il)
        std::cout << s << " ";
}

int main()
{
    foo ({"red", "green", "blue"});
}
