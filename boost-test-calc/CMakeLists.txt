cmake_minimum_required(VERSION 3.10)
project(Calculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER clang++)

# Set Boost path
set(BOOST_ROOT "\\boost")
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(calculator_test test_calculator.cpp calculator.cpp)
target_link_libraries(calculator_test ${Boost_LIBRARIES})
