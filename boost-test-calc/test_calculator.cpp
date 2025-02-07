#define BOOST_TEST_MODULE Calculator_Test
#include <boost/test/included/unit_test.hpp>
#include "calculator.hpp"

BOOST_AUTO_TEST_SUITE(CalculatorTests)

BOOST_AUTO_TEST_CASE(test_basic_operations) {
    Calculator calc;
    BOOST_CHECK_CLOSE(calc.add(2, 3), 5.0, 0.001);
    BOOST_CHECK_CLOSE(calc.subtract(5, 3), 2.0, 0.001);
    BOOST_CHECK_CLOSE(calc.multiply(4, 3), 12.0, 0.001);
    BOOST_CHECK_CLOSE(calc.divide(6, 2), 3.0, 0.001);
}

BOOST_AUTO_TEST_CASE(test_division_by_zero) {
    Calculator calc;
    BOOST_CHECK_THROW(calc.divide(1, 0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_expression_evaluation) {
    Calculator calc;
    BOOST_CHECK_CLOSE(calc.evaluate("2 + 3"), 5.0, 0.001);
    BOOST_CHECK_CLOSE(calc.evaluate("10 - 4"), 6.0, 0.001);
    BOOST_CHECK_CLOSE(calc.evaluate("3 * 4"), 12.0, 0.001);
    BOOST_CHECK_CLOSE(calc.evaluate("8 / 2"), 4.0, 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
