#define BOOST_TEST_MODULE TypeTests
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TypeSystemTests)

BOOST_AUTO_TEST_CASE(BasicTypeProperties) {
    BOOST_CHECK_EQUAL(voidType.kind(), VoidKind);
    BOOST_CHECK_EQUAL(voidType.size(), 0);
    
    BOOST_CHECK_EQUAL(floatType.kind(), FloatKind);
    BOOST_CHECK_EQUAL(floatType.size(), 4);
    
    BOOST_CHECK_EQUAL(doubleType.kind(), DoubleKind);
    BOOST_CHECK_EQUAL(doubleType.size(), 8);
    
    BOOST_CHECK_EQUAL(boolType.kind(), IntegerKind);
    BOOST_CHECK_EQUAL(boolType.size(), 1);
}

BOOST_AUTO_TEST_CASE(TypeEquality) {
    BOOST_CHECK(voidType == voidType);
    BOOST_CHECK(floatType == floatType);
    BOOST_CHECK(doubleType == doubleType);
    BOOST_CHECK(boolType == boolType);
    
    BOOST_CHECK(voidType != floatType);
    BOOST_CHECK(floatType != doubleType);
    BOOST_CHECK(doubleType != boolType);
}

BOOST_AUTO_TEST_CASE(ArrayTypes) {
    Type intArray = Type(ArrayKind, 10); // Assuming constructor exists
    BOOST_CHECK_EQUAL(intArray.kind(), ArrayKind);
    BOOST_CHECK_EQUAL(intArray.size(), 40); // Assuming 4-byte integers
    
    // Test array element access
    Type elementType = intArray[0];
    BOOST_CHECK_EQUAL(elementType.kind(), IntegerKind);
    BOOST_CHECK_EQUAL(elementType.size(), 4);
}

BOOST_AUTO_TEST_CASE(PointerTypes) {
    Type intPtr = Type(PtrKind, IntegerKind); // Assuming constructor exists
    BOOST_CHECK_EQUAL(intPtr.kind(), PtrKind);
    BOOST_CHECK_EQUAL(intPtr.size(), 8); // Assuming 64-bit system
    
    Type voidPtr = Type(PtrKind, VoidKind);
    BOOST_CHECK_EQUAL(voidPtr.kind(), PtrKind);
    BOOST_CHECK_EQUAL(voidPtr.size(), 8);
}

BOOST_AUTO_TEST_CASE(FunctionTypes) {
    // Test function type with void return and no parameters
    Type voidFunc = Type(FuncKind, voidType);
    BOOST_CHECK_EQUAL(voidFunc.kind(), FuncKind);
    BOOST_CHECK_EQUAL(voidFunc.size(), 0);
    
    // Test function type with parameters
    std::vector<Type> params = {intType, floatType};
    Type funcWithParams = Type(FuncKind, voidType, params);
    BOOST_CHECK_EQUAL(funcWithParams.kind(), FuncKind);
    BOOST_CHECK_EQUAL(funcWithParams[0], intType);
    BOOST_CHECK_EQUAL(funcWithParams[1], floatType);
}

BOOST_AUTO_TEST_SUITE_END()
