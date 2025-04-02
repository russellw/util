#include "all.h"

#define BOOST_TEST_MODULE Unit_Test
#include <boost/test/included/unit_test.hpp>

Term arithmetic(Tag tag, Term a) {
	return Term(tag, a.ty(), a);
}

Term arithmetic(Tag tag, Term a, Term b) {
	return Term(tag, a.ty(), a, b);
}

Type funcTy(const vector<Type>& v) {
	ASSERT(v.size());
	return funcTy(v[0], tail(v));
}

BOOST_AUTO_TEST_CASE(BasicTypeProperties) {
	BOOST_CHECK_EQUAL(voidTy().kind(), VoidKind);
	BOOST_CHECK_EQUAL(voidTy().size(), 0);

	BOOST_CHECK_EQUAL(floatTy().kind(), FloatKind);
	BOOST_CHECK_EQUAL(floatTy().size(), 0);

	BOOST_CHECK_EQUAL(doubleTy().kind(), DoubleKind);
	BOOST_CHECK_EQUAL(doubleTy().size(), 0);

	BOOST_CHECK_EQUAL(boolTy().kind(), IntKind);
	BOOST_CHECK_EQUAL(boolTy().size(), 0);
}

BOOST_AUTO_TEST_CASE(TypeEquality) {
	BOOST_CHECK(voidTy() == voidTy());
	BOOST_CHECK(floatTy() == floatTy());
	BOOST_CHECK(doubleTy() == doubleTy());
	BOOST_CHECK(boolTy() == boolTy());

	BOOST_CHECK(voidTy() != floatTy());
	BOOST_CHECK(floatTy() != doubleTy());
	BOOST_CHECK(doubleTy() != boolTy());
}

BOOST_AUTO_TEST_CASE(IntegerTypeProperties) {
	// Test common integer widths
	Type int8 = intTy(8);
	Type int16 = intTy(16);
	Type int32 = intTy(32);
	Type int64 = intTy(64);

	// Check kind
	BOOST_CHECK_EQUAL(int8.kind(), IntKind);
	BOOST_CHECK_EQUAL(int16.kind(), IntKind);
	BOOST_CHECK_EQUAL(int32.kind(), IntKind);
	BOOST_CHECK_EQUAL(int64.kind(), IntKind);

	// Check bit widths
	BOOST_CHECK_EQUAL(int8.len(), 8);
	BOOST_CHECK_EQUAL(int16.len(), 16);
	BOOST_CHECK_EQUAL(int32.len(), 32);
	BOOST_CHECK_EQUAL(int64.len(), 64);

	// Check sizes (should be 0 for scalar types)
	BOOST_CHECK_EQUAL(int8.size(), 0);
	BOOST_CHECK_EQUAL(int16.size(), 0);
	BOOST_CHECK_EQUAL(int32.size(), 0);
	BOOST_CHECK_EQUAL(int64.size(), 0);

	// Check bool type bits
	BOOST_CHECK_EQUAL(boolTy().len(), 1);
}

BOOST_AUTO_TEST_CASE(IntegerTypeEquality) {
	// Test equality of same-width integers
	Type int32_1 = intTy(32);
	Type int32_2 = intTy(32);
	BOOST_CHECK(int32_1 == int32_2);

	// Test inequality of different-width integers
	Type int16 = intTy(16);
	Type int32 = intTy(32);
	Type int64 = intTy(64);
	BOOST_CHECK(int16 != int32);
	BOOST_CHECK(int32 != int64);
	BOOST_CHECK(int16 != int64);

	// Test inequality with other types
	BOOST_CHECK(int32 != floatTy());
	BOOST_CHECK(int64 != doubleTy());
	BOOST_CHECK(int16 != voidTy());
}

BOOST_AUTO_TEST_CASE(IntegerTypeEdgeCases) {
	// Test unusual bit widths
	Type int1 = intTy(1);
	Type int3 = intTy(3);
	Type int128 = intTy(128);

	// Check properties
	BOOST_CHECK_EQUAL(int1.kind(), IntKind);
	BOOST_CHECK_EQUAL(int3.kind(), IntKind);
	BOOST_CHECK_EQUAL(int128.kind(), IntKind);

	BOOST_CHECK_EQUAL(int1.len(), 1);
	BOOST_CHECK_EQUAL(int3.len(), 3);
	BOOST_CHECK_EQUAL(int128.len(), 128);

	// Verify 1-bit integer is equivalent to bool type
	BOOST_CHECK(int1 == boolTy());

	// Test equality with same unusual widths
	Type int3_2 = intTy(3);
	Type int128_2 = intTy(128);
	BOOST_CHECK(int3 == int3_2);
	BOOST_CHECK(int128 == int128_2);
}

BOOST_AUTO_TEST_CASE(ArrayTypeProperties) {
	// Test array types with different element types and lengths
	Type int32 = intTy(32);
	Type float_arr_10 = arrayTy(10, floatTy());
	Type int_arr_5 = arrayTy(5, int32);
	Type bool_arr_2 = arrayTy(2, boolTy());

	// Check kinds
	BOOST_CHECK_EQUAL(float_arr_10.kind(), ArrayKind);
	BOOST_CHECK_EQUAL(int_arr_5.kind(), ArrayKind);
	BOOST_CHECK_EQUAL(bool_arr_2.kind(), ArrayKind);

	// Check lengths (number of elements)
	BOOST_CHECK_EQUAL(float_arr_10.len(), 10);
	BOOST_CHECK_EQUAL(int_arr_5.len(), 5);
	BOOST_CHECK_EQUAL(bool_arr_2.len(), 2);

	// Check sizes (should be 1 for array types)
	BOOST_CHECK_EQUAL(float_arr_10.size(), 1);
	BOOST_CHECK_EQUAL(int_arr_5.size(), 1);
	BOOST_CHECK_EQUAL(bool_arr_2.size(), 1);

	// Check element types
	BOOST_CHECK(float_arr_10[0] == floatTy());
	BOOST_CHECK(int_arr_5[0] == int32);
	BOOST_CHECK(bool_arr_2[0] == boolTy());
}

BOOST_AUTO_TEST_CASE(ArrayTypeEquality) {
	Type int32 = intTy(32);

	// Test equality of arrays with same element type and length
	Type int_arr_5_1 = arrayTy(5, int32);
	Type int_arr_5_2 = arrayTy(5, int32);
	BOOST_CHECK(int_arr_5_1 == int_arr_5_2);

	// Test inequality with different lengths
	Type int_arr_10 = arrayTy(10, int32);
	BOOST_CHECK(int_arr_5_1 != int_arr_10);

	// Test inequality with different element types
	Type float_arr_5 = arrayTy(5, floatTy());
	BOOST_CHECK(int_arr_5_1 != float_arr_5);

	// Test nested arrays
	Type nested_arr = arrayTy(3, arrayTy(2, int32));
	Type nested_arr_2 = arrayTy(3, arrayTy(2, int32));
	BOOST_CHECK(nested_arr == nested_arr_2);
}

BOOST_AUTO_TEST_CASE(VectorTypeProperties) {
	Type int32 = intTy(32);
	Type float_vec_4 = vecTy(4, floatTy());
	Type int_vec_8 = vecTy(8, int32);

	// Check kinds
	BOOST_CHECK_EQUAL(float_vec_4.kind(), VecKind);
	BOOST_CHECK_EQUAL(int_vec_8.kind(), VecKind);

	// Check lengths
	BOOST_CHECK_EQUAL(float_vec_4.len(), 4);
	BOOST_CHECK_EQUAL(int_vec_8.len(), 8);

	// Check sizes (should be 1 for vector types)
	BOOST_CHECK_EQUAL(float_vec_4.size(), 1);
	BOOST_CHECK_EQUAL(int_vec_8.size(), 1);

	// Check element types
	BOOST_CHECK(float_vec_4[0] == floatTy());
	BOOST_CHECK(int_vec_8[0] == int32);
}

BOOST_AUTO_TEST_CASE(StructTypeProperties) {
	// Test structure with various field types
	vector<Type> fields = {intTy(32), floatTy(), boolTy(), ptrTy()};
	Type struct_type = structTy(fields);

	// Check kind
	BOOST_CHECK_EQUAL(struct_type.kind(), StructKind);

	// Check size (should be number of fields)
	BOOST_CHECK_EQUAL(struct_type.size(), 4);

	// Check field types
	BOOST_CHECK(struct_type[0] == intTy(32));
	BOOST_CHECK(struct_type[1] == floatTy());
	BOOST_CHECK(struct_type[2] == boolTy());
	BOOST_CHECK(struct_type[3] == ptrTy());

	// Test empty struct
	vector<Type> empty_fields;
	Type empty_struct = structTy(empty_fields);
	BOOST_CHECK_EQUAL(empty_struct.kind(), StructKind);
	BOOST_CHECK_EQUAL(empty_struct.size(), 0);
}

BOOST_AUTO_TEST_CASE(StructTypeEquality) {
	vector<Type> fields1 = {intTy(32), floatTy()};
	vector<Type> fields2 = {intTy(32), floatTy()};
	vector<Type> fields3 = {floatTy(), intTy(32)};

	Type struct1 = structTy(fields1);
	Type struct2 = structTy(fields2);
	Type struct3 = structTy(fields3);

	// Test equality of identical structs
	BOOST_CHECK(struct1 == struct2);

	// Test inequality of structs with same types in different order
	BOOST_CHECK(struct1 != struct3);

	// Test nested structs
	vector<Type> nested_fields = {struct1, floatTy()};
	Type nested_struct1 = structTy(nested_fields);
	Type nested_struct2 = structTy(nested_fields);
	BOOST_CHECK(nested_struct1 == nested_struct2);
}

BOOST_AUTO_TEST_CASE(FuncTypeProperties) {
	// Test function type with various parameter types
	vector<Type> params = {
		intTy(32), // return type
		floatTy(), // param 1
		boolTy(),  // param 2
		ptrTy()	   // param 3
	};
	Type func_type = funcTy(params);

	// Check kind
	BOOST_CHECK_EQUAL(func_type.kind(), FuncKind);

	// Check size (should be 1 + number of parameters)
	BOOST_CHECK_EQUAL(func_type.size(), 4);

	// Check return type (component 0)
	BOOST_CHECK(func_type[0] == intTy(32));

	// Check parameter types
	BOOST_CHECK(func_type[1] == floatTy());
	BOOST_CHECK(func_type[2] == boolTy());
	BOOST_CHECK(func_type[3] == ptrTy());

	// Test function with no parameters (just return type)
	vector<Type> void_return = {voidTy()};
	Type void_func = funcTy(void_return);
	BOOST_CHECK_EQUAL(void_func.kind(), FuncKind);
	BOOST_CHECK_EQUAL(void_func.size(), 1);
	BOOST_CHECK(void_func[0] == voidTy());
}

BOOST_AUTO_TEST_CASE(FuncTypeEquality) {
	vector<Type> params1 = {intTy(32), floatTy(), boolTy()};
	vector<Type> params2 = {intTy(32), floatTy(), boolTy()};
	vector<Type> params3 = {intTy(32), boolTy(), floatTy()};

	Type func1 = funcTy(params1);
	Type func2 = funcTy(params2);
	Type func3 = funcTy(params3);

	// Test equality of identical function types
	BOOST_CHECK(func1 == func2);

	// Test inequality of functions with same types in different order
	BOOST_CHECK(func1 != func3);

	// Test functions with different return types
	vector<Type> params4 = {floatTy(), floatTy(), boolTy()};
	Type func4 = funcTy(params4);
	BOOST_CHECK(func1 != func4);
}

BOOST_AUTO_TEST_CASE(ComplexTypeCompositions) {
	Type int32 = intTy(32);

	// Create a struct containing an array of vectors
	Type vec4_float = vecTy(4, floatTy());
	Type arr3_vec = arrayTy(3, vec4_float);
	vector<Type> struct_fields = {int32, arr3_vec};
	Type complex_struct = structTy(struct_fields);

	// Check the structure
	BOOST_CHECK_EQUAL(complex_struct.kind(), StructKind);
	BOOST_CHECK_EQUAL(complex_struct.size(), 2);
	BOOST_CHECK(complex_struct[0] == int32);
	BOOST_CHECK(complex_struct[1] == arr3_vec);

	// Create a function type that uses this struct
	vector<Type> func_params = {voidTy(), complex_struct, ptrTy()};
	Type complex_func = funcTy(func_params);

	BOOST_CHECK_EQUAL(complex_func.kind(), FuncKind);
	BOOST_CHECK_EQUAL(complex_func.size(), 3);
	BOOST_CHECK(complex_func[0] == voidTy());
	BOOST_CHECK(complex_func[1] == complex_struct);
	BOOST_CHECK(complex_func[2] == ptrTy());
}
--types.cpp
// Test construction and basic properties of constants
BOOST_AUTO_TEST_CASE(ConstantTerms) {
	// Test boolean constants
	BOOST_CHECK_EQUAL(trueConst.ty(), boolTy());
	BOOST_CHECK_EQUAL(trueConst.tag(), Int);
	BOOST_CHECK_EQUAL(trueConst.intVal(), 1);

	BOOST_CHECK_EQUAL(falseConst.ty(), boolTy());
	BOOST_CHECK_EQUAL(falseConst.tag(), Int);
	BOOST_CHECK_EQUAL(falseConst.intVal(), 0);

	// Test null constant
	BOOST_CHECK_EQUAL(nullConst.ty(), ptrTy());
	BOOST_CHECK_EQUAL(nullConst.tag(), Null);

	// Test integer constant creation
	Type int32Type = intTy(32);
	cpp_int val(42);
	Term intTerm = intConst(int32Type, val);
	BOOST_CHECK_EQUAL(intTerm.ty(), int32Type);
	BOOST_CHECK_EQUAL(intTerm.tag(), Int);
	BOOST_CHECK_EQUAL(intTerm.intVal(), val);

	// Test float constant creation
	Term floatTerm = floatConst(floatTy(), "3.14");
	BOOST_CHECK_EQUAL(floatTerm.ty(), floatTy());
	BOOST_CHECK_EQUAL(floatTerm.tag(), Float);
	BOOST_CHECK_EQUAL(floatTerm.str(), "3.14");
}

// Test variable creation and properties
BOOST_AUTO_TEST_CASE(Variables) {
	Type int64Type = intTy(64);
	Term var1 = var(int64Type, 1);
	Term var2 = var(int64Type, 2);

	BOOST_CHECK_EQUAL(var1.ty(), int64Type);
	BOOST_CHECK_EQUAL(var1.tag(), Var);
	BOOST_CHECK(var1 != var2);
}

Term compound(Tag tag, const vector<Term>& v) {
	ASSERT(v.size());
	auto ty = v[0].ty();
	return Term(tag, ty, v);
}

// Test arithmetic operations
BOOST_AUTO_TEST_CASE(ArithmeticOperations) {
	Type int32Type = intTy(32);
	Term a = var(int32Type, 1);
	Term b = var(int32Type, 2);

	// Test addition
	vector<Term> addOps = {a, b};
	Term add = compound(Add, addOps);
	BOOST_CHECK_EQUAL(add.ty(), int32Type);
	BOOST_CHECK_EQUAL(add.tag(), Add);
	BOOST_CHECK_EQUAL(add.size(), 2);
	BOOST_CHECK_EQUAL(add[0], a);
	BOOST_CHECK_EQUAL(add[1], b);

	// Test multiplication
	vector<Term> mulOps = {a, b};
	Term mul = compound(Mul, mulOps);
	BOOST_CHECK_EQUAL(mul.ty(), int32Type);
	BOOST_CHECK_EQUAL(mul.tag(), Mul);
	BOOST_CHECK_EQUAL(mul.size(), 2);

	// Test floating point operations
	Term f1 = var(floatTy(), 3);
	Term f2 = var(floatTy(), 4);
	vector<Term> faddOps = {f1, f2};
	Term fadd = compound(FAdd, faddOps);
	BOOST_CHECK_EQUAL(fadd.ty(), floatTy());
	BOOST_CHECK_EQUAL(fadd.tag(), FAdd);

	// Test unary operations
	vector<Term> fnegOps = {f1};
	Term fneg = compound(FNeg, fnegOps);
	BOOST_CHECK_EQUAL(fneg.ty(), floatTy());
	BOOST_CHECK_EQUAL(fneg.tag(), FNeg);
	BOOST_CHECK_EQUAL(fneg.size(), 1);
}

// Test equality comparison
BOOST_AUTO_TEST_CASE(TermEquality) {
	Type int32Type = intTy(32);
	cpp_int val(42);

	Term int1 = intConst(int32Type, val);
	Term int2 = intConst(int32Type, val);
	Term int3 = intConst(int32Type, val + 1);

	BOOST_CHECK(int1 == int2);
	BOOST_CHECK(int1 != int3);

	Term var1 = var(int32Type, 1);
	Term var2 = var(int32Type, 1);
	Term var3 = var(int32Type, 2);

	BOOST_CHECK(var1 == var2);
	BOOST_CHECK(var1 != var3);
}
--terms.cpp
// Helper function to convert Type to string
std::string typeToString(Type ty) {
	std::ostringstream oss;
	oss << ty;
	return oss.str();
}

BOOST_AUTO_TEST_CASE(BasicTypeOutput) {
	// Test void type
	BOOST_CHECK_EQUAL(typeToString(voidTy()), "void");

	// Test float and double
	BOOST_CHECK_EQUAL(typeToString(floatTy()), "float");
	BOOST_CHECK_EQUAL(typeToString(doubleTy()), "double");

	// Test bool (1-bit integer)
	BOOST_CHECK_EQUAL(typeToString(boolTy()), "i1");

	// Test pointer
	BOOST_CHECK_EQUAL(typeToString(ptrTy()), "ptr");
}

BOOST_AUTO_TEST_CASE(IntegerTypeOutput) {
	// Test common integer sizes
	BOOST_CHECK_EQUAL(typeToString(intTy(8)), "i8");
	BOOST_CHECK_EQUAL(typeToString(intTy(16)), "i16");
	BOOST_CHECK_EQUAL(typeToString(intTy(32)), "i32");
	BOOST_CHECK_EQUAL(typeToString(intTy(64)), "i64");

	// Test unusual sizes
	BOOST_CHECK_EQUAL(typeToString(intTy(7)), "i7");
	BOOST_CHECK_EQUAL(typeToString(intTy(128)), "i128");
}

BOOST_AUTO_TEST_CASE(ArrayTypeOutput) {
	// Test arrays of basic types
	BOOST_CHECK_EQUAL(typeToString(arrayTy(4, intTy(32))), "[4 x i32]");
	BOOST_CHECK_EQUAL(typeToString(arrayTy(2, floatTy())), "[2 x float]");

	// Test nested arrays
	Type nestedArray = arrayTy(3, arrayTy(2, intTy(8)));
	BOOST_CHECK_EQUAL(typeToString(nestedArray), "[3 x [2 x i8]]");
}

BOOST_AUTO_TEST_CASE(VectorTypeOutput) {
	// Test vectors of basic types
	BOOST_CHECK_EQUAL(typeToString(vecTy(4, intTy(32))), "<4 x i32>");
	BOOST_CHECK_EQUAL(typeToString(vecTy(2, floatTy())), "<2 x float>");

	// Test unusual vector sizes
	BOOST_CHECK_EQUAL(typeToString(vecTy(3, intTy(1))), "<3 x i1>");
}

BOOST_AUTO_TEST_CASE(StructTypeOutput) {
	std::vector<Type> fields;

	// Test empty struct
	BOOST_CHECK_EQUAL(typeToString(structTy(fields)), "{}");

	// Test simple struct
	fields.push_back(intTy(32));
	fields.push_back(floatTy());
	BOOST_CHECK_EQUAL(typeToString(structTy(fields)), "{i32, float}");

	// Test nested struct
	std::vector<Type> innerFields;
	innerFields.push_back(intTy(8));
	innerFields.push_back(doubleTy());
	fields.push_back(structTy(innerFields));
	BOOST_CHECK_EQUAL(typeToString(structTy(fields)), "{i32, float, {i8, double}}");
}

BOOST_AUTO_TEST_CASE(FuncTypeOutput) {
	std::vector<Type> params;

	// Test function with no parameters
	params.push_back(voidTy()); // return type
	BOOST_CHECK_EQUAL(typeToString(funcTy(params)), "void ()");

	// Test function with basic parameters
	params.push_back(intTy(32));
	params.push_back(floatTy());
	BOOST_CHECK_EQUAL(typeToString(funcTy(params)), "void (i32, float)");

	// Test function returning non-void
	params[0] = ptrTy();
	BOOST_CHECK_EQUAL(typeToString(funcTy(params)), "ptr (i32, float)");

	// Test function with complex parameter types
	params.push_back(arrayTy(4, intTy(8)));
	BOOST_CHECK_EQUAL(typeToString(funcTy(params)), "ptr (i32, float, [4 x i8])");
}

BOOST_AUTO_TEST_CASE(ComplexTypeOutput) {
	// Test combination of various type constructs
	std::vector<Type> fields;
	fields.push_back(arrayTy(2, vecTy(4, intTy(32))));
	fields.push_back(ptrTy());

	std::vector<Type> funcParams;
	funcParams.push_back(structTy(fields)); // return type
	funcParams.push_back(doubleTy());
	funcParams.push_back(arrayTy(3, floatTy()));

	Type complexType = funcTy(funcParams);

	BOOST_CHECK_EQUAL(typeToString(complexType), "{[2 x <4 x i32>], ptr} (double, [3 x float])");
}
--type-output.cpp
BOOST_AUTO_TEST_CASE(BasicTypeMapping) {
	std::unordered_map<Type, int> typeMap;

	// Test primitive types
	typeMap[voidTy()] = 1;
	typeMap[intTy(32)] = 2;
	typeMap[boolTy()] = 3;

	BOOST_CHECK_EQUAL(typeMap[voidTy()], 1);
	BOOST_CHECK_EQUAL(typeMap[intTy(32)], 2);
	BOOST_CHECK_EQUAL(typeMap[boolTy()], 3);
}

BOOST_AUTO_TEST_CASE(CompoundTypeMapping) {
	std::unordered_map<Type, std::string> typeMap;

	// Create some compound types
	Type arrayOfInt = arrayTy(10, intTy(32));
	Type vectorOfFloat = vecTy(4, floatTy());

	typeMap[arrayOfInt] = "array of int";
	typeMap[vectorOfFloat] = "vector of float";

	BOOST_CHECK_EQUAL(typeMap[arrayOfInt], "array of int");
	BOOST_CHECK_EQUAL(typeMap[vectorOfFloat], "vector of float");

	// Test that identical types map to the same value
	Type sameArrayType = arrayTy(10, intTy(32));
	BOOST_CHECK_EQUAL(typeMap[sameArrayType], "array of int");
}

BOOST_AUTO_TEST_CASE(StructureTypeMapping) {
	std::unordered_map<Type, int> typeMap;

	std::vector<Type> fields1 = {intTy(32), floatTy()};
	std::vector<Type> fields2 = {intTy(32), floatTy()}; // Same structure
	std::vector<Type> fields3 = {floatTy(), intTy(32)}; // Different order

	Type struct1 = structTy(fields1);
	Type struct2 = structTy(fields2);
	Type struct3 = structTy(fields3);

	typeMap[struct1] = 1;

	// Test structural equality
	BOOST_CHECK_EQUAL(typeMap[struct2], 1);

	// Different structure should get different slot
	typeMap[struct3] = 2;
	BOOST_CHECK_EQUAL(typeMap[struct3], 2);
}

BOOST_AUTO_TEST_CASE(FuncTypeMapping) {
	std::unordered_map<Type, std::string> typeMap;

	// Function type: int32 (float, bool)
	std::vector<Type> params1 = {intTy(32), floatTy(), boolTy()};
	Type func1 = funcTy(params1);

	// Same function type
	std::vector<Type> params2 = {intTy(32), floatTy(), boolTy()};
	Type func2 = funcTy(params2);

	typeMap[func1] = "int32 (float, bool)";

	// Test that identical function types map to the same value
	BOOST_CHECK_EQUAL(typeMap[func2], "int32 (float, bool)");
}

BOOST_AUTO_TEST_CASE(TypeMapOverwrite) {
	std::unordered_map<Type, int> typeMap;

	Type int32Type = intTy(32);
	typeMap[int32Type] = 1;
	BOOST_CHECK_EQUAL(typeMap[int32Type], 1);

	// Overwrite existing value
	typeMap[int32Type] = 2;
	BOOST_CHECK_EQUAL(typeMap[int32Type], 2);
}

BOOST_AUTO_TEST_CASE(TypeMapErase) {
	std::unordered_map<Type, int> typeMap;

	Type int32Type = intTy(32);
	typeMap[int32Type] = 1;

	// Test erase
	size_t eraseCount = typeMap.erase(int32Type);
	BOOST_CHECK_EQUAL(eraseCount, 1);
	BOOST_CHECK_EQUAL(typeMap.count(int32Type), 0);
}
--type-map.cpp
BOOST_AUTO_TEST_CASE(BasicTermMapping) {
	std::unordered_map<Term, int> termMap;

	// Test constant terms
	termMap[trueConst] = 1;
	termMap[falseConst] = 2;
	termMap[nullConst] = 3;

	BOOST_CHECK_EQUAL(termMap[trueConst], 1);
	BOOST_CHECK_EQUAL(termMap[falseConst], 2);
	BOOST_CHECK_EQUAL(termMap[nullConst], 3);
}

BOOST_AUTO_TEST_CASE(IntegerTermMapping) {
	std::unordered_map<Term, std::string> termMap;

	// Create some integer constants
	Term int32_5 = intConst(intTy(32), 5);
	Term int32_10 = intConst(intTy(32), 10);
	Term int64_5 = intConst(intTy(64), 5); // Same value, different type

	termMap[int32_5] = "32-bit 5";
	termMap[int32_10] = "32-bit 10";
	termMap[int64_5] = "64-bit 5";

	BOOST_CHECK_EQUAL(termMap[int32_5], "32-bit 5");
	BOOST_CHECK_EQUAL(termMap[int32_10], "32-bit 10");
	BOOST_CHECK_EQUAL(termMap[int64_5], "64-bit 5");
}

BOOST_AUTO_TEST_CASE(FloatTermMapping) {
	std::unordered_map<Term, int> termMap;

	Term float1 = floatConst(floatTy(), "1.0");
	Term float2 = floatConst(floatTy(), "2.0");
	Term double1 = floatConst(doubleTy(), "1.0"); // Same value, different type

	termMap[float1] = 1;
	termMap[float2] = 2;
	termMap[double1] = 3;

	BOOST_CHECK_EQUAL(termMap[float1], 1);
	BOOST_CHECK_EQUAL(termMap[float2], 2);
	BOOST_CHECK_EQUAL(termMap[double1], 3);
}

BOOST_AUTO_TEST_CASE(VariableTermMapping) {
	std::unordered_map<Term, std::string> termMap;

	// Create some variables
	Term var1 = var(intTy(32), 1);
	Term var2 = var(intTy(32), 2);
	Term var1_float = var(floatTy(), 1); // Same index, different type

	termMap[var1] = "int var 1";
	termMap[var2] = "int var 2";
	termMap[var1_float] = "float var 1";

	BOOST_CHECK_EQUAL(termMap[var1], "int var 1");
	BOOST_CHECK_EQUAL(termMap[var2], "int var 2");
	BOOST_CHECK_EQUAL(termMap[var1_float], "float var 1");
}

BOOST_AUTO_TEST_CASE(CompoundTermMapping) {
	std::unordered_map<Term, std::string> termMap;

	// Create some arithmetic terms
	Term a = var(intTy(32), 1);
	Term b = var(intTy(32), 2);

	Term add = arithmetic(Add, a, b);
	Term mul = arithmetic(Mul, a, b);
	Term add_same = arithmetic(Add, a, b); // Same as first add

	termMap[add] = "a + b";
	termMap[mul] = "a * b";

	BOOST_CHECK_EQUAL(termMap[add], "a + b");
	BOOST_CHECK_EQUAL(termMap[mul], "a * b");
	BOOST_CHECK_EQUAL(termMap[add_same], "a + b"); // Should map to same value
}

BOOST_AUTO_TEST_CASE(ComparisonTermMapping) {
	std::unordered_map<Term, std::string> termMap;

	Term a = var(intTy(32), 1);
	Term b = var(intTy(32), 2);

	Term eq = cmp(Eq, a, b);
	Term lt = cmp(SLt, a, b);
	Term eq_same = cmp(Eq, a, b); // Same as first eq

	termMap[eq] = "a == b";
	termMap[lt] = "a < b";

	BOOST_CHECK_EQUAL(termMap[eq], "a == b");
	BOOST_CHECK_EQUAL(termMap[lt], "a < b");
	BOOST_CHECK_EQUAL(termMap[eq_same], "a == b");
}

BOOST_AUTO_TEST_CASE(TermMapOperations) {
	std::unordered_map<Term, int> termMap;

	Term var1 = var(intTy(32), 1);

	// Test insert and lookup
	termMap[var1] = 1;
	BOOST_CHECK_EQUAL(termMap[var1], 1);

	// Test overwrite
	termMap[var1] = 2;
	BOOST_CHECK_EQUAL(termMap[var1], 2);

	// Test erase
	size_t eraseCount = termMap.erase(var1);
	BOOST_CHECK_EQUAL(eraseCount, 1);
	BOOST_CHECK_EQUAL(termMap.count(var1), 0);
}
--term-map.cpp
// Helper function to create test environment
unordered_map<Term, Term> makeEnv(const vector<pair<Term, Term>>& bindings) {
	unordered_map<Term, Term> env;
	for (const auto& p : bindings) {
		env[p.first] = p.second;
	}
	return env;
}

BOOST_AUTO_TEST_CASE(constants_and_variables) {
	// Constants should remain unchanged
	Term nullTerm = nullConst;
	Term intTerm = intConst(intTy(32), 42);
	Term floatTerm = floatConst(floatTy(), "3.14");

	unordered_map<Term, Term> emptyEnv;
	BOOST_CHECK_EQUAL(simplify(emptyEnv, nullTerm), nullTerm);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, intTerm), intTerm);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, floatTerm), floatTerm);

	// Variables should be looked up in environment
	Term var1 = var(intTy(32), 1);
	Term val1 = intConst(intTy(32), 123);
	auto env = makeEnv({{var1, val1}});

	BOOST_CHECK_EQUAL(simplify(env, var1), val1);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, var1), var1); // Not in environment
}

BOOST_AUTO_TEST_CASE(basic_arithmetic) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	// Addition
	Term a = intConst(i32, 5);
	Term b = intConst(i32, 3);
	Term sum = compound(Add, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, sum).intVal(), 8);

	// x + 0 = x
	Term zero = intConst(i32, 0);
	Term addZero = compound(Add, {a, zero});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, addZero), a);

	// Subtraction
	Term diff = compound(Sub, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, diff).intVal(), 2);

	// x - x = 0
	Term subSelf = compound(Sub, {a, a});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, subSelf).intVal(), 0);

	// Multiplication
	Term prod = compound(Mul, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, prod).intVal(), 15);

	// x * 1 = x
	Term one = intConst(i32, 1);
	Term mulOne = compound(Mul, {a, one});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, mulOne), a);

	// x * 0 = 0
	Term mulZero = compound(Mul, {a, zero});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, mulZero).intVal(), 0);
}

BOOST_AUTO_TEST_CASE(division_and_remainder) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	Term a = intConst(i32, 15);
	Term b = intConst(i32, 4);
	Term zero = intConst(i32, 0);

	// Unsigned division
	Term udiv = compound(UDiv, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, udiv).intVal(), 3);

	// Division by zero should not be simplified
	Term divZero = compound(UDiv, {a, zero});
	BOOST_CHECK(simplify(emptyEnv, divZero).tag() == UDiv);

	// Signed division
	Term sdiv = compound(SDiv, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, sdiv).intVal(), 3);

	// Remainder
	Term urem = compound(URem, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, urem).intVal(), 3);
}

BOOST_AUTO_TEST_CASE(bitwise_operations) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	Term a = intConst(i32, 0b1100);
	Term b = intConst(i32, 0b1010);
	Term zero = intConst(i32, 0);

	// AND
	Term andOp = compound(And, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, andOp).intVal(), 0b1000);

	// x & 0 = 0
	Term andZero = compound(And, {a, zero});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, andZero).intVal(), 0);

	// OR
	Term orOp = compound(Or, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, orOp).intVal(), 0b1110);

	// x | 0 = x
	Term orZero = compound(Or, {a, zero});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, orZero), a);

	// XOR
	Term xorOp = compound(Xor, {a, b});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, xorOp).intVal(), 0b0110);

	// x ^ x = 0
	Term xorSelf = compound(Xor, {a, a});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, xorSelf).intVal(), 0);
}

BOOST_AUTO_TEST_CASE(shift_operations) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	Term a = intConst(i32, 0b1100);
	Term shift2 = intConst(i32, 2);
	Term shiftTooLarge = intConst(i32, 33); // Larger than type size

	// Logical left shift
	Term shl = compound(Shl, {a, shift2});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, shl).intVal(), 0b110000);

	// Invalid shift amount should not be simplified
	Term shlInvalid = compound(Shl, {a, shiftTooLarge});
	BOOST_CHECK(simplify(emptyEnv, shlInvalid).tag() == Shl);

	// Logical right shift
	Term lshr = compound(LShr, {a, shift2});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, lshr).intVal(), 0b11);

	// Arithmetic right shift (with sign bit)
	Term negative = intConst(i32, -16); // 0xFFFFFFF0 in two's complement
	Term ashr = compound(AShr, {negative, shift2});
	Term result = simplify(emptyEnv, ashr);
	BOOST_CHECK(result.intVal() < 0); // Should preserve sign
	BOOST_CHECK_EQUAL(result.intVal(), -4);
}

BOOST_AUTO_TEST_CASE(comparison_operations) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	Term a = intConst(i32, 5);
	Term b = intConst(i32, 3);
	Term equalA = intConst(i32, 5);

	// Equal
	Term eq1 = cmp(Eq, a, b);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, eq1), falseConst);

	Term eq2 = cmp(Eq, a, equalA);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, eq2), trueConst);

	// Unsigned comparison
	Term ult = cmp(ULt, a, b);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, ult), falseConst);

	// Signed comparison
	Term slt = cmp(SLt, a, b);
	BOOST_CHECK_EQUAL(simplify(emptyEnv, slt), falseConst);
}

BOOST_AUTO_TEST_CASE(floating_point_operations) {
	unordered_map<Term, Term> emptyEnv;

	// Floating point operations should not be simplified
	Term a = floatConst(floatTy(), "3.14");
	Term b = floatConst(floatTy(), "2.0");

	Term fadd = compound(FAdd, {a, b});
	BOOST_CHECK(simplify(emptyEnv, fadd).tag() == FAdd);

	Term fmul = compound(FMul, {a, b});
	BOOST_CHECK(simplify(emptyEnv, fmul).tag() == FMul);

	Term fneg = compound(FNeg, {a});
	BOOST_CHECK(simplify(emptyEnv, fneg).tag() == FNeg);
}

BOOST_AUTO_TEST_CASE(complex_expressions) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	// Test nested expressions: (5 + 3) * (10 - 4)
	Term a = intConst(i32, 5);
	Term b = intConst(i32, 3);
	Term c = intConst(i32, 10);
	Term d = intConst(i32, 4);

	Term sum = compound(Add, {a, b});		   // 5 + 3
	Term diff = compound(Sub, {c, d});		   // 10 - 4
	Term product = compound(Mul, {sum, diff}); // (5 + 3) * (10 - 4)

	BOOST_CHECK_EQUAL(simplify(emptyEnv, product).intVal(), 48); // (8 * 6)
}

// New test case specifically for same-component simplifications
BOOST_AUTO_TEST_CASE(same_component_simplifications) {
	unordered_map<Term, Term> emptyEnv;
	Type i32 = intTy(32);

	// Create some variables
	Term x = var(i32, 1);
	Term y = var(i32, 2);

	// x - x = 0
	Term subSelf = compound(Sub, {x, x});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, subSelf).intVal(), 0);

	// y - y = 0 (using different variable)
	Term subSelfY = compound(Sub, {y, y});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, subSelfY).intVal(), 0);

	// x ^ x = 0
	Term xorSelf = compound(Xor, {x, x});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, xorSelf).intVal(), 0);

	// x & x = x
	Term andSelf = compound(And, {x, x});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, andSelf), x);

	// x | x = x
	Term orSelf = compound(Or, {x, x});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, orSelf), x);

	// Nested cases to ensure simplification happens even when subexpressions don't change
	// (x & y) - (x & y) = 0
	Term complex1 = compound(And, {x, y});
	Term subComplex = compound(Sub, {complex1, complex1});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, subComplex).intVal(), 0);

	// (x | y) ^ (x | y) = 0
	Term complex2 = compound(Or, {x, y});
	Term xorComplex = compound(Xor, {complex2, complex2});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, xorComplex).intVal(), 0);

	// Test with constants to ensure the same-component rules take precedence
	// even when components are constants
	Term c = intConst(i32, 42);
	Term subConst = compound(Sub, {c, c});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, subConst).intVal(), 0);

	// Multiple levels of nesting
	// ((x & y) | (x & y)) = (x & y)
	Term nested1 = compound(And, {x, y});
	Term orNested = compound(Or, {nested1, nested1});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, orNested), nested1);

	// More complex expressions that should still simplify
	// (x & x & x) = x
	Term multiAnd = compound(And, {compound(And, {x, x}), x});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, multiAnd), x);

	// (x | (x | x)) = x
	Term multiOr = compound(Or, {x, compound(Or, {x, x})});
	BOOST_CHECK_EQUAL(simplify(emptyEnv, multiOr), x);
}
--simplify.cpp
BOOST_AUTO_TEST_CASE(test_fneg_float_constant) {
	// Create a float constant to negate
	Term f = floatConst(floatTy(), "1.0");
	Term neg = arithmetic(FNeg, f);

	// Verify properties
	BOOST_CHECK_EQUAL(neg.tag(), FNeg);
	BOOST_CHECK_EQUAL(neg.ty(), floatTy());
	BOOST_CHECK_EQUAL(neg.size(), 1);
	BOOST_CHECK_EQUAL(neg[0], f);
}

BOOST_AUTO_TEST_CASE(test_fneg_double_constant) {
	// Create a double constant to negate
	Term d = floatConst(doubleTy(), "2.5");
	Term neg = arithmetic(FNeg, d);

	// Verify properties
	BOOST_CHECK_EQUAL(neg.tag(), FNeg);
	BOOST_CHECK_EQUAL(neg.ty(), doubleTy());
	BOOST_CHECK_EQUAL(neg.size(), 1);
	BOOST_CHECK_EQUAL(neg[0], d);
}

BOOST_AUTO_TEST_CASE(test_fneg_variable) {
	// Create a float variable to negate
	Term var1 = var(floatTy(), 1);
	Term neg = arithmetic(FNeg, var1);

	// Verify properties
	BOOST_CHECK_EQUAL(neg.tag(), FNeg);
	BOOST_CHECK_EQUAL(neg.ty(), floatTy());
	BOOST_CHECK_EQUAL(neg.size(), 1);
	BOOST_CHECK_EQUAL(neg[0], var1);
}

BOOST_AUTO_TEST_CASE(test_double_negation) {
	// Verify that negating twice preserves type and structure
	Term f = floatConst(floatTy(), "3.14");
	Term neg1 = arithmetic(FNeg, f);
	Term neg2 = arithmetic(FNeg, neg1);

	BOOST_CHECK_EQUAL(neg2.tag(), FNeg);
	BOOST_CHECK_EQUAL(neg2.ty(), floatTy());
	BOOST_CHECK_EQUAL(neg2.size(), 1);
	BOOST_CHECK_EQUAL(neg2[0], neg1);
}

BOOST_AUTO_TEST_CASE(test_fneg_equality) {
	// Create two identical FNeg expressions
	Term f = floatConst(floatTy(), "1.0");
	Term neg1 = arithmetic(FNeg, f);
	Term neg2 = arithmetic(FNeg, f);

	// They should be equal due to value semantics
	BOOST_CHECK_EQUAL(neg1, neg2);
}

BOOST_AUTO_TEST_CASE(test_fneg_hash_consistency) {
	// Create two identical FNeg expressions
	Term f = floatConst(floatTy(), "1.0");
	Term neg1 = arithmetic(FNeg, f);
	Term neg2 = arithmetic(FNeg, f);

	// Their hashes should be equal
	std::hash<Term> hasher;
	BOOST_CHECK_EQUAL(hasher(neg1), hasher(neg2));
}

BOOST_AUTO_TEST_CASE(test_fneg_type_preservation) {
	// Test with both float and double
	Term f = floatConst(floatTy(), "1.0");
	Term d = floatConst(doubleTy(), "1.0");

	Term negF = arithmetic(FNeg, f);
	Term negD = arithmetic(FNeg, d);

	BOOST_CHECK_EQUAL(negF.ty(), f.ty());
	BOOST_CHECK_EQUAL(negD.ty(), d.ty());
}
--fneg.cpp
BOOST_AUTO_TEST_CASE(basic_match) {
	BOOST_TEST(containsAt("Hello World", 6, "World") == true);
}

BOOST_AUTO_TEST_CASE(match_at_beginning) {
	BOOST_TEST(containsAt("Hello World", 0, "Hello") == true);
}

BOOST_AUTO_TEST_CASE(match_at_end) {
	BOOST_TEST(containsAt("Hello World", 10, "d") == true);
}

BOOST_AUTO_TEST_CASE(no_match_wrong_position) {
	BOOST_TEST(containsAt("Hello World", 1, "Hello") == false);
}

BOOST_AUTO_TEST_CASE(empty_needle) {
	BOOST_TEST(containsAt("Hello World", 5, "") == true);
	BOOST_TEST(containsAt("Hello", 5, "") == true);	 // Empty needle at end of string
	BOOST_TEST(containsAt("Hello", 6, "") == false); // Position beyond string length
}

BOOST_AUTO_TEST_CASE(empty_haystack) {
	BOOST_TEST(containsAt("", 0, "") == true);
	BOOST_TEST(containsAt("", 0, "x") == false);
	BOOST_TEST(containsAt("", 1, "") == false); // Position beyond empty string
}

BOOST_AUTO_TEST_CASE(position_out_of_bounds) {
	BOOST_TEST(containsAt("Hello", 6, "") == false);
	BOOST_TEST(containsAt("Hello", 6, "x") == false);
}

BOOST_AUTO_TEST_CASE(needle_too_long) {
	BOOST_TEST(containsAt("Hello", 3, "loWorld") == false);
}

BOOST_AUTO_TEST_CASE(case_sensitivity) {
	BOOST_TEST(containsAt("Hello World", 6, "world") == false);
}

BOOST_AUTO_TEST_CASE(special_characters) {
	BOOST_TEST(containsAt("Hello\n\tWorld", 5, "\n\t") == true);
}
--contains-at.cpp
BOOST_AUTO_TEST_CASE(test_basic_parsing) {
	std::string input = "1A2B3C";
	size_t pos = 0;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0x1A2B3C);
	BOOST_CHECK_EQUAL(pos, 6);
}

BOOST_AUTO_TEST_CASE(test_lowercase_hex) {
	std::string input = "deadbeef";
	size_t pos = 0;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0xdeadbeef);
	BOOST_CHECK_EQUAL(pos, 8);
}

BOOST_AUTO_TEST_CASE(test_mixed_case) {
	std::string input = "AbCdEf";
	size_t pos = 0;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0xABCDEF);
	BOOST_CHECK_EQUAL(pos, 6);
}

BOOST_AUTO_TEST_CASE(test_max_length_limit) {
	std::string input = "123456789";
	size_t pos = 0;
	unsigned result = parseHex(input, pos, 4);
	BOOST_CHECK_EQUAL(result, 0x1234);
	BOOST_CHECK_EQUAL(pos, 4);
}

BOOST_AUTO_TEST_CASE(test_partial_string) {
	std::string input = "12XY34";
	size_t pos = 0;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0x12);
	BOOST_CHECK_EQUAL(pos, 2);
}

BOOST_AUTO_TEST_CASE(test_starting_position) {
	std::string input = "XX12AB";
	size_t pos = 2;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0x12AB);
	BOOST_CHECK_EQUAL(pos, 6);
}

BOOST_AUTO_TEST_CASE(test_empty_string) {
	std::string input = "";
	size_t pos = 0;
	BOOST_CHECK_THROW(parseHex(input, pos), std::runtime_error);
	BOOST_CHECK_EQUAL(pos, 0);
}

BOOST_AUTO_TEST_CASE(test_no_hex_digits) {
	std::string input = "XYZ";
	size_t pos = 0;
	BOOST_CHECK_THROW(parseHex(input, pos), std::runtime_error);
	BOOST_CHECK_EQUAL(pos, 0);
}

BOOST_AUTO_TEST_CASE(test_position_beyond_string) {
	std::string input = "123";
	size_t pos = 5;
	BOOST_CHECK_THROW(parseHex(input, pos), std::runtime_error);
	BOOST_CHECK_EQUAL(pos, 5);
}

BOOST_AUTO_TEST_CASE(test_zero) {
	std::string input = "0";
	size_t pos = 0;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0x0);
	BOOST_CHECK_EQUAL(pos, 1);
}

BOOST_AUTO_TEST_CASE(test_max_value) {
	std::string input = "FFFFFFFF";
	size_t pos = 0;
	unsigned result = parseHex(input, pos);
	BOOST_CHECK_EQUAL(result, 0xFFFFFFFF);
	BOOST_CHECK_EQUAL(pos, 8);
}
--parse-hex.cpp
// Test basic identifier without any special characters
BOOST_AUTO_TEST_CASE(BasicIdentifier) {
	BOOST_CHECK_EQUAL(unwrap("identifier"), "identifier");
	BOOST_CHECK_EQUAL(unwrap("abc123"), "abc123");
	BOOST_CHECK_EQUAL(unwrap("_underscore"), "_underscore");
}

// Test leading sigil removal
BOOST_AUTO_TEST_CASE(LeadingSigil) {
	BOOST_CHECK_EQUAL(unwrap("@variable"), "variable");
	BOOST_CHECK_EQUAL(unwrap("%local"), "local");
}

// Test quoted strings without escape sequences
BOOST_AUTO_TEST_CASE(QuotedStrings) {
	BOOST_CHECK_EQUAL(unwrap("\"quoted\""), "quoted");
	BOOST_CHECK_EQUAL(unwrap("\"\""), ""); // Empty quoted string
}

// Test strings with escape sequences
BOOST_AUTO_TEST_CASE(EscapeSequences) {
	BOOST_CHECK_EQUAL(unwrap("\"\\\\\""), "\\"); // Escaped backslash
}

// Test combinations of features
BOOST_AUTO_TEST_CASE(CombinedFeatures) {
	BOOST_CHECK_EQUAL(unwrap("@\"quoted\""), "quoted"); // Sigil with quotes
}

// Test error cases
BOOST_AUTO_TEST_CASE(ErrorCases) {
	// Unmatched quotes
	BOOST_CHECK_THROW(unwrap("\"unmatched"), std::runtime_error);

	// Invalid escape sequences
	BOOST_CHECK_THROW(unwrap("\"\\"), std::runtime_error); // Trailing backslash

	// Invalid characters in identifiers
	BOOST_CHECK_THROW(unwrap("invalid!char"), std::runtime_error);
	BOOST_CHECK_THROW(unwrap("space invalid"), std::runtime_error);
}

// Test edge cases
BOOST_AUTO_TEST_CASE(EdgeCases) {
	BOOST_CHECK_THROW(unwrap(""), std::runtime_error);
	BOOST_CHECK_EQUAL(unwrap("a"), "a");				// Single character
	BOOST_CHECK_EQUAL(unwrap("_"), "_");				// Just underscore
	BOOST_CHECK_THROW(unwrap("@"), std::runtime_error); // Just sigil
}

// Test whitespace handling
BOOST_AUTO_TEST_CASE(WhitespaceHandling) {
	BOOST_CHECK_EQUAL(unwrap("\"  spaced  \""), "  spaced  "); // Preserve internal spaces
}
--unwrap.cpp
// Test fixture for Parser tests
class ParserFixture {
protected:
	void parseFiles(const std::string& content1, const std::string& content2 = "") {
		if (!content1.empty()) {
			Parser("test1.ll", content1);
		}
		if (!content2.empty()) {
			Parser("test2.ll", content2);
		}
	}

	void expectError(const std::string& content1, const std::string& content2, const std::string& expectedError) {
		try {
			Parser("test1.ll", content1);
			Parser("test2.ll", content2);
			BOOST_FAIL("Expected error was not thrown");
		} catch (const std::runtime_error& e) {
			BOOST_CHECK_EQUAL(std::string(e.what()), expectedError);
		}
	}

	void expectError(const std::string& content, const std::string& expectedError) {
		try {
			Parser("test.ll", content);
			BOOST_FAIL("Expected error was not thrown");
		} catch (const std::runtime_error&) {
		}
	}
};

BOOST_FIXTURE_TEST_SUITE(ParserTests, ParserFixture)

BOOST_AUTO_TEST_CASE(ParseTargetTriple) {
	const std::string input = "target triple = \"x86_64-pc-linux-gnu\"\n";
	this->parseFiles(input);
	BOOST_CHECK_EQUAL(context::triple, "x86_64-pc-linux-gnu");
}

BOOST_AUTO_TEST_CASE(ParseTargetDatalayout) {
	const std::string input = "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n";
	this->parseFiles(input);
	BOOST_CHECK_EQUAL(context::datalayout, "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
}

BOOST_AUTO_TEST_CASE(ParseBothTargets) {
	const std::string input = "target triple = \"x86_64-pc-linux-gnu\"\n"
							  "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n";
	this->parseFiles(input);
	BOOST_CHECK_EQUAL(context::triple, "x86_64-pc-linux-gnu");
	BOOST_CHECK_EQUAL(context::datalayout, "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
}

BOOST_AUTO_TEST_CASE(ConsistentTargetsAcrossFiles) {
	const std::string input1 = "target triple = \"x86_64-pc-linux-gnu\"\n";
	const std::string input2 = "target triple = \"x86_64-pc-linux-gnu\"\n";
	this->parseFiles(input1, input2);
	BOOST_CHECK_EQUAL(context::triple, "x86_64-pc-linux-gnu");
}

BOOST_AUTO_TEST_CASE(InconsistentTriple) {
	const std::string input1 = "target triple = \"x86_64-pc-linux-gnu\"\n";
	const std::string input2 = "target triple = \"aarch64-apple-darwin\"\n";
	this->expectError(input1, input2, "test2.ll:1: inconsistent triple");
}

BOOST_AUTO_TEST_CASE(InconsistentDatalayout) {
	const std::string input1 = "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n";
	const std::string input2 = "target datalayout = \"e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n";
	this->expectError(input1, input2, "test2.ll:1: inconsistent datalayout");
}

BOOST_AUTO_TEST_CASE(MissingQuotes) {
	const std::string input = "target triple = x86_64-pc-linux-gnu\n";
	this->expectError(input, "test.ll:1: expected string");
}

BOOST_AUTO_TEST_CASE(UnclosedQuote) {
	const std::string input = "target triple = \"x86_64-pc-linux-gnu\n";
	this->expectError(input, "test.ll:1: unclosed quote");
}

BOOST_AUTO_TEST_CASE(MissingEquals) {
	const std::string input = "target triple \"x86_64-pc-linux-gnu\"\n";
	this->expectError(input, "test.ll:1: expected '='");
}

BOOST_AUTO_TEST_CASE(IgnoreComments) {
	const std::string input = "; This is a comment\n"
							  "target triple = \"x86_64-pc-linux-gnu\" ; Another comment\n";
	this->parseFiles(input);
	BOOST_CHECK_EQUAL(context::triple, "x86_64-pc-linux-gnu");
}

BOOST_AUTO_TEST_CASE(IgnoreWhitespace) {
	const std::string input = "   \t  target    triple    =    \"x86_64-pc-linux-gnu\"   \n";
	this->parseFiles(input);
	BOOST_CHECK_EQUAL(context::triple, "x86_64-pc-linux-gnu");
}

BOOST_AUTO_TEST_SUITE_END()
--parser.cpp
BOOST_AUTO_TEST_SUITE(ParserTests)

// Test parsing target triple and datalayout
BOOST_AUTO_TEST_CASE(TargetInfo) {
	context::clear();
	Parser parser("test.ll", "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n"
							 "target triple = \"x86_64-unknown-linux-gnu\"\n");

	BOOST_CHECK_EQUAL(context::datalayout, "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
	BOOST_CHECK_EQUAL(context::triple, "x86_64-unknown-linux-gnu");
}

BOOST_AUTO_TEST_SUITE_END()
--target.cpp
BOOST_AUTO_TEST_SUITE(TypeIteratorTests)

// Test scalar types have empty iteration range
BOOST_AUTO_TEST_CASE(ScalarTypeIterators) {
	// Test void type
	{
		Type t = voidTy();
		BOOST_CHECK(t.begin() == t.end());
		BOOST_CHECK(t.cbegin() == t.cend());
		BOOST_CHECK_EQUAL(std::distance(t.begin(), t.end()), 0);
	}

	// Test integer type
	{
		Type t = intTy(32);
		BOOST_CHECK(t.begin() == t.end());
		BOOST_CHECK(t.cbegin() == t.cend());
		BOOST_CHECK_EQUAL(std::distance(t.begin(), t.end()), 0);
	}

	// Test float type
	{
		Type t = floatTy();
		BOOST_CHECK(t.begin() == t.end());
		BOOST_CHECK(t.cbegin() == t.cend());
		BOOST_CHECK_EQUAL(std::distance(t.begin(), t.end()), 0);
	}
}

// Test vector type iteration
BOOST_AUTO_TEST_CASE(VectorTypeIterators) {
	Type elementType = intTy(32);
	Type vecT = vecTy(4, elementType);

	BOOST_CHECK(vecT.begin() != vecT.end());
	BOOST_CHECK(vecT.cbegin() != vecT.cend());
	BOOST_CHECK_EQUAL(std::distance(vecT.begin(), vecT.end()), 1);

	// Check element type through iterator
	BOOST_CHECK(*vecT.begin() == elementType);
}

// Test array type iteration
BOOST_AUTO_TEST_CASE(ArrayTypeIterators) {
	Type elementType = doubleTy();
	Type arrT = arrayTy(10, elementType);

	BOOST_CHECK(arrT.begin() != arrT.end());
	BOOST_CHECK(arrT.cbegin() != arrT.cend());
	BOOST_CHECK_EQUAL(std::distance(arrT.begin(), arrT.end()), 1);

	// Check element type through iterator
	BOOST_CHECK(*arrT.begin() == elementType);
}

// Test struct type iteration
BOOST_AUTO_TEST_CASE(StructTypeIterators) {
	std::vector<Type> fields = {intTy(32), floatTy(), doubleTy()};
	Type structT = structTy(fields);

	BOOST_CHECK(structT.begin() != structT.end());
	BOOST_CHECK(structT.cbegin() != structT.cend());
	BOOST_CHECK_EQUAL(std::distance(structT.begin(), structT.end()), fields.size());

	// Check field types through iterators
	auto it = structT.begin();
	for (const auto& field : fields) {
		BOOST_CHECK(*it == field);
		++it;
	}
}

// Test function type iteration
BOOST_AUTO_TEST_CASE(FuncTypeIterators) {
	std::vector<Type> params = {intTy(32), floatTy(), ptrTy()};
	Type rty = voidTy();
	std::vector<Type> funcTys = params;
	funcTys.insert(funcTys.begin(), rty);
	Type funcT = funcTy(funcTys);

	BOOST_CHECK(funcT.begin() != funcT.end());
	BOOST_CHECK(funcT.cbegin() != funcT.cend());
	BOOST_CHECK_EQUAL(std::distance(funcT.begin(), funcT.end()), params.size() + 1);

	// Check return type
	BOOST_CHECK(*funcT.begin() == rty);

	// Check parameter types
	auto it = std::next(funcT.begin());
	for (const auto& param : params) {
		BOOST_CHECK(*it == param);
		++it;
	}
}

// Test iterator comparison and assignment
BOOST_AUTO_TEST_CASE(IteratorOperations) {
	Type structT = structTy({intTy(32), floatTy()});

	// Test iterator assignment and comparison
	auto it1 = structT.begin();
	auto it2 = it1;
	BOOST_CHECK(it1 == it2);

	// Test iterator increment
	++it2;
	BOOST_CHECK(it1 != it2);

	// Test const_iterator assignment and comparison
	auto cit1 = structT.cbegin();
	auto cit2 = cit1;
	BOOST_CHECK(cit1 == cit2);

	// Test const_iterator increment
	++cit2;
	BOOST_CHECK(cit1 != cit2);

	// Test iterator to const_iterator conversion
	Type::const_iterator cit3 = it1;
	BOOST_CHECK(cit3 == it1);
}

// Test iterator invalidation
BOOST_AUTO_TEST_CASE(IteratorInvalidation) {
	Type structT1 = structTy({intTy(32), floatTy()});
	auto it1 = structT1.begin();

	// Create a new struct type
	Type structT2 = structTy({doubleTy(), ptrTy()});

	// Original iterator should still be valid and point to the original type
	BOOST_CHECK(*it1 == intTy(32));
}

BOOST_AUTO_TEST_SUITE_END()
--type-iterator.cpp
BOOST_AUTO_TEST_SUITE(TermIteratorTests)

// Test empty term iteration
BOOST_AUTO_TEST_CASE(EmptyTermTest) {
	Term emptyTerm;
	BOOST_CHECK(emptyTerm.begin() == emptyTerm.end());
	BOOST_CHECK(emptyTerm.cbegin() == emptyTerm.cend());

	// Verify iterator equality
	BOOST_CHECK(emptyTerm.begin() == emptyTerm.cbegin());
	BOOST_CHECK(emptyTerm.end() == emptyTerm.cend());
}

// Test iteration over function parameters
BOOST_AUTO_TEST_CASE(ParametersIterationTest) {
	// Create parameters with different types
	std::vector<Term> params = {var(intTy(32), "a"), var(doubleTy(), "b"), var(ptrTy(), "c")};

	Term paramTerm = tuple(params);

	// Check size matches number of parameters
	BOOST_CHECK_EQUAL(paramTerm.size(), 3);

	// Test forward iteration
	auto it = paramTerm.begin();
	BOOST_CHECK_EQUAL((*it).ty(), intTy(32));
	++it;
	BOOST_CHECK_EQUAL((*it).ty(), doubleTy());
	++it;
	BOOST_CHECK_EQUAL((*it).ty(), ptrTy());
	++it;
	BOOST_CHECK(it == paramTerm.end());
}

// Test iteration over arithmetic expressions
BOOST_AUTO_TEST_CASE(ArithmeticExpressionIterationTest) {
	Term a = var(intTy(32), "a");
	Term b = var(intTy(32), "b");
	Term addExpr = arithmetic(Add, a, b);

	// Check size
	BOOST_CHECK_EQUAL(addExpr.size(), 2);

	// Test const iteration
	auto cit = addExpr.cbegin();
	BOOST_CHECK_EQUAL((*cit).str(), "a");
	++cit;
	BOOST_CHECK_EQUAL((*cit).str(), "b");
	++cit;
	BOOST_CHECK(cit == addExpr.cend());
}

// Test comparison operations
BOOST_AUTO_TEST_CASE(IteratorComparisonTest) {
	Term a = var(intTy(32), "a");
	Term b = var(intTy(32), "b");
	Term expr = arithmetic(Add, a, b);

	auto it1 = expr.begin();
	auto it2 = expr.begin();
	auto end = expr.end();

	// Test equality
	BOOST_CHECK(it1 == it2);

	// Test inequality
	++it2;
	BOOST_CHECK(it1 != it2);

	// Test const iterator comparison
	auto cit1 = expr.cbegin();
	auto cit2 = expr.cbegin();
	BOOST_CHECK(cit1 == cit2);
}

// Test iterator invalidation
BOOST_AUTO_TEST_CASE(IteratorInvalidationTest) {
	std::vector<Term> params = {var(intTy(32), "a"), var(intTy(32), "b")};

	Term paramTerm = tuple(params);
	auto it = paramTerm.begin();
	auto end = paramTerm.end();

	// Store initial values
	std::vector<Term> initialValues;
	for (; it != end; ++it) {
		initialValues.push_back(*it);
	}

	// Create new term with same structure
	Term newParamTerm = tuple(params);

	// Verify iterators on original term still valid
	it = paramTerm.begin();
	for (const auto& initial : initialValues) {
		BOOST_CHECK_EQUAL(*it, initial);
		++it;
	}
}

BOOST_AUTO_TEST_SUITE_END()
--term-iterator.cpp

Term createMockVar(const string& name) {
	return var(floatTy(), name);
}

BOOST_AUTO_TEST_SUITE(ParserTestSuite)

BOOST_AUTO_TEST_CASE(ParseAddInst) {
	// Test error cases
	{
		// Mismatched types
		const string badInput = R"(
define i32 @test() {
    %1 = add i32 %2, i64 %3
    ret i32 %1
}
)";
		BOOST_CHECK_THROW(Parser("test.ll", badInput), runtime_error);

		// Missing operand
		const string missingOperand = R"(
define i32 @test() {
    %1 = add i32 %2
    ret i32 %1
}
)";
		BOOST_CHECK_THROW(Parser("test.ll", missingOperand), runtime_error);

		// Missing type
		const string missingType = R"(
define i32 @test() {
    %1 = add %2, %3
    ret i32 %1
}
)";
		BOOST_CHECK_THROW(Parser("test.ll", missingType), runtime_error);
	}
}

BOOST_AUTO_TEST_SUITE_END()
--parse-add.cpp
BOOST_AUTO_TEST_SUITE(ParseRefTests)

// Test basic numeric references
BOOST_AUTO_TEST_CASE(BasicNumericRefs) {
	// Test simple numeric references
	auto ref1 = parseRef("%0");
	BOOST_CHECK(std::holds_alternative<size_t>(ref1));
	BOOST_CHECK_EQUAL(std::get<size_t>(ref1), 0);

	auto ref2 = parseRef("%42");
	BOOST_CHECK(std::holds_alternative<size_t>(ref2));
	BOOST_CHECK_EQUAL(std::get<size_t>(ref2), 42);
}

// Test string references
BOOST_AUTO_TEST_CASE(StringRefs) {
	// Test basic string reference
	auto ref1 = parseRef("%foo");
	BOOST_CHECK(std::holds_alternative<std::string>(ref1));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref1), "foo");

	// Test string reference with underscore
	auto ref2 = parseRef("%my_var");
	BOOST_CHECK(std::holds_alternative<std::string>(ref2));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref2), "my_var");
}

// Test quoted references
BOOST_AUTO_TEST_CASE(QuotedRefs) {
	// Test quoted string that looks like a number
	auto ref1 = parseRef("%\"42\"");
	BOOST_CHECK(std::holds_alternative<std::string>(ref1));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref1), "42");

	// Test quoted string with spaces
	auto ref2 = parseRef("%\"hello world\"");
	BOOST_CHECK(std::holds_alternative<std::string>(ref2));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref2), "hello world");
}

// Test escaped sequences in quoted strings
BOOST_AUTO_TEST_CASE(EscapedRefs) {
	// Test string with hex escape
	auto ref2 = parseRef("%\"hello\\20world\"");
	BOOST_CHECK(std::holds_alternative<std::string>(ref2));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref2), "hello world");
}

// Test error cases
BOOST_AUTO_TEST_CASE(ErrorCases) {
	// Empty string
	BOOST_CHECK_THROW(parseRef(""), std::runtime_error);

	// Just sigil
	BOOST_CHECK_THROW(parseRef("%"), std::runtime_error);

	// Invalid number format
	BOOST_CHECK_THROW(parseRef("%42a"), std::runtime_error);

	// Unclosed quote
	BOOST_CHECK_THROW(parseRef("%\"unclosed"), std::runtime_error);
}

// Test edge cases
BOOST_AUTO_TEST_CASE(EdgeCases) {
	// Test maximum numeric value
	auto ref1 = parseRef("%18446744073709551615"); // max size_t
	BOOST_CHECK(std::holds_alternative<size_t>(ref1));
	BOOST_CHECK_EQUAL(std::get<size_t>(ref1), 18446744073709551615ULL);

	// Test empty quoted string
	auto ref2 = parseRef("%\"\"");
	BOOST_CHECK(std::holds_alternative<std::string>(ref2));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref2), "");

	// Test string with just spaces
	auto ref3 = parseRef("%\"   \"");
	BOOST_CHECK(std::holds_alternative<std::string>(ref3));
	BOOST_CHECK_EQUAL(std::get<std::string>(ref3), "   ");
}

BOOST_AUTO_TEST_SUITE_END()
--parse-ref.cpp
BOOST_AUTO_TEST_SUITE(HashVectorTests)

BOOST_AUTO_TEST_CASE(EmptyVectorTest) {
	vector<int> empty;
	BOOST_CHECK_EQUAL(hashVector(empty), 0);
}

BOOST_AUTO_TEST_CASE(SingleElementVectorTest) {
	vector<int> v{42};
	// Store the hash value calculated by hashVector
	size_t actual = hashVector(v);
	// Verify the hash is non-zero and consistent
	BOOST_CHECK_NE(actual, 0);
	BOOST_CHECK_EQUAL(hashVector(v), actual);
}

BOOST_AUTO_TEST_CASE(MultipleElementsVectorTest) {
	vector<int> v{1, 2, 3, 4, 5};
	size_t actual = hashVector(v);
	// Verify the hash is non-zero and consistent
	BOOST_CHECK_NE(actual, 0);
	BOOST_CHECK_EQUAL(hashVector(v), actual);
}

BOOST_AUTO_TEST_CASE(StringVectorTest) {
	vector<string> v{"hello", "world"};
	size_t actual = hashVector(v);
	// Verify the hash is non-zero and consistent
	BOOST_CHECK_NE(actual, 0);
	BOOST_CHECK_EQUAL(hashVector(v), actual);
}

BOOST_AUTO_TEST_CASE(OrderDependencyTest) {
	vector<int> v1{1, 2, 3};
	vector<int> v2{3, 2, 1};
	// Hashes should be different for different orders
	BOOST_CHECK_NE(hashVector(v1), hashVector(v2));
}

BOOST_AUTO_TEST_CASE(ConsistencyTest) {
	vector<int> v{1, 2, 3, 4, 5};
	size_t hash1 = hashVector(v);
	size_t hash2 = hashVector(v);
	// Same vector should produce same hash
	BOOST_CHECK_EQUAL(hash1, hash2);
}

BOOST_AUTO_TEST_CASE(DifferentTypesTest) {
	vector<int> v1{1, 2, 3};
	vector<double> v2{1.0, 2.0, 3.0};
	// Different types should produce different hashes even with "same" values
	BOOST_CHECK_NE(hashVector(v1), hashVector(v2));
}

BOOST_AUTO_TEST_CASE(LargeVectorTest) {
	vector<int> large(1000);
	for (int i = 0; i < 1000; ++i) {
		large[i] = i;
	}
	// Just verify it doesn't crash and returns non-zero
	BOOST_CHECK_NE(hashVector(large), 0);
}

BOOST_AUTO_TEST_SUITE_END()
--hash-vector.cpp
// Test suite for cons function
BOOST_AUTO_TEST_SUITE(ConsTests)

BOOST_AUTO_TEST_CASE(cons_empty_vector) {
	vector<int> empty;
	auto result = cons(1, empty);

	BOOST_CHECK_EQUAL(result.size(), 1);
	BOOST_CHECK_EQUAL(result[0], 1);
}

BOOST_AUTO_TEST_CASE(cons_nonempty_vector) {
	vector<int> v{2, 3, 4};
	auto result = cons(1, v);

	BOOST_CHECK_EQUAL(result.size(), 4);
	BOOST_CHECK_EQUAL(result[0], 1);
	BOOST_CHECK_EQUAL(result[1], 2);
	BOOST_CHECK_EQUAL(result[2], 3);
	BOOST_CHECK_EQUAL(result[3], 4);
}

BOOST_AUTO_TEST_CASE(cons_preserves_original) {
	vector<int> original{2, 3, 4};
	vector<int> originalCopy = original;
	auto result = cons(1, original);

	BOOST_CHECK_EQUAL_COLLECTIONS(original.begin(), original.end(), originalCopy.begin(), originalCopy.end());
}

BOOST_AUTO_TEST_CASE(cons_with_string) {
	vector<string> v{"world", "!"};
	auto result = cons(string("hello"), v);

	BOOST_CHECK_EQUAL(result.size(), 3);
	BOOST_CHECK_EQUAL(result[0], "hello");
	BOOST_CHECK_EQUAL(result[1], "world");
	BOOST_CHECK_EQUAL(result[2], "!");
}

BOOST_AUTO_TEST_SUITE_END()
--cons.cpp
// Test suite for call function
BOOST_AUTO_TEST_SUITE(CallTests)

BOOST_AUTO_TEST_CASE(call_no_args) {
	Type returnType = intTy(32);
	Term func = var(funcTy(returnType, {}), Ref("main"));
	vector<Term> emptyArgs;

	Term result = call(returnType, func, emptyArgs);

	BOOST_CHECK_EQUAL(result.tag(), Call);
	BOOST_CHECK_EQUAL(result.ty(), returnType);
	BOOST_CHECK_EQUAL(result.size(), 1);
	BOOST_CHECK_EQUAL(result[0], func);
}

BOOST_AUTO_TEST_CASE(call_with_args) {
	Type returnType = intTy(32);
	Type paramType = intTy(32);
	vector<Type> paramTypes{paramType, paramType};

	Term func = var(funcTy(returnType, paramTypes), Ref("add"));
	Term arg1 = intConst(paramType, cpp_int(1));
	Term arg2 = intConst(paramType, cpp_int(2));
	vector<Term> args{arg1, arg2};

	Term result = call(returnType, func, args);

	BOOST_CHECK_EQUAL(result.tag(), Call);
	BOOST_CHECK_EQUAL(result.ty(), returnType);
	BOOST_CHECK_EQUAL(result.size(), 3);
	BOOST_CHECK_EQUAL(result[0], func);
	BOOST_CHECK_EQUAL(result[1], arg1);
	BOOST_CHECK_EQUAL(result[2], arg2);
}

BOOST_AUTO_TEST_CASE(call_preserves_args) {
	Type returnType = intTy(32);
	Type paramType = intTy(32);
	Term func = var(funcTy(returnType, {paramType}), Ref("inc"));

	vector<Term> originalArgs{intConst(paramType, cpp_int(42))};
	vector<Term> argsCopy = originalArgs;

	Term result = call(returnType, func, originalArgs);

	BOOST_CHECK_EQUAL_COLLECTIONS(originalArgs.begin(), originalArgs.end(), argsCopy.begin(), argsCopy.end());
}

BOOST_AUTO_TEST_CASE(call_void_return) {
	Type vTy = voidTy();
	Term func = var(funcTy(vTy, {}), Ref("exit"));
	vector<Term> emptyArgs;

	Term result = call(vTy, func, emptyArgs);

	BOOST_CHECK_EQUAL(result.tag(), Call);
	BOOST_CHECK_EQUAL(result.ty(), vTy);
	BOOST_CHECK_EQUAL(result.size(), 1);
	BOOST_CHECK_EQUAL(result[0], func);
}

BOOST_AUTO_TEST_SUITE_END()
--call.cpp
BOOST_AUTO_TEST_SUITE(ZeroValTests)

// Test void type
BOOST_AUTO_TEST_CASE(TestVoidType) {
	Type ty = voidTy();
	Term result = zeroVal(ty);
	BOOST_CHECK_EQUAL(result.tag(), Tag::Null);
}

// Test integer types
BOOST_AUTO_TEST_CASE(TestIntegerTypes) {
	// Test bool (1-bit integer)
	{
		Type ty = boolTy();
		Term result = zeroVal(ty);
		BOOST_CHECK_EQUAL(result.tag(), Tag::Int);
		BOOST_CHECK_EQUAL(result.intVal(), 0);
	}

	// Test 32-bit integer
	{
		Type ty = intTy(32);
		Term result = zeroVal(ty);
		BOOST_CHECK_EQUAL(result.tag(), Tag::Int);
		BOOST_CHECK_EQUAL(result.intVal(), 0);
	}

	// Test 64-bit integer
	{
		Type ty = intTy(64);
		Term result = zeroVal(ty);
		BOOST_CHECK_EQUAL(result.tag(), Tag::Int);
		BOOST_CHECK_EQUAL(result.intVal(), 0);
	}
}

// Test floating point types
BOOST_AUTO_TEST_CASE(TestFloatTypes) {
	// Test float
	{
		Type ty = floatTy();
		Term result = zeroVal(ty);
		BOOST_CHECK_EQUAL(result.tag(), Tag::Float);
		BOOST_CHECK_EQUAL(result.str(), "0.0");
	}

	// Test double
	{
		Type ty = doubleTy();
		Term result = zeroVal(ty);
		BOOST_CHECK_EQUAL(result.tag(), Tag::Float);
		BOOST_CHECK_EQUAL(result.str(), "0.0");
	}
}

// Test pointer type
BOOST_AUTO_TEST_CASE(TestPointerType) {
	Type ty = ptrTy();
	Term result = zeroVal(ty);
	BOOST_CHECK_EQUAL(result, nullConst);
}

// Test array type
BOOST_AUTO_TEST_CASE(TestArrayType) {
	// Create an array of 3 integers
	Type elemTy = intTy(32);
	Type arrayTy3 = arrayTy(3, elemTy);
	Term result = zeroVal(arrayTy3);

	BOOST_CHECK_EQUAL(result.tag(), Tag::Array);
	BOOST_CHECK_EQUAL(result.size(), 3);

	// Check each element is zero
	for (size_t i = 0; i < 3; ++i) {
		BOOST_CHECK_EQUAL(result[i].tag(), Tag::Int);
		BOOST_CHECK_EQUAL(result[i].intVal(), 0);
	}
}

// Test vector type
BOOST_AUTO_TEST_CASE(TestVectorType) {
	// Create a vector of 4 floats
	Type elemTy = floatTy();
	Type vecTy4 = vecTy(4, elemTy);
	Term result = zeroVal(vecTy4);

	BOOST_CHECK_EQUAL(result.tag(), Tag::Vec);
	BOOST_CHECK_EQUAL(result.size(), 4);

	// Check each element is zero
	for (size_t i = 0; i < 4; ++i) {
		BOOST_CHECK_EQUAL(result[i].tag(), Tag::Float);
		BOOST_CHECK_EQUAL(result[i].str(), "0.0");
	}
}

// Test struct type
BOOST_AUTO_TEST_CASE(TestStructType) {
	// Create a struct with mixed types: {int32, float, ptr}
	vector<Type> fields = {intTy(32), floatTy(), ptrTy()};
	Type structTy1 = structTy(fields);
	Term result = zeroVal(structTy1);

	BOOST_CHECK_EQUAL(result.tag(), Tag::Tuple);
	BOOST_CHECK_EQUAL(result.size(), 3);

	// Check each field
	BOOST_CHECK_EQUAL(result[0].tag(), Tag::Int);
	BOOST_CHECK_EQUAL(result[0].intVal(), 0);

	BOOST_CHECK_EQUAL(result[1].tag(), Tag::Float);
	BOOST_CHECK_EQUAL(result[1].str(), "0.0");

	BOOST_CHECK_EQUAL(result[2], nullConst);
}

// Test nested types
BOOST_AUTO_TEST_CASE(TestNestedTypes) {
	// Create a struct containing an array of pointers
	Type ptrTy1 = ptrTy();
	Type arrayTy2 = arrayTy(2, ptrTy1);
	vector<Type> fields = {intTy(32), arrayTy2};
	Type structTy1 = structTy(fields);

	Term result = zeroVal(structTy1);

	BOOST_CHECK_EQUAL(result.tag(), Tag::Tuple);
	BOOST_CHECK_EQUAL(result.size(), 2);

	// Check integer field
	BOOST_CHECK_EQUAL(result[0].tag(), Tag::Int);
	BOOST_CHECK_EQUAL(result[0].intVal(), 0);

	// Check array field
	BOOST_CHECK_EQUAL(result[1].tag(), Tag::Array);
	BOOST_CHECK_EQUAL(result[1].size(), 2);
	BOOST_CHECK_EQUAL(result[1][0], nullConst);
	BOOST_CHECK_EQUAL(result[1][1], nullConst);
}

// Test function type (should throw)
BOOST_AUTO_TEST_CASE(TestFunctionType) {
	Type returnTy = voidTy();
	vector<Type> paramTys = {intTy(32)};
	Type funcTy1 = funcTy(returnTy, paramTys);

	BOOST_CHECK_THROW(zeroVal(funcTy1), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
--zero-val.cpp
BOOST_AUTO_TEST_CASE(test_parse_alloca_instruction) {
	// The test input: a function with a single alloca instruction.
	// Note: the alloca instruction has three operands:
	//  - The LHS variable (%1) which is a Var (with type ptrTy()),
	//  - A “zero value” for the allocated type (here, zero of i32),
	//  - And the number of elements (the constant 1, of type i32).
	std::string input = "define void @test() {\n"
						"%1 = alloca i32, i64 1\n"
						"}\n";

	// Construct the parser with the input
	Parser parser("test.ll", input);

	// Retrieve the parsed module
	Module* mod = parser.module;
	BOOST_REQUIRE_EQUAL(mod->defs.size(), 1);

	// Get the function @test and check its instruction count
	const Fn& func = mod->defs[0];
	BOOST_REQUIRE_EQUAL(func.size(), 1);

	// Get the parsed instruction, which should be the alloca instruction.
	const Inst& inst = func[0];
	BOOST_CHECK_EQUAL(inst.opcode(), Alloca);
	BOOST_REQUIRE_EQUAL(inst.size(), 3);

	// Check operand 0: the left-hand side variable.
	// It should be a Var term with type ptrTy().
	const Term& lval = inst[0];
	BOOST_CHECK_EQUAL(lval.tag(), Var);
	BOOST_CHECK(lval.ty() == ptrTy());

	// Check operand 1: the zero value for the allocated type.
	// It is constructed as zeroVal(i32) by the parser.
	const Term& zeroValTerm = inst[1];
	BOOST_CHECK(zeroValTerm.ty() == intTy(32));
	// Compare with an expected zero value for type i32.
	Term expectedZero = zeroVal(intTy(32));
	BOOST_CHECK_EQUAL(zeroValTerm, expectedZero);

	// Check operand 2: the number of elements (should be constant 1 of type i32).
	const Term& numElem = inst[2];
	BOOST_CHECK_EQUAL(numElem.intVal(), 1);
	BOOST_CHECK(numElem.ty() == intTy(64));
}
--parse-alloca.cpp
BOOST_AUTO_TEST_SUITE(ElementPtrTests)

// Test elementPtr with a basic integer array
BOOST_AUTO_TEST_CASE(BasicIntArrayTest) {
	// Create an array type of 10 32-bit integers
	Type elementType = intTy(32);
	Type arrayType = arrayTy(10, elementType);

	// Create a pointer variable to the array
	Term arrayPtr = var(ptrTy(), Ref("arr"));

	// Create an index
	Term index = intConst(intTy(64), 5);

	// Create the elementPtr term
	Term result = elementPtr(elementType, arrayPtr, index);

	// Verify the result
	BOOST_CHECK_EQUAL(result.tag(), ElementPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
	BOOST_CHECK_EQUAL(result.size(), 3);

	// Check that first operand is zero value of element type
	BOOST_CHECK_EQUAL(result[0], zeroVal(elementType));

	// Check pointer operand
	BOOST_CHECK_EQUAL(result[1], arrayPtr);

	// Check index operand
	BOOST_CHECK_EQUAL(result[2], index);
}

// Test elementPtr with a structure array
BOOST_AUTO_TEST_CASE(StructArrayTest) {
	// Create a struct type with two fields: int32 and float
	vector<Type> fields = {intTy(32), floatTy()};
	Type structType = structTy(fields);
	Type arrayType = arrayTy(5, structType);

	// Create a pointer variable to the array
	Term arrayPtr = var(ptrTy(), Ref("structArr"));

	// Create an index
	Term index = intConst(intTy(64), 2);

	// Create the elementPtr term
	Term result = elementPtr(structType, arrayPtr, index);

	// Verify the result
	BOOST_CHECK_EQUAL(result.tag(), ElementPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
	BOOST_CHECK_EQUAL(result.size(), 3);

	// Check that first operand is zero value of struct type
	BOOST_CHECK_EQUAL(result[0], zeroVal(structType));

	// Check pointer operand
	BOOST_CHECK_EQUAL(result[1], arrayPtr);

	// Check index operand
	BOOST_CHECK_EQUAL(result[2], index);
}

// Test elementPtr with variable index
BOOST_AUTO_TEST_CASE(VariableIndexTest) {
	Type elementType = doubleTy();
	Type arrayType = arrayTy(100, elementType);

	// Create a pointer variable to the array
	Term arrayPtr = var(ptrTy(), Ref("doubleArr"));

	// Create a variable index
	Term index = var(intTy(64), Ref("i"));

	// Create the elementPtr term
	Term result = elementPtr(elementType, arrayPtr, index);

	// Verify the result
	BOOST_CHECK_EQUAL(result.tag(), ElementPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());

	// Check that variable index is properly used
	BOOST_CHECK_EQUAL(result[2], index);
	BOOST_CHECK_EQUAL(result[2].tag(), Var);
}

// Test elementPtr with vectors
BOOST_AUTO_TEST_CASE(VectorTest) {
	// Create a vector type of 4 floats
	Type elementType = floatTy();
	Type vecType = vecTy(4, elementType);

	// Create a pointer variable to the vector
	Term vecPtr = var(ptrTy(), Ref("vec"));

	// Create an index
	Term index = intConst(intTy(64), 1);

	// Create the elementPtr term
	Term result = elementPtr(elementType, vecPtr, index);

	// Verify the result
	BOOST_CHECK_EQUAL(result.tag(), ElementPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());

	// Check that first operand is zero value of element type
	BOOST_CHECK_EQUAL(result[0], zeroVal(elementType));
}

// Test error cases (these might need to be handled at a higher level)
BOOST_AUTO_TEST_CASE(EdgeCases) {
	Type elementType = intTy(32);

	// Test with null pointer
	Term nullPtr = nullConst;
	Term index = intConst(intTy(64), 0);

	Term result = elementPtr(elementType, nullPtr, index);

	// Even with null pointer, the term should be well-formed
	BOOST_CHECK_EQUAL(result.tag(), ElementPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
	BOOST_CHECK_EQUAL(result[1], nullPtr);
}

BOOST_AUTO_TEST_SUITE_END()
--element-ptr.cpp
BOOST_AUTO_TEST_SUITE(FieldPtrTests)

// Test accessing first field of a simple struct
BOOST_AUTO_TEST_CASE(SimpleStructFirstField) {
	// Create a struct type with two fields: int32 and double
	vector<Type> fields = {intTy(32), doubleTy()};
	Type structType = structTy(fields);

	// Create a pointer to the struct
	Term ptr = var(ptrTy(), Ref("myStruct"));

	// Get pointer to first field
	Term result = fieldPtr(structType, ptr, 0);

	// Verify the result
	BOOST_CHECK_EQUAL(result.tag(), FieldPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
	BOOST_CHECK_EQUAL(result.size(), 3);
	BOOST_CHECK_EQUAL(result[0].ty(), structType);
	BOOST_CHECK_EQUAL(result[1], ptr);
	BOOST_CHECK_EQUAL(result[2].tag(), Int);
	BOOST_CHECK_EQUAL(result[2].intVal(), 0);
}

// Test accessing last field of a struct
BOOST_AUTO_TEST_CASE(SimpleStructLastField) {
	// Create a struct type with three fields
	vector<Type> fields = {intTy(32), floatTy(), doubleTy()};
	Type structType = structTy(fields);

	// Create a pointer to the struct
	Term ptr = var(ptrTy(), Ref("myStruct"));

	// Get pointer to last field
	Term result = fieldPtr(structType, ptr, 2);

	// Verify the result
	BOOST_CHECK_EQUAL(result.tag(), FieldPtr);
	BOOST_CHECK_EQUAL(result[2].intVal(), 2);
}

// Test with nested struct
BOOST_AUTO_TEST_CASE(NestedStruct) {
	// Create inner struct type
	vector<Type> innerFields = {intTy(32), floatTy()};
	Type innerStruct = structTy(innerFields);

	// Create outer struct type containing the inner struct
	vector<Type> outerFields = {doubleTy(), innerStruct};
	Type outerStruct = structTy(outerFields);

	// Create a pointer to the outer struct
	Term ptr = var(ptrTy(), Ref("outerStruct"));

	// Get pointer to the inner struct field
	Term result = fieldPtr(outerStruct, ptr, 1);

	BOOST_CHECK_EQUAL(result.tag(), FieldPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
	BOOST_CHECK_EQUAL(result[0].ty(), outerStruct);
	BOOST_CHECK_EQUAL(result[2].intVal(), 1);
}

// Test with null pointer
BOOST_AUTO_TEST_CASE(NullPointer) {
	vector<Type> fields = {intTy(32)};
	Type structType = structTy(fields);

	Term result = fieldPtr(structType, nullConst, 0);

	BOOST_CHECK_EQUAL(result.tag(), FieldPtr);
	BOOST_CHECK_EQUAL(result[1], nullConst);
}

// Test with empty struct
BOOST_AUTO_TEST_CASE(EmptyStruct) {
	vector<Type> fields;
	Type structType = structTy(fields);
	Term ptr = var(ptrTy(), Ref("emptyStruct"));

	// Since we're dealing with an empty struct, any index would be invalid
	// but the function itself should still construct a valid FieldPtr term
	Term result = fieldPtr(structType, ptr, 0);

	BOOST_CHECK_EQUAL(result.tag(), FieldPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
}

// Test accessing field in a struct containing array
BOOST_AUTO_TEST_CASE(StructWithArray) {
	// Create a struct with an array field
	Type arrayType = arrayTy(10, intTy(32));
	vector<Type> fields = {floatTy(), arrayType};
	Type structType = structTy(fields);

	Term ptr = var(ptrTy(), Ref("structWithArray"));

	// Get pointer to the array field
	Term result = fieldPtr(structType, ptr, 1);

	BOOST_CHECK_EQUAL(result.tag(), FieldPtr);
	BOOST_CHECK_EQUAL(result.ty(), ptrTy());
	BOOST_CHECK_EQUAL(result[0].ty(), structType);
	BOOST_CHECK_EQUAL(result[2].intVal(), 1);
}

BOOST_AUTO_TEST_SUITE_END()
--field-ptr.cpp
BOOST_AUTO_TEST_SUITE(GetElementPtrTests)

// Test GEP on a simple array type
BOOST_AUTO_TEST_CASE(SimpleArrayAccess) {
	// Create an array type of 4 integers
	auto elementType = intTy(32);
	auto arrayType = arrayTy(4, elementType);

	// Create a pointer and index
	auto ptr = var(ptrTy(), Ref("ptr"));
	auto idx = intConst(intTy(64), 2); // Access index 2

	// Get the element pointer
	auto result = getElementPtr(arrayType, ptr, {idx});

	// The result should be equivalent to elementPtr(elementType, ptr, idx)
	auto expected = elementPtr(elementType, ptr, idx);
	BOOST_CHECK_EQUAL(result, expected);
}

// Test GEP on nested array type
BOOST_AUTO_TEST_CASE(NestedArrayAccess) {
	// Create a 3x4 array of integers
	auto elementType = intTy(32);
	auto innerArrayType = arrayTy(4, elementType);
	auto outerArrayType = arrayTy(3, innerArrayType);

	auto ptr = var(ptrTy(), Ref("ptr"));
	auto idx1 = intConst(intTy(64), 1); // First dimension
	auto idx2 = intConst(intTy(64), 2); // Second dimension

	auto result = getElementPtr(outerArrayType, ptr, {idx1, idx2});

	// The result should be equivalent to nested elementPtr calls
	auto intermediate = elementPtr(innerArrayType, ptr, idx1);
	auto expected = elementPtr(elementType, intermediate, idx2);
	BOOST_CHECK_EQUAL(result, expected);
}

// Test GEP on struct type
BOOST_AUTO_TEST_CASE(StructAccess) {
	// Create a struct type with multiple fields
	vector<Type> fields = {
		intTy(32),	// field 0: int32
		doubleTy(), // field 1: double
		ptrTy()		// field 2: pointer
	};
	auto structType = structTy(fields);

	auto ptr = var(ptrTy(), Ref("ptr"));
	auto idx = intConst(intTy(64), 1); // Access field 1 (double)

	auto result = getElementPtr(structType, ptr, {idx});

	// The result should be equivalent to fieldPtr
	auto expected = fieldPtr(structType, ptr, 1);
	BOOST_CHECK_EQUAL(result, expected);
}

// Test GEP on struct containing array
BOOST_AUTO_TEST_CASE(StructWithArrayAccess) {
	// Create a struct containing an array
	auto arrayType = arrayTy(4, intTy(32));
	vector<Type> fields = {
		intTy(64), // field 0: int64
		arrayType  // field 1: array of 4 int32
	};
	auto structType = structTy(fields);

	auto ptr = var(ptrTy(), Ref("ptr"));
	auto fieldIdx = intConst(intTy(64), 1); // Access field 1 (array)
	auto arrayIdx = intConst(intTy(64), 2); // Access array index 2

	auto result = getElementPtr(structType, ptr, {fieldIdx, arrayIdx});

	// The result should be equivalent to fieldPtr followed by elementPtr
	auto intermediate = fieldPtr(structType, ptr, 1);
	auto expected = elementPtr(intTy(32), intermediate, arrayIdx);
	BOOST_CHECK_EQUAL(result, expected);
}

// Test GEP with empty index list
BOOST_AUTO_TEST_CASE(EmptyIndexList) {
	auto arrayType = arrayTy(4, intTy(32));
	auto ptr = var(ptrTy(), Ref("ptr"));

	auto result = getElementPtr(arrayType, ptr, {});

	// With no indices, should return the original pointer
	BOOST_CHECK_EQUAL(result, ptr);
}

// Test GEP with variable indices
BOOST_AUTO_TEST_CASE(VariableIndices) {
	auto arrayType = arrayTy(4, intTy(32));
	auto ptr = var(ptrTy(), Ref("ptr"));
	auto idx = var(intTy(64), Ref("i")); // Variable index

	auto result = getElementPtr(arrayType, ptr, {idx});

	// Should work the same as with constant indices
	auto expected = elementPtr(intTy(32), ptr, idx);
	BOOST_CHECK_EQUAL(result, expected);
}

// Test GEP with null pointer
BOOST_AUTO_TEST_CASE(NullPointerBase) {
	auto arrayType = arrayTy(4, intTy(32));
	auto idx = intConst(intTy(64), 2);

	auto result = getElementPtr(arrayType, nullConst, {idx});

	// Should work with null pointer base
	auto expected = elementPtr(intTy(32), nullConst, idx);
	BOOST_CHECK_EQUAL(result, expected);
}

BOOST_AUTO_TEST_SUITE_END()
--gep.cpp
BOOST_AUTO_TEST_SUITE(TermChecker)

// Helper functions to create common test terms
Term makeIntTerm(int bits, cpp_int value) {
	return intConst(intTy(bits), value);
}

Term makeFloatTerm(const string& val) {
	return floatConst(floatTy(), val);
}

Term makeDoubleTerm(const string& val) {
	return floatConst(doubleTy(), val);
}

// Null tests
BOOST_AUTO_TEST_CASE(NullTermValid) {
	BOOST_CHECK_NO_THROW(check(nullConst));
}

BOOST_AUTO_TEST_CASE(NullTermInvalidType) {
	Term invalidNull(Null, intTy(32), Ref());
	BOOST_CHECK_THROW(check(invalidNull), runtime_error);
}

// Integer constant tests
BOOST_AUTO_TEST_CASE(IntConstValid) {
	BOOST_CHECK_NO_THROW(check(makeIntTerm(32, 42)));
}

BOOST_AUTO_TEST_CASE(IntConstInvalidType) {
	Term invalidInt(Int, floatTy(), Ref());
	BOOST_CHECK_THROW(check(invalidInt), runtime_error);
}

// Floating point constant tests
BOOST_AUTO_TEST_CASE(FloatConstValid) {
	BOOST_CHECK_NO_THROW(check(makeFloatTerm("3.14")));
	BOOST_CHECK_NO_THROW(check(makeDoubleTerm("3.14")));
}

BOOST_AUTO_TEST_CASE(FloatConstInvalidType) {
	Term invalidFloat(Float, intTy(32), Ref("3.14"));
	BOOST_CHECK_THROW(check(invalidFloat), runtime_error);
}

// Arithmetic operation tests
BOOST_AUTO_TEST_CASE(IntegerArithmeticValid) {
	auto a = makeIntTerm(32, 1);
	auto b = makeIntTerm(32, 2);

	BOOST_CHECK_NO_THROW(check(Term(Add, intTy(32), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(Sub, intTy(32), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(Mul, intTy(32), a, b)));
}

BOOST_AUTO_TEST_CASE(IntegerArithmeticTypeMismatch) {
	auto a = makeIntTerm(32, 1);
	auto b = makeIntTerm(64, 2);

	BOOST_CHECK_THROW(check(Term(Add, intTy(32), a, b)), runtime_error);
}

BOOST_AUTO_TEST_CASE(FloatingArithmeticValid) {
	auto a = makeFloatTerm("1.0");
	auto b = makeFloatTerm("2.0");

	BOOST_CHECK_NO_THROW(check(Term(FAdd, floatTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(FSub, floatTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(FMul, floatTy(), a, b)));
}

BOOST_AUTO_TEST_CASE(FloatingArithmeticTypeMismatch) {
	auto a = makeFloatTerm("1.0");
	auto b = makeDoubleTerm("2.0");

	BOOST_CHECK_THROW(check(Term(FAdd, floatTy(), a, b)), runtime_error);
}

// Comparison tests
BOOST_AUTO_TEST_CASE(IntegerComparisonValid) {
	auto a = makeIntTerm(32, 1);
	auto b = makeIntTerm(32, 2);

	BOOST_CHECK_NO_THROW(check(Term(ULt, boolTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(ULe, boolTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(SLt, boolTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(SLe, boolTy(), a, b)));
}

BOOST_AUTO_TEST_CASE(FloatComparisonValid) {
	auto a = makeFloatTerm("1.0");
	auto b = makeFloatTerm("2.0");

	BOOST_CHECK_NO_THROW(check(Term(FLt, boolTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(FLe, boolTy(), a, b)));
}

// Logical operation tests
BOOST_AUTO_TEST_CASE(LogicalOperationsValid) {
	auto a = Term(Int, boolTy(), Ref("1"));
	auto b = Term(Int, boolTy(), Ref("0"));

	BOOST_CHECK_NO_THROW(check(Term(And, boolTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(Or, boolTy(), a, b)));
	BOOST_CHECK_NO_THROW(check(Term(Not, boolTy(), a)));
}

// Array tests
BOOST_AUTO_TEST_CASE(ArrayValid) {
	vector<Term> elements = {makeIntTerm(32, 1), makeIntTerm(32, 2), makeIntTerm(32, 3)};
	BOOST_CHECK_NO_THROW(check(array(intTy(32), elements)));
}

BOOST_AUTO_TEST_CASE(ArrayTypeMismatch) {
	vector<Term> elements = {makeIntTerm(32, 1), makeFloatTerm("2.0")};
	BOOST_CHECK_THROW(check(array(intTy(32), elements)), runtime_error);
}

// Tuple tests
BOOST_AUTO_TEST_CASE(TupleValid) {
	vector<Term> elements = {makeIntTerm(32, 1), makeFloatTerm("2.0"), makeDoubleTerm("3.0")};
	vector<Type> types = {intTy(32), floatTy(), doubleTy()};
	BOOST_CHECK_NO_THROW(check(Term(Tuple, structTy(types), elements)));
}

BOOST_AUTO_TEST_CASE(TupleTypeMismatch) {
	vector<Term> elements = {makeIntTerm(32, 1), makeFloatTerm("2.0")};
	vector<Type> types = {intTy(64), floatTy()};
	BOOST_CHECK_THROW(check(Term(Tuple, structTy(types), elements)), runtime_error);
}

// Function call tests
BOOST_AUTO_TEST_CASE(FunctionCallValid) {
	// Create a function type: int32(float, double)
	vector<Type> paramTypes = {floatTy(), doubleTy()};
	Type funcType = funcTy(intTy(32), paramTypes);

	// Create function reference and arguments
	Term func = Term(GlobalRef, funcType, Ref("test_func"));
	Term arg1 = makeFloatTerm("1.0");
	Term arg2 = makeDoubleTerm("2.0");

	vector<Term> args = {func, arg1, arg2};
	BOOST_CHECK_NO_THROW(check(Term(Call, intTy(32), args)));
}

BOOST_AUTO_TEST_CASE(FunctionCallWrongArgCount) {
	vector<Type> paramTypes = {floatTy(), doubleTy()};
	Type funcType = funcTy(intTy(32), paramTypes);

	Term func = Term(GlobalRef, funcType, Ref("test_func"));
	Term arg1 = makeFloatTerm("1.0");

	vector<Term> args = {func, arg1}; // Missing one argument
	BOOST_CHECK_THROW(check(Term(Call, intTy(32), args)), runtime_error);
}

// Pointer operation tests
BOOST_AUTO_TEST_CASE(LoadValid) {
	Term ptr(Var, ptrTy(), Ref("ptr"));
	BOOST_CHECK_NO_THROW(check(Term(Load, intTy(32), ptr)));
}

BOOST_AUTO_TEST_CASE(LoadInvalidPointer) {
	Term nonPtr(Var, intTy(32), Ref("not_a_ptr"));
	BOOST_CHECK_THROW(check(Term(Load, intTy(32), nonPtr)), runtime_error);
}

BOOST_AUTO_TEST_CASE(ElementPtrValid) {
	Term ptr(Var, ptrTy(), Ref("array_ptr"));
	Term idx = makeIntTerm(32, 0);
	Term zero = zeroVal(intTy(32));
	BOOST_CHECK_NO_THROW(check(Term(ElementPtr, ptrTy(), zero, ptr, idx)));
}

BOOST_AUTO_TEST_CASE(ElementPtrInvalidIndex) {
	Term ptr(Var, ptrTy(), Ref("array_ptr"));
	Term invalidIdx = makeFloatTerm("0.0");
	Term zero = zeroVal(intTy(32));
	BOOST_CHECK_THROW(check(Term(ElementPtr, ptrTy(), zero, ptr, invalidIdx)), runtime_error);
}

// Cast operation tests
BOOST_AUTO_TEST_CASE(CastValid) {
	auto intVal = makeIntTerm(32, 42);
	BOOST_CHECK_NO_THROW(check(Term(Cast, floatTy(), intVal)));
	BOOST_CHECK_NO_THROW(check(Term(SCast, doubleTy(), intVal)));
}

BOOST_AUTO_TEST_CASE(CastInvalidTypes) {
	Term ptr(Var, ptrTy(), Ref("ptr"));
	BOOST_CHECK_THROW(check(Term(Cast, intTy(32), ptr)), runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
--check-terms.cpp
BOOST_AUTO_TEST_SUITE(RecursiveTermChecker)

// Helper function to create test terms
Term makeIntTerm(int bits, cpp_int value) {
	return intConst(intTy(bits), value);
}

// Test deeply nested arithmetic expressions
BOOST_AUTO_TEST_CASE(NestedArithmeticValid) {
	// Build expression: ((1 + 2) * (3 + 4)) + 5
	auto one = makeIntTerm(32, 1);
	auto two = makeIntTerm(32, 2);
	auto three = makeIntTerm(32, 3);
	auto four = makeIntTerm(32, 4);
	auto five = makeIntTerm(32, 5);

	auto sum1 = Term(Add, intTy(32), one, two);
	auto sum2 = Term(Add, intTy(32), three, four);
	auto product = Term(Mul, intTy(32), sum1, sum2);
	auto final = Term(Add, intTy(32), product, five);

	BOOST_CHECK_NO_THROW(checkRecursive(final));
}

BOOST_AUTO_TEST_CASE(NestedArithmeticInvalid) {
	// Build expression with type mismatch: (32-bit + 64-bit)
	auto a = makeIntTerm(32, 1);
	auto b = makeIntTerm(64, 2);
	auto sum = Term(Add, intTy(32), a, b);

	BOOST_CHECK_THROW(checkRecursive(sum), runtime_error);
}

// Test nested arrays
BOOST_AUTO_TEST_CASE(NestedArrayValid) {
	// Create an array of arrays
	vector<Term> inner1 = {makeIntTerm(32, 1), makeIntTerm(32, 2)};
	vector<Term> inner2 = {makeIntTerm(32, 3), makeIntTerm(32, 4)};

	auto array1 = array(intTy(32), inner1);
	auto array2 = array(intTy(32), inner2);

	vector<Term> outer = {array1, array2};
	auto arrayOfArrays = Term(Array, arrayTy(2, arrayTy(2, intTy(32))), outer);

	BOOST_CHECK_NO_THROW(checkRecursive(arrayOfArrays));
}

BOOST_AUTO_TEST_CASE(NestedArrayInvalid) {
	// Create an array with inconsistent inner array types
	vector<Term> inner1 = {makeIntTerm(32, 1), makeIntTerm(32, 2)};
	vector<Term> inner2 = {makeIntTerm(64, 3), makeIntTerm(64, 4)}; // Different bit width

	auto array1 = array(intTy(32), inner1);
	auto array2 = array(intTy(64), inner2);

	vector<Term> outer = {array1, array2};
	auto arrayOfArrays = Term(Array, arrayTy(2, arrayTy(2, intTy(32))), outer);

	BOOST_CHECK_THROW(checkRecursive(arrayOfArrays), runtime_error);
}

// Test nested tuples
BOOST_AUTO_TEST_CASE(NestedTupleValid) {
	// Create a tuple containing another tuple
	vector<Type> innerTypes = {intTy(32), intTy(32)};
	vector<Term> innerElements = {makeIntTerm(32, 1), makeIntTerm(32, 2)};
	auto innerTuple = Term(Tuple, structTy(innerTypes), innerElements);

	vector<Type> outerTypes = {structTy(innerTypes), intTy(64)};
	vector<Term> outerElements = {innerTuple, makeIntTerm(64, 3)};
	auto outerTuple = Term(Tuple, structTy(outerTypes), outerElements);

	BOOST_CHECK_NO_THROW(checkRecursive(outerTuple));
}

BOOST_AUTO_TEST_CASE(NestedTupleInvalid) {
	// Create a tuple with type mismatch in inner tuple
	vector<Type> innerTypes = {intTy(32), intTy(32)};
	vector<Term> innerElements = {makeIntTerm(32, 1), makeIntTerm(64, 2)}; // Type mismatch
	auto innerTuple = Term(Tuple, structTy(innerTypes), innerElements);

	vector<Type> outerTypes = {structTy(innerTypes), intTy(64)};
	vector<Term> outerElements = {innerTuple, makeIntTerm(64, 3)};
	auto outerTuple = Term(Tuple, structTy(outerTypes), outerElements);

	BOOST_CHECK_THROW(checkRecursive(outerTuple), runtime_error);
}

// Test nested function calls
BOOST_AUTO_TEST_CASE(NestedFunctionCallValid) {
	// Create a function call where one argument is the result of another function call
	vector<Type> innerParamTypes = {intTy(32), intTy(32)};
	Type innerFuncType = funcTy(intTy(32), innerParamTypes);
	Term innerFunc = Term(GlobalRef, innerFuncType, Ref("inner_func"));

	vector<Term> innerArgs = {innerFunc, makeIntTerm(32, 1), makeIntTerm(32, 2)};
	auto innerCall = Term(Call, intTy(32), innerArgs);

	vector<Type> outerParamTypes = {intTy(32)};
	Type outerFuncType = funcTy(intTy(64), outerParamTypes);
	Term outerFunc = Term(GlobalRef, outerFuncType, Ref("outer_func"));

	vector<Term> outerArgs = {outerFunc, innerCall};
	auto outerCall = Term(Call, intTy(64), outerArgs);

	BOOST_CHECK_NO_THROW(checkRecursive(outerCall));
}

BOOST_AUTO_TEST_CASE(NestedFunctionCallInvalid) {
	// Create nested function calls with type mismatch
	vector<Type> innerParamTypes = {intTy(32), intTy(32)};
	Type innerFuncType = funcTy(intTy(64), innerParamTypes); // Returns int64
	Term innerFunc = Term(GlobalRef, innerFuncType, Ref("inner_func"));

	vector<Term> innerArgs = {innerFunc, makeIntTerm(32, 1), makeIntTerm(32, 2)};
	auto innerCall = Term(Call, intTy(64), innerArgs);

	vector<Type> outerParamTypes = {intTy(32)}; // Expects int32
	Type outerFuncType = funcTy(intTy(64), outerParamTypes);
	Term outerFunc = Term(GlobalRef, outerFuncType, Ref("outer_func"));

	vector<Term> outerArgs = {outerFunc, innerCall}; // Type mismatch
	auto outerCall = Term(Call, intTy(64), outerArgs);

	BOOST_CHECK_THROW(checkRecursive(outerCall), runtime_error);
}

// Test complex nested expressions
BOOST_AUTO_TEST_CASE(ComplexNestedExpressionValid) {
	// Create a complex expression with multiple levels of nesting
	// ((array[i + 1] * 2) + func(x, y)) where array is a pointer to int32

	// Create array access
	auto ptr = Term(Var, ptrTy(), Ref("array"));
	auto i = Term(Var, intTy(32), Ref("i"));
	auto one = makeIntTerm(32, 1);
	auto index = Term(Add, intTy(32), i, one);
	auto elemPtr = elementPtr(intTy(32), ptr, index);
	auto arrayElem = Term(Load, intTy(32), elemPtr);

	// Create multiplication
	auto two = makeIntTerm(32, 2);
	auto product = Term(Mul, intTy(32), arrayElem, two);

	// Create function call
	vector<Type> paramTypes = {intTy(32), intTy(32)};
	Type funcType = funcTy(intTy(32), paramTypes);
	Term func = Term(GlobalRef, funcType, Ref("func"));
	Term x = Term(Var, intTy(32), Ref("x"));
	Term y = Term(Var, intTy(32), Ref("y"));
	vector<Term> args = {func, x, y};
	auto funcCall = Term(Call, intTy(32), args);

	// Add everything together
	auto final = Term(Add, intTy(32), product, funcCall);

	BOOST_CHECK_NO_THROW(checkRecursive(final));
}

BOOST_AUTO_TEST_SUITE_END()
--check-terms-recur.cpp

// Helper functions to create test values
Term makeIntVar(const string& name, size_t bits = 32) {
	return var(intTy(bits), Ref(name));
}

Term makePtrVar(const string& name) {
	return var(ptrTy(), Ref(name));
}

Term makeLabel(const string& name) {
	return label(Ref(name));
}

BOOST_AUTO_TEST_SUITE(InstructionCheckerTests)

BOOST_AUTO_TEST_CASE(test_alloca) {
	// Valid alloca
	Term ptr = makePtrVar("ptr");
	Term type = zeroVal(intTy(32));
	Term size = intConst(intTy(32), 1);
	Inst valid = alloca(ptr, intTy(32), size);
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooFew(Alloca, {ptr, type});
	BOOST_CHECK_THROW(check(tooFew), runtime_error);

	// First operand not a variable
	Inst badFirst = alloca(intConst(intTy(32), 0), intTy(32), size);
	BOOST_CHECK_THROW(check(badFirst), runtime_error);

	// Second operand not a zero value
	Inst badSecond = Inst(Alloca, ptr, intConst(intTy(32), 1), size);
	BOOST_CHECK_THROW(check(badSecond), runtime_error);

	// Third operand not an integer
	Term floatSize = floatConst(floatTy(), "1.0");
	Inst badThird = Inst(Alloca, ptr, type, floatSize);
	BOOST_CHECK_THROW(check(badThird), runtime_error);

	// Result not a pointer type
	Term intResult = makeIntVar("result");
	Inst badResult = Inst(Alloca, intResult, type, size);
	BOOST_CHECK_THROW(check(badResult), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_assign) {
	// Valid assignment
	Term lhs = makeIntVar("x");
	Term rhs = intConst(intTy(32), 42);
	Inst valid(Assign, {lhs, rhs});
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooFew(Assign, {lhs});
	BOOST_CHECK_THROW(check(tooFew), runtime_error);

	// Left hand side not a variable
	Inst badLhs(Assign, {intConst(intTy(32), 0), rhs});
	BOOST_CHECK_THROW(check(badLhs), runtime_error);

	// Mismatched types
	Term floatRhs = floatConst(floatTy(), "42.0");
	Inst mismatch(Assign, {lhs, floatRhs});
	BOOST_CHECK_THROW(check(mismatch), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_block) {
	// Valid block
	Inst valid = block(Ref("L1"));
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooMany(Block, {makeLabel("L1"), makeLabel("L2")});
	BOOST_CHECK_THROW(check(tooMany), runtime_error);

	// Not a label
	Inst notLabel(Block, {makeIntVar("x")});
	BOOST_CHECK_THROW(check(notLabel), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_br) {
	// Valid branch
	Term cond = var(boolTy(), Ref("cond"));
	Term thenLabel = makeLabel("then");
	Term elseLabel = makeLabel("else");
	Inst valid = br(cond, thenLabel, elseLabel);
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooFew(Br, {cond, thenLabel});
	BOOST_CHECK_THROW(check(tooFew), runtime_error);

	// Condition not boolean
	Term intCond = makeIntVar("x");
	Inst badCond = br(intCond, thenLabel, elseLabel);
	BOOST_CHECK_THROW(check(badCond), runtime_error);

	// Targets not labels
	Term notLabel = makeIntVar("x");
	Inst badTarget = br(cond, notLabel, elseLabel);
	BOOST_CHECK_THROW(check(badTarget), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_jmp) {
	// Valid jump
	Inst valid = jmp(makeLabel("L1"));
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooMany(Jmp, {makeLabel("L1"), makeLabel("L2")});
	BOOST_CHECK_THROW(check(tooMany), runtime_error);

	// Target not a label
	Inst badTarget = jmp(makeIntVar("x"));
	BOOST_CHECK_THROW(check(badTarget), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_phi) {
	// Valid phi
	Term result = makeIntVar("x");
	Term val1 = intConst(intTy(32), 1);
	Term val2 = intConst(intTy(32), 2);
	Term label1 = makeLabel("L1");
	Term label2 = makeLabel("L2");
	Inst valid(Phi, {result, val1, label1, val2, label2});
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands (must be 1 + 2n where n >= 1)
	Inst tooFew(Phi, {result, val1});
	BOOST_CHECK_THROW(check(tooFew), runtime_error);

	// First operand not a variable
	Inst badFirst(Phi, {intConst(intTy(32), 0), val1, label1, val2, label2});
	BOOST_CHECK_THROW(check(badFirst), runtime_error);

	// Value type mismatch
	Term floatVal = floatConst(floatTy(), "1.0");
	Inst typeMismatch(Phi, {result, floatVal, label1, val2, label2});
	BOOST_CHECK_THROW(check(typeMismatch), runtime_error);

	// Label operand not a label
	Inst badLabel(Phi, {result, val1, makeIntVar("x"), val2, label2});
	BOOST_CHECK_THROW(check(badLabel), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_ret) {
	// Valid return
	Term val = intConst(intTy(32), 42);
	Inst valid(Ret, {val});
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooMany(Ret, {val, val});
	BOOST_CHECK_THROW(check(tooMany), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_ret_void) {
	// Valid void return
	Inst valid(RetVoid);
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooMany(RetVoid, {intConst(intTy(32), 0)});
	BOOST_CHECK_THROW(check(tooMany), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_store) {
	// Valid store
	Term val = intConst(intTy(32), 42);
	Term ptr = makePtrVar("ptr");
	Inst valid(Store, {val, ptr});
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooFew(Store, {val});
	BOOST_CHECK_THROW(check(tooFew), runtime_error);

	// Second operand not a pointer
	Term notPtr = makeIntVar("x");
	Inst badPtr(Store, {val, notPtr});
	BOOST_CHECK_THROW(check(badPtr), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_switch) {
	// Valid switch
	Term val = makeIntVar("x");
	Term defaultLabel = makeLabel("default");
	Term case1 = intConst(intTy(32), 1);
	Term label1 = makeLabel("L1");
	Term case2 = intConst(intTy(32), 2);
	Term label2 = makeLabel("L2");
	Inst valid(Switch, {val, defaultLabel, case1, label1, case2, label2});
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands (must be 2 + 2n where n >= 0)
	Inst tooFew(Switch, {val});
	BOOST_CHECK_THROW(check(tooFew), runtime_error);

	// Default target not a label
	Inst badDefault(Switch, {val, makeIntVar("x"), case1, label1});
	BOOST_CHECK_THROW(check(badDefault), runtime_error);

	// Case value type mismatch
	Term floatCase = floatConst(floatTy(), "1.0");
	Inst typeMismatch(Switch, {val, defaultLabel, floatCase, label1});
	BOOST_CHECK_THROW(check(typeMismatch), runtime_error);

	// Case target not a label
	Inst badTarget(Switch, {val, defaultLabel, case1, makeIntVar("x")});
	BOOST_CHECK_THROW(check(badTarget), runtime_error);
}

BOOST_AUTO_TEST_CASE(test_unreachable) {
	// Valid unreachable
	Inst valid(Unreachable);
	BOOST_CHECK_NO_THROW(check(valid));

	// Wrong number of operands
	Inst tooMany(Unreachable, {makeIntVar("x")});
	BOOST_CHECK_THROW(check(tooMany), runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
--check-inst.cpp
// Helper function to parse LLVM IR string
std::unique_ptr<Module> parseString(const std::string& input) {
	return std::unique_ptr<Module>(Parser("test.ll", input).module);
}

BOOST_AUTO_TEST_SUITE(ParserTests)

// Test basic type parsing
BOOST_AUTO_TEST_CASE(BasicTypes) {
	auto module = parseString(R"(
define void @test() {
    ret void
}
)");

	BOOST_CHECK_EQUAL(module->defs[0].rty().kind(), VoidKind);
}

// Test integer types with different bit widths
BOOST_AUTO_TEST_CASE(IntegerTypes) {
	auto module = parseString(R"(
define i32 @test(i1 %cond, i64 %val) {
    ret i32 0
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func.rty().kind(), IntKind);
	BOOST_CHECK_EQUAL(func.rty().len(), 32);
	BOOST_CHECK_EQUAL(func.params()[0].ty().len(), 1);
	BOOST_CHECK_EQUAL(func.params()[1].ty().len(), 64);
}

// Test floating point types
BOOST_AUTO_TEST_CASE(FloatingPointTypes) {
	auto module = parseString(R"(
define double @test(float %x) {
    ret double 0.0
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func.rty().kind(), DoubleKind);
	BOOST_CHECK_EQUAL(func.params()[0].ty().kind(), FloatKind);
}

// Test pointer types
BOOST_AUTO_TEST_CASE(PointerTypes) {
	auto module = parseString(R"(
define ptr @allocate() {
    %ptr = alloca i32, i32 1
    ret ptr %ptr
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func.rty().kind(), PtrKind);
}

// Test array types
BOOST_AUTO_TEST_CASE(ArrayTypes) {
	auto module = parseString(R"(
define void @test() {
    %arr = alloca [4 x i32], i32 1
    ret void
}
)");

	auto inst = module->defs[0][0]; // First instruction
	BOOST_CHECK_EQUAL(inst.opcode(), Alloca);
	auto arrayType = inst[1].ty();
	BOOST_CHECK_EQUAL(arrayType.kind(), ArrayKind);
	BOOST_CHECK_EQUAL(arrayType.len(), 4);
	BOOST_CHECK_EQUAL(arrayType[0].kind(), IntKind);
	BOOST_CHECK_EQUAL(arrayType[0].len(), 32);
}

// Test vector types
BOOST_AUTO_TEST_CASE(VectorTypes) {
	auto module = parseString(R"(
define void @test() {
    %vec = alloca <4 x float>, i32 1
    ret void
}
)");

	auto inst = module->defs[0][0];
	BOOST_CHECK_EQUAL(inst.opcode(), Alloca);
	auto vecType = inst[1].ty();
	BOOST_CHECK_EQUAL(vecType.kind(), VecKind);
	BOOST_CHECK_EQUAL(vecType.len(), 4);
	BOOST_CHECK_EQUAL(vecType[0].kind(), FloatKind);
}

// Test basic arithmetic expressions
BOOST_AUTO_TEST_CASE(ArithmeticExpressions) {
	auto module = parseString(R"(
define i32 @test(i32 %a, i32 %b) {
entry:
    %sum = add i32 %a, %b
    %diff = sub i32 %sum, %a
    %prod = mul i32 %diff, %b
    %quot = sdiv i32 %prod, %b
    ret i32 %quot
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func[1][1].tag(), Add);
	BOOST_CHECK_EQUAL(func[2][1].tag(), Sub);
	BOOST_CHECK_EQUAL(func[3][1].tag(), Mul);
	BOOST_CHECK_EQUAL(func[4][1].tag(), SDiv);
}

// Test comparison expressions
BOOST_AUTO_TEST_CASE(ComparisonExpressions) {
	auto module = parseString(R"(
define i1 @test(i32 %a, i32 %b) {
entry:
    %eq = icmp eq i32 %a, %b
    %lt = icmp slt i32 %a, %b
    %gt = icmp sgt i32 %a, %b
    %result = and i1 %eq, %lt
    ret i1 %result
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func[1][1].tag(), Eq);
	BOOST_CHECK_EQUAL(func[2][1].tag(), SLt);
	// Note: sgt is internally represented as SLt with swapped operands
	BOOST_CHECK_EQUAL(func[4][1].tag(), And);
}

// Test control flow instructions
BOOST_AUTO_TEST_CASE(ControlFlow) {
	auto module = parseString(R"(
define void @test(i1 %cond) {
entry:
    br i1 %cond, label %then, label %else
then:
    br label %merge
else:
    br label %merge
merge:
    ret void
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func[0].opcode(), Block);
	BOOST_CHECK_EQUAL(func[1].opcode(), Br);
	BOOST_CHECK_EQUAL(func[2].opcode(), Block);
	BOOST_CHECK_EQUAL(func[3].opcode(), Jmp);
}

// Test memory operations
BOOST_AUTO_TEST_CASE(MemoryOperations) {
	auto module = parseString(R"(
define i32 @test() {
entry:
    %ptr = alloca i32, i32 1
    store i32 42, ptr %ptr
    %val = load i32, ptr %ptr
    ret i32 %val
}
)");

	auto& func = module->defs[0];
	BOOST_CHECK_EQUAL(func[1].opcode(), Alloca);
	BOOST_CHECK_EQUAL(func[2].opcode(), Store);
	BOOST_CHECK_EQUAL(func[3][1].tag(), Load);
}

// Test function declarations
BOOST_AUTO_TEST_CASE(FunctionDeclarations) {
	auto module = parseString(R"(
declare i32 @external_func(i32)
define i32 @test(i32 %x) {
    %result = call i32 @external_func(i32 %x)
    ret i32 %result
}
)");

	BOOST_CHECK_EQUAL(module->decls.size(), 1);
	BOOST_CHECK_EQUAL(module->defs.size(), 1);
	BOOST_CHECK_EQUAL(module->decls[0].size(), 0); // Declaration has no body
	BOOST_CHECK_EQUAL(module->defs[0].size(), 2);  // Definition has body
}

// Test error handling
BOOST_AUTO_TEST_CASE(ErrorHandling) {
	BOOST_CHECK_THROW(parseString("define"), std::runtime_error);
	BOOST_CHECK_THROW(parseString("define @invalid"), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
--parse-types.cpp
// Test atomic term outputs
BOOST_AUTO_TEST_CASE(test_atomic_terms) {
	std::ostringstream os;

	// Test float constant
	Term floatTerm = Term(Float, floatTy(), Ref("3.14"));
	os << floatTerm;
	BOOST_CHECK_EQUAL(os.str(), "3.14");
	os.str("");

	// Test integer constant (bool)
	Term boolTerm = intConst(boolTy(), 1);
	os << boolTerm;
	BOOST_CHECK_EQUAL(os.str(), "true");
	os.str("");

	// Test integer constant (non-bool)
	Term intTerm = intConst(intTy(32), 42);
	os << intTerm;
	BOOST_CHECK_EQUAL(os.str(), "42");
	os.str("");

	// Test null constant
	os << nullConst;
	BOOST_CHECK_EQUAL(os.str(), "null");
	os.str("");

	// Test variable reference
	Term varTerm = var(intTy(32), Ref("x"));
	os << varTerm;
	BOOST_CHECK_EQUAL(os.str(), "%x");
}

// Test arithmetic operations
BOOST_AUTO_TEST_CASE(test_arithmetic_operations) {
	std::ostringstream os;

	// Setup operands
	Term a = var(intTy(32), Ref("a"));
	Term b = var(intTy(32), Ref("b"));

	// Test addition
	Term add = Term(Add, intTy(32), a, b);
	os << add;
	BOOST_CHECK_EQUAL(os.str(), "add (i32 %a, i32 %b)");
	os.str("");

	// Test subtraction
	Term sub = Term(Sub, intTy(32), a, b);
	os << sub;
	BOOST_CHECK_EQUAL(os.str(), "sub (i32 %a, i32 %b)");
	os.str("");

	// Test multiplication
	Term mul = Term(Mul, intTy(32), a, b);
	os << mul;
	BOOST_CHECK_EQUAL(os.str(), "mul (i32 %a, i32 %b)");
}

// Test floating-point operations
BOOST_AUTO_TEST_CASE(test_floating_point_operations) {
	std::ostringstream os;

	// Setup operands
	Term a = var(floatTy(), Ref("x"));
	Term b = var(floatTy(), Ref("y"));

	// Test floating-point addition
	Term fadd = Term(FAdd, floatTy(), a, b);
	os << fadd;
	BOOST_CHECK_EQUAL(os.str(), "fadd (float %x, float %y)");
	os.str("");

	// Test floating-point negation (unary)
	Term fneg = Term(FNeg, floatTy(), a);
	os << fneg;
	BOOST_CHECK_EQUAL(os.str(), "fneg (float %x)");
}

// Test comparison operations
BOOST_AUTO_TEST_CASE(test_comparison_operations) {
	std::ostringstream os;

	// Setup operands
	Term a = var(intTy(32), Ref("a"));
	Term b = var(intTy(32), Ref("b"));

	// Test equality comparison
	Term eq = Term(Eq, boolTy(), a, b);
	os << eq;
	BOOST_CHECK_EQUAL(os.str(), "icmp eq (i32 %a, i32 %b)");
	os.str("");

	// Test signed less than or equal
	Term sle = Term(SLe, boolTy(), a, b);
	os << sle;
	BOOST_CHECK_EQUAL(os.str(), "icmp sle (i32 %a, i32 %b)");
	os.str("");

	// Test unsigned less than
	Term ult = Term(ULt, boolTy(), a, b);
	os << ult;
	BOOST_CHECK_EQUAL(os.str(), "icmp ult (i32 %a, i32 %b)");
}

// Test bitwise operations
BOOST_AUTO_TEST_CASE(test_bitwise_operations) {
	std::ostringstream os;

	// Setup operands
	Term a = var(intTy(32), Ref("a"));
	Term b = var(intTy(32), Ref("b"));

	// Test AND
	Term and_op = Term(And, intTy(32), a, b);
	os << and_op;
	BOOST_CHECK_EQUAL(os.str(), "and (i32 %a, i32 %b)");
	os.str("");

	// Test OR
	Term or_op = Term(Or, intTy(32), a, b);
	os << or_op;
	BOOST_CHECK_EQUAL(os.str(), "or (i32 %a, i32 %b)");
	os.str("");

	// Test XOR
	Term xor_op = Term(Xor, intTy(32), a, b);
	os << xor_op;
	BOOST_CHECK_EQUAL(os.str(), "xor (i32 %a, i32 %b)");
	os.str("");

	// Test shift operations
	Term shl = Term(Shl, intTy(32), a, b);
	os << shl;
	BOOST_CHECK_EQUAL(os.str(), "shl (i32 %a, i32 %b)");
	os.str("");

	Term lshr = Term(LShr, intTy(32), a, b);
	os << lshr;
	BOOST_CHECK_EQUAL(os.str(), "lshr (i32 %a, i32 %b)");
}

// Test division operations
BOOST_AUTO_TEST_CASE(test_division_operations) {
	std::ostringstream os;

	// Setup operands
	Term a = var(intTy(32), Ref("a"));
	Term b = var(intTy(32), Ref("b"));

	// Test signed division
	Term sdiv = Term(SDiv, intTy(32), a, b);
	os << sdiv;
	BOOST_CHECK_EQUAL(os.str(), "sdiv (i32 %a, i32 %b)");
	os.str("");

	// Test unsigned division
	Term udiv = Term(UDiv, intTy(32), a, b);
	os << udiv;
	BOOST_CHECK_EQUAL(os.str(), "udiv (i32 %a, i32 %b)");
	os.str("");

	// Test signed remainder
	Term srem = Term(SRem, intTy(32), a, b);
	os << srem;
	BOOST_CHECK_EQUAL(os.str(), "srem (i32 %a, i32 %b)");
	os.str("");

	// Test unsigned remainder
	Term urem = Term(URem, intTy(32), a, b);
	os << urem;
	BOOST_CHECK_EQUAL(os.str(), "urem (i32 %a, i32 %b)");
}
--term-output.cpp
BOOST_AUTO_TEST_SUITE(WrapTests)

// Test valid identifiers that shouldn't need quotes
BOOST_AUTO_TEST_CASE(ValidIdentifiers) {
	// Basic valid identifiers
	BOOST_CHECK_EQUAL(wrap("foo"), "foo");
	BOOST_CHECK_EQUAL(wrap("_foo"), "_foo");
	BOOST_CHECK_EQUAL(wrap(".foo"), ".foo");
	BOOST_CHECK_EQUAL(wrap("foo_bar"), "foo_bar");
	BOOST_CHECK_EQUAL(wrap("foo.bar"), "foo.bar");

	// Valid identifiers with numbers
	BOOST_CHECK_EQUAL(wrap("foo123"), "foo123");
	BOOST_CHECK_EQUAL(wrap("foo_123"), "foo_123");

	// Valid identifiers with mixed case
	BOOST_CHECK_EQUAL(wrap("FooBar"), "FooBar");
	BOOST_CHECK_EQUAL(wrap("fooBar"), "fooBar");
}

// Test invalid identifiers that need quotes
BOOST_AUTO_TEST_CASE(InvalidIdentifiers) {
	// Empty string
	BOOST_CHECK_EQUAL(wrap(""), "\"\"");

	// Starting with number
	BOOST_CHECK_EQUAL(wrap("123foo"), "\"123foo\"");

	// Contains spaces
	BOOST_CHECK_EQUAL(wrap("foo bar"), "\"foo bar\"");

	// Contains special characters
	BOOST_CHECK_EQUAL(wrap("foo+bar"), "\"foo+bar\"");
	BOOST_CHECK_EQUAL(wrap("foo-bar"), "foo-bar");
	BOOST_CHECK_EQUAL(wrap("foo@bar"), "\"foo@bar\"");
}

// Test strings with characters that need hex escaping
BOOST_AUTO_TEST_CASE(HexEscapes) {
	// Quote character
	BOOST_CHECK_EQUAL(wrap("foo\"bar"), "\"foo\\22bar\"");

	// Multiple quotes
	BOOST_CHECK_EQUAL(wrap("\"foo\"bar\""), "\"\\22foo\\22bar\\22\"");

	// Non-printable characters
	BOOST_CHECK_EQUAL(wrap("foo\nbar"), "\"foo\\0abar\"");
	BOOST_CHECK_EQUAL(wrap("foo\tbar"), "\"foo\\09bar\"");
	BOOST_CHECK_EQUAL(wrap("foo\rbar"), "\"foo\\0dbar\"");

	// Control characters
	char control1 = 1;
	char control2 = 31;
	BOOST_CHECK_EQUAL(wrap(string("foo") + control1 + "bar"), "\"foo\\01bar\"");
	BOOST_CHECK_EQUAL(wrap(string("foo") + control2 + "bar"), "\"foo\\1fbar\"");

	// Extended ASCII
	char extended1 = char(128);
	char extended2 = char(255);
	BOOST_CHECK_EQUAL(wrap(string("foo") + extended1 + "bar"), "\"foo\\80bar\"");
	BOOST_CHECK_EQUAL(wrap(string("foo") + extended2 + "bar"), "\"foo\\ffbar\"");
}

// Test strings with backslashes
BOOST_AUTO_TEST_CASE(Backslashes) {
	// Single backslash
	BOOST_CHECK_EQUAL(wrap("foo\\bar"), "\"foo\\\\bar\"");

	// Multiple backslashes
	BOOST_CHECK_EQUAL(wrap("foo\\\\bar"), "\"foo\\\\\\\\bar\"");

	// Backslash at start/end
	BOOST_CHECK_EQUAL(wrap("\\foo"), "\"\\\\foo\"");
	BOOST_CHECK_EQUAL(wrap("foo\\"), "\"foo\\\\\"");
}

// Test edge cases
BOOST_AUTO_TEST_CASE(EdgeCases) {
	// Single character strings
	BOOST_CHECK_EQUAL(wrap("a"), "a");
	BOOST_CHECK_EQUAL(wrap("\\"), "\"\\\\\"");
	BOOST_CHECK_EQUAL(wrap("\""), "\"\\22\"");
	BOOST_CHECK_EQUAL(wrap("\n"), "\"\\0a\"");

	// Mixed special cases
	BOOST_CHECK_EQUAL(wrap("foo\\\"\nbar"), "\"foo\\\\\\22\\0abar\"");

	string controlChars;
	controlChars += char(1);
	controlChars += char(2);
	controlChars += char(3);
	BOOST_CHECK_EQUAL(wrap(controlChars), "\"\\01\\02\\03\"");

	// Valid identifier characters mixed with invalid ones
	BOOST_CHECK_EQUAL(wrap("foo_bar+baz"), "\"foo_bar+baz\"");
	BOOST_CHECK_EQUAL(wrap("foo.bar@baz"), "\"foo.bar@baz\"");
}

BOOST_AUTO_TEST_SUITE_END()
--wrap
// Disable max/min macros to avoid conflicts with std::numeric_limits
#undef max
#undef min

BOOST_AUTO_TEST_SUITE(RefStreamOperatorTests)

// Test numeric references
BOOST_AUTO_TEST_CASE(NumericRef) {
	Ref ref(static_cast<size_t>(42));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "42");
}

// Test valid identifier strings that don't need quoting
BOOST_AUTO_TEST_CASE(ValidIdentifierString) {
	Ref ref(std::string("valid_name"));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "valid_name");
}

// Test string starting with digit that needs quoting
BOOST_AUTO_TEST_CASE(StringStartingWithDigit) {
	Ref ref(std::string("123name"));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "\"123name\"");
}

// Test string with special characters that need escaping
BOOST_AUTO_TEST_CASE(StringWithSpecialChars) {
	Ref ref(std::string("test\\path"));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "\"test\\\\path\"");
}

// Test string with quotes and special characters
BOOST_AUTO_TEST_CASE(StringWithQuotesAndSpecials) {
	Ref ref(std::string("\"quoted\""));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "\"\\22quoted\\22\"");
}

// Test empty string
BOOST_AUTO_TEST_CASE(EmptyString) {
	Ref ref(std::string(""));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "\"\"");
}

// Test string with spaces
BOOST_AUTO_TEST_CASE(StringWithSpaces) {
	Ref ref(std::string("my variable"));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "\"my variable\"");
}

// Test string with non-printable characters
BOOST_AUTO_TEST_CASE(StringWithNonPrintable) {
	Ref ref(std::string("test\n\tname"));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "\"test\\0a\\09name\"");
}

// Test large numeric value
BOOST_AUTO_TEST_CASE(LargeNumericRef) {
	const size_t large_value = (std::numeric_limits<size_t>::max)(); // Use parentheses to avoid macro issues
	Ref ref(large_value);
	std::ostringstream oss;
	oss << ref;
	std::ostringstream expected;
	expected << large_value;
	BOOST_CHECK_EQUAL(oss.str(), expected.str());
}

// Test zero numeric value
BOOST_AUTO_TEST_CASE(ZeroNumericRef) {
	Ref ref(static_cast<size_t>(0));
	std::ostringstream oss;
	oss << ref;
	BOOST_CHECK_EQUAL(oss.str(), "0");
}

BOOST_AUTO_TEST_SUITE_END()
--ref-output.cpp
BOOST_AUTO_TEST_SUITE(InstStreamOperatorTests)

// Helper function to compare instruction string output
std::string instToString(const Inst& inst) {
	std::ostringstream oss;
	oss << inst;
	return oss.str();
}

BOOST_AUTO_TEST_CASE(EmptyInstructions) {
	// Test ret void
	Inst retVoid(RetVoid);
	BOOST_CHECK_EQUAL(instToString(retVoid), "ret void");

	// Test unreachable
	Inst unreachableInst(Unreachable);
	BOOST_CHECK_EQUAL(instToString(unreachableInst), "unreachable");
}

BOOST_AUTO_TEST_CASE(AllocaInstruction) {
	// Test alloca with constant size
	Term var1 = var(ptrTy(), Ref("ptr"));
	Term type = zeroVal(intTy(32));
	Term size = intConst(intTy(32), 1);
	Inst allocaInst = alloca(var1, type.ty(), size);

	BOOST_CHECK_EQUAL(instToString(allocaInst), "%ptr = alloca i32");
}

BOOST_AUTO_TEST_CASE(AssignInstruction) {
	// Test simple assignment
	Term lhs = var(intTy(32), Ref("result"));
	Term rhs = intConst(intTy(32), 42);
	Inst assignInst = assign(lhs, rhs);

	BOOST_CHECK_EQUAL(instToString(assignInst), "%result = 42");
}

BOOST_AUTO_TEST_CASE(BlockInstruction) {
	// Test block label
	Inst blockInst = block(Ref("entry"));
	BOOST_CHECK_EQUAL(instToString(blockInst), "entry:");
}

BOOST_AUTO_TEST_CASE(BranchInstructions) {
	// Test conditional branch
	Term cond = var(boolTy(), Ref("cond"));
	Inst brInst = br(cond, Ref("true_bb"), Ref("false_bb"));
	BOOST_CHECK_EQUAL(instToString(brInst), "br i1 %cond, label %true_bb, label %false_bb");

	// Test unconditional branch
	Inst jmpInst = jmp(Ref("target_bb"));
	BOOST_CHECK_EQUAL(instToString(jmpInst), "br label %target_bb");
}

BOOST_AUTO_TEST_CASE(PhiInstruction) {
	// Test phi with two incoming values
	vector<Term> phiOps;
	phiOps.push_back(var(intTy(32), Ref("result"))); // Result variable
	phiOps.push_back(intConst(intTy(32), 1));		 // First value
	phiOps.push_back(label(Ref("bb1")));			 // First label
	phiOps.push_back(intConst(intTy(32), 2));		 // Second value
	phiOps.push_back(label(Ref("bb2")));			 // Second label

	Inst phiInst(Phi, phiOps);
	BOOST_CHECK_EQUAL(instToString(phiInst), "%result = phi i32 [ 1, %bb1 ], [ 2, %bb2 ]");
}

BOOST_AUTO_TEST_CASE(ReturnInstructions) {
	// Test return with value
	Term retVal = intConst(intTy(32), 42);
	Inst retInst = ret(retVal);
	BOOST_CHECK_EQUAL(instToString(retInst), "ret i32 42");
}

BOOST_AUTO_TEST_CASE(StoreInstruction) {
	// Test store instruction
	Term val = intConst(intTy(32), 42);
	Term ptr = var(ptrTy(), Ref("ptr"));
	Inst storeInst(Store, {val, ptr});

	BOOST_CHECK_EQUAL(instToString(storeInst), "store i32 42, ptr %ptr");
}

BOOST_AUTO_TEST_CASE(SwitchInstruction) {
	// Test switch with multiple cases
	Term switchVal = var(intTy(32), Ref("val"));
	Term defaultLabel = label(Ref("default"));
	Term case1Val = intConst(intTy(32), 1);
	Term case1Label = label(Ref("case1"));
	Term case2Val = intConst(intTy(32), 2);
	Term case2Label = label(Ref("case2"));

	vector<Term> switchOps = {switchVal, defaultLabel, case1Val, case1Label, case2Val, case2Label};

	Inst switchInst(Switch, switchOps);
	BOOST_CHECK_EQUAL(instToString(switchInst), "switch i32 %val, label %default [\n"
												"    i32 1, label %case1\n"
												"    i32 2, label %case2\n"
												"  ]");
}

BOOST_AUTO_TEST_CASE(DropInstruction) {
	// Test drop instruction
	Term callExpr = call(voidTy(), globalRef(funcTy(voidTy(), {}), Ref("func")), {});
	Inst dropInst(Drop, {callExpr});

	BOOST_CHECK_EQUAL(instToString(dropInst), "call void @func()");
}

// Test invalid instructions throw assertions
BOOST_AUTO_TEST_CASE(InvalidInstructions) {
	// Test invalid empty instruction
	BOOST_CHECK_THROW(instToString(Inst(Alloca)), std::runtime_error);

	// Test Alloca with wrong number of operands
	vector<Term> invalidAllocaOps = {var(ptrTy(), Ref("%ptr"))};
	BOOST_CHECK_THROW(instToString(Inst(Alloca, invalidAllocaOps)), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
--inst-output.cpp
BOOST_AUTO_TEST_SUITE(FuncOperatorTests)

// Helper function to get string output of a Func
string toString(const Fn& f) {
	ostringstream oss;
	oss << f;
	return oss.str();
}

BOOST_AUTO_TEST_CASE(EmptyFunctionDeclaration) {
	// Test a simple void function declaration with no parameters
	auto f = Fn(voidTy(), Ref("empty"), vector<Term>{});
	BOOST_CHECK_EQUAL(toString(f), "declare void @empty()");
}

BOOST_AUTO_TEST_CASE(SimpleIntFunctionDeclaration) {
	// Test an int32 function declaration with one parameter
	vector<Term> params = {var(intTy(32), Ref(0))};
	auto f = Fn(intTy(32), Ref("simple"), params);
	BOOST_CHECK_EQUAL(toString(f), "declare i32 @simple(i32 %0)");
}

BOOST_AUTO_TEST_CASE(MultiParamFunctionDeclaration) {
	// Test function with multiple parameters of different types
	vector<Term> params = {var(intTy(32), Ref(0)), var(ptrTy(), Ref(1)), var(doubleTy(), Ref(2))};
	auto f = Fn(intTy(64), Ref("multi"), params);
	BOOST_CHECK_EQUAL(toString(f), "declare i64 @multi(i32 %0, ptr %1, double %2)");
}

BOOST_AUTO_TEST_CASE(EmptyFunctionDefinition) {
	// Test a function definition with no parameters and empty body
	vector<Term> params;
	vector<Inst> body;
	auto f = Fn(voidTy(), Ref("empty_def"), params, body);
	BOOST_CHECK_EQUAL(toString(f), "declare void @empty_def()");
}

BOOST_AUTO_TEST_CASE(SimpleFunctionDefinition) {
	// Test a function definition with one parameter and simple body
	vector<Term> params = {var(intTy(32), Ref(0))};
	vector<Inst> body = {alloca(var(ptrTy(), Ref(1)), intTy(32), intConst(intTy(32), 1)), store(params[0], var(ptrTy(), Ref(1))),
						 ret()};
	auto f = Fn(voidTy(), Ref("simple_def"), params, body);

	string expected = "define void @simple_def(i32 %0) {\n"
					  "  %1 = alloca i32\n"
					  "  store i32 %0, ptr %1\n"
					  "  ret void\n"
					  "}";

	BOOST_CHECK_EQUAL(toString(f), expected);
}

BOOST_AUTO_TEST_CASE(ComplexFunctionDefinition) {
	// Test a function with control flow and multiple blocks
	vector<Term> params = {var(intTy(32), Ref(0))};
	vector<Inst> body = {block(Ref("entry")),
						 alloca(var(ptrTy(), Ref(1)), intTy(32), intConst(intTy(32), 1)),
						 store(params[0], var(ptrTy(), Ref(1))),
						 br(trueConst, Ref("then"), Ref("else")),

						 block(Ref("then")),
						 ret(intConst(intTy(32), 1)),

						 block(Ref("else")),
						 ret(intConst(intTy(32), 0))};

	auto f = Fn(intTy(32), Ref("complex_def"), params, body);

	string expected = "define i32 @complex_def(i32 %0) {\n"
					  "entry:\n"
					  "  %1 = alloca i32\n"
					  "  store i32 %0, ptr %1\n"
					  "  br i1 true, label %then, label %else\n"
					  "then:\n"
					  "  ret i32 1\n"
					  "else:\n"
					  "  ret i32 0\n"
					  "}";

	BOOST_CHECK_EQUAL(toString(f), expected);
}

BOOST_AUTO_TEST_CASE(QuotedFunctionName) {
	// Test function with a name that needs quoting
	auto f = Fn(voidTy(), Ref("1invalid"), vector<Term>{});
	BOOST_CHECK_EQUAL(toString(f), "declare void @\"1invalid\"()");
}

BOOST_AUTO_TEST_CASE(EscapedFunctionName) {
	// Test function with a name that needs escaping
	auto f = Fn(voidTy(), Ref("name\\with\"quotes"), vector<Term>{});
	BOOST_CHECK_EQUAL(toString(f), "declare void @\"name\\\\with\\22quotes\"()");
}

BOOST_AUTO_TEST_SUITE_END()
--fn.cpp
// Helper function to create a cpp_int with specific bits set
cpp_int create_test_number(const std::vector<size_t>& set_bits) {
	cpp_int result = 0;
	for (size_t bit : set_bits) {
		bit_set(result, bit);
	}
	return result;
}

BOOST_AUTO_TEST_SUITE(TruncateBitsTests)

BOOST_AUTO_TEST_CASE(test_zero_input) {
	cpp_int input = 0;
	BOOST_CHECK_EQUAL(truncate_to_bits(input, 32), 0);
}

BOOST_AUTO_TEST_CASE(test_small_number) {
	cpp_int input = 42; // 101010 in binary
	BOOST_CHECK_EQUAL(truncate_to_bits(input, 32), 42);
}

BOOST_AUTO_TEST_CASE(test_exact_bit_length) {
	// Create number with exactly 8 bits: 10101010 (170 in decimal)
	cpp_int input = create_test_number({1, 3, 5, 7});
	BOOST_CHECK_EQUAL(truncate_to_bits(input, 8), 170);
}

BOOST_AUTO_TEST_CASE(test_truncation_needed) {
	// Create number with bits set beyond desired length
	cpp_int input = create_test_number({0, 2, 4, 8, 16, 32});
	cpp_int expected = create_test_number({0, 2, 4});
	BOOST_CHECK_EQUAL(truncate_to_bits(input, 5), expected);
}

BOOST_AUTO_TEST_CASE(test_large_numbers) {
	// Test with a very large number
	cpp_int input = cpp_int(1) << 1000;
	input -= 1; // Creates a number with 1000 ones

	// Truncate to 64 bits
	cpp_int expected = (cpp_int(1) << 64) - 1;
	BOOST_CHECK_EQUAL(truncate_to_bits(input, 64), expected);
}

BOOST_AUTO_TEST_CASE(test_single_bit) {
	cpp_int input = create_test_number({0, 1, 2, 3, 4});
	BOOST_CHECK_EQUAL(truncate_to_bits(input, 1), 1);
}

BOOST_AUTO_TEST_CASE(test_invalid_input) {
	cpp_int input = 42;
	BOOST_CHECK_THROW(truncate_to_bits(input, 0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_specific_bit_patterns) {
	// Test alternating bit pattern
	cpp_int alternating = create_test_number({0, 2, 4, 6, 8, 10});
	BOOST_CHECK_EQUAL(truncate_to_bits(alternating, 4), 5); // Should keep only 0101

	// Test consecutive bits
	cpp_int consecutive = create_test_number({0, 1, 2, 3, 4, 5});
	BOOST_CHECK_EQUAL(truncate_to_bits(consecutive, 4), 15); // Should keep only 1111
}

BOOST_AUTO_TEST_SUITE_END()
--int-bits.cpp
namespace fw = fixed_width_ops;

BOOST_AUTO_TEST_SUITE(FixedWidthOpsTests)

BOOST_AUTO_TEST_CASE(test_arithmetic) {
	// Test addition with overflow
	BOOST_CHECK_EQUAL(fw::add(15, 1, 4), 0);

	// Test subtraction with underflow
	BOOST_CHECK_EQUAL(fw::sub(0, 1, 4), 15);

	// Test multiplication with overflow
	BOOST_CHECK_EQUAL(fw::mul(8, 2, 4), 0);
}

BOOST_AUTO_TEST_CASE(test_signed_division) {
	// Test signed division with negative numbers
	// In 4 bits: 7 = 0111, -3 = 1101 (13 unsigned)
	BOOST_CHECK_EQUAL(fw::sdiv(7, 13, 4), 14); // 7 / -3 = -2 (14 unsigned)

	// Test signed remainder
	BOOST_CHECK_EQUAL(fw::srem(7, 13, 4), 1); // 7 % -3 = 1
}

BOOST_AUTO_TEST_CASE(test_shifts) {
	// Test logical shifts
	BOOST_CHECK_EQUAL(fw::shl(5, 1, 4), 10 & 15);
	BOOST_CHECK_EQUAL(fw::lshr(12, 1, 4), 6);

	// Test arithmetic shift
	BOOST_CHECK_EQUAL(fw::ashr(12, 1, 4), 14); // 1100 -> 1110
}

BOOST_AUTO_TEST_CASE(test_comparisons) {
	// Test unsigned comparisons
	BOOST_CHECK_EQUAL(fw::ult(5, 10, 4), 1);
	BOOST_CHECK_EQUAL(fw::ule(5, 5, 4), 1);

	// Test signed comparisons
	// 5 = 0101, -3 = 1101 (13 unsigned)
	BOOST_CHECK_EQUAL(fw::slt(5, 13, 4), 0);  // 5 < -3 is false
	BOOST_CHECK_EQUAL(fw::sle(13, 13, 4), 1); // -3 <= -3 is true
}

BOOST_AUTO_TEST_CASE(test_edge_cases) {
	// Test division by zero
	BOOST_CHECK_THROW(fw::udiv(1, 0, 4), std::domain_error);
	BOOST_CHECK_THROW(fw::sdiv(1, 0, 4), std::domain_error);

	// Test large shifts
	BOOST_CHECK_EQUAL(fw::shl(1, 4, 4), 0);
	BOOST_CHECK_EQUAL(fw::lshr(15, 4, 4), 0);
	BOOST_CHECK_EQUAL(fw::ashr(8, 4, 4), 15); // Sign extends 1000 to 1111
}

BOOST_AUTO_TEST_SUITE_END()
--fixed-width.cpp
// Helper functions to create test functions
namespace {
Term createIntVar(const string& name, size_t bits = 32) {
	return var(intTy(bits), Ref(name));
}

Term createBoolVar(const string& name) {
	return var(boolTy(), Ref(name));
}

Term createPtrVar(const string& name) {
	return var(ptrTy(), Ref(name));
}
} // namespace

BOOST_AUTO_TEST_CASE(ValidSimpleFunction) {
	// Create a simple function that takes an int and returns an int
	vector<Term> params = {createIntVar("x")};
	vector<Inst> body = {
		block(Ref("entry")),
		ret(params[0]) // Just return the parameter
	};

	Fn f(intTy(32), Ref("simple"), params, body);

	// Should not throw
	BOOST_CHECK_NO_THROW(check(f));
}

BOOST_AUTO_TEST_CASE(ValidVoidFunction) {
	vector<Term> params = {};
	vector<Inst> body = {block(Ref("entry")), ret()};

	Fn f(voidTy(), Ref("void_func"), params, body);

	BOOST_CHECK_NO_THROW(check(f));
}

BOOST_AUTO_TEST_CASE(ValidFunctionWithBranching) {
	vector<Term> params = {createBoolVar("cond")};
	vector<Inst> body = {block(Ref("entry")), br(params[0], Ref("then"), Ref("else")),
						 block(Ref("then")),  ret(intConst(intTy(32), 1)),
						 block(Ref("else")),  ret(intConst(intTy(32), 0))};

	Fn f(intTy(32), Ref("branching"), params, body);

	BOOST_CHECK_NO_THROW(check(f));
}

BOOST_AUTO_TEST_CASE(ValidFunctionWithAlloca) {
	vector<Term> params = {};
	vector<Inst> body = {block(Ref("entry")), alloca(createPtrVar("ptr"), intTy(32), intConst(intTy(32), 1)), ret()};

	Fn f(voidTy(), Ref("alloca_func"), params, body);

	BOOST_CHECK_NO_THROW(check(f));
}

BOOST_AUTO_TEST_CASE(InvalidReturnType) {
	vector<Term> params = {};
	vector<Inst> body = {
		block(Ref("entry")), ret(intConst(intTy(32), 0)) // Returning int from void function
	};

	Fn f(voidTy(), Ref("invalid_ret"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(InvalidBranchCondition) {
	vector<Term> params = {createIntVar("not_bool")};								   // Int instead of bool
	vector<Inst> body = {block(Ref("entry")), br(params[0], Ref("then"), Ref("else")), // Using int as condition
						 block(Ref("then")),  ret(intConst(intTy(32), 1)),
						 block(Ref("else")),  ret(intConst(intTy(32), 0))};

	Fn f(intTy(32), Ref("invalid_branch"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(UndefinedLabel) {
	vector<Term> params = {};
	vector<Inst> body = {
		block(Ref("entry")),
		jmp(Ref("nonexistent")) // Jump to undefined label
	};

	Fn f(voidTy(), Ref("undefined_label"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(InvalidPhiInstruction) {
	vector<Term> params = {};
	Term result = createIntVar("result");
	vector<Inst> body = {block(Ref("entry")),
						 Inst(Phi, {result, intConst(intTy(32), 1), label(Ref("l1")), intConst(intTy(32), 2), label(Ref("l2"))}),
						 ret(result)};

	Fn f(intTy(32), Ref("phi_func"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(InconsistentVariableTypes) {
	vector<Term> params = {createIntVar("x", 32)};
	vector<Inst> body = {block(Ref("entry")),
						 // Try to assign 64-bit integer to 32-bit variable
						 assign(params[0], intConst(intTy(64), 42)), ret(params[0])};

	Fn f(intTy(32), Ref("inconsistent_types"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(ValidStore) {
	vector<Term> params = {createPtrVar("ptr"), createIntVar("val")};
	vector<Inst> body = {block(Ref("entry")), store(params[1], params[0]), ret()};

	Fn f(voidTy(), Ref("valid_store"), params, body);

	BOOST_CHECK_NO_THROW(check(f));
}

BOOST_AUTO_TEST_CASE(InvalidStore) {
	vector<Term> params = {createIntVar("not_ptr"), createIntVar("val")};
	vector<Inst> body = {block(Ref("entry")), store(params[1], params[0]), // First param should be a pointer
						 ret()};

	Fn f(voidTy(), Ref("invalid_store"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(ValidComplexFunction) {
	// Test a more complex function with multiple blocks, variables and operations
	vector<Term> params = {createIntVar("x"), createIntVar("y")};
	Term result = createIntVar("result");
	Term temp = createIntVar("temp");
	vector<Inst> body = {block(Ref("entry")),
						 assign(temp, Term(Add, intTy(32), params[0], params[1])),
						 br(Term(Eq, boolTy(), temp, intConst(intTy(32), 0)), Ref("zero"), Ref("nonzero")),

						 block(Ref("zero")),
						 assign(result, intConst(intTy(32), 42)),
						 jmp(Ref("exit")),

						 block(Ref("nonzero")),
						 assign(result, temp),
						 jmp(Ref("exit")),

						 block(Ref("exit")),
						 ret(result)};

	Fn f(intTy(32), Ref("complex"), params, body);

	BOOST_CHECK_NO_THROW(check(f));
}

BOOST_AUTO_TEST_CASE(EmptyFunction) {
	vector<Term> params = {};
	vector<Inst> body = {}; // Empty body

	Fn f(voidTy(), Ref("empty"), params, body);

	// Empty function should be invalid
	BOOST_CHECK_THROW(check(f), runtime_error);
}

BOOST_AUTO_TEST_CASE(MissingReturn) {
	vector<Term> params = {createIntVar("x")};
	vector<Inst> body = {
		block(Ref("entry")), assign(createIntVar("y"), params[0])
		// Missing return instruction
	};

	Fn f(intTy(32), Ref("no_return"), params, body);

	BOOST_CHECK_THROW(check(f), runtime_error);
}
--check-fn.cpp
BOOST_AUTO_TEST_SUITE(PhiEliminationTests)

// Helper function to create a simple function with phi nodes
Fn createTestFunction(Type returnType, const vector<Term>& params, const vector<Inst>& body) {
	return Fn(returnType, Ref("test_func"), params, body);
}

BOOST_AUTO_TEST_CASE(EmptyFunction) {
	Fn empty;
	Fn transformed = eliminatePhiNodes(empty);
	BOOST_CHECK(transformed.empty());
}

BOOST_AUTO_TEST_CASE(FunctionWithoutPhi) {
	// Create a simple function that just returns a parameter
	Type i32 = intTy(32);
	vector<Term> params = {var(i32, Ref("param"))};
	vector<Inst> body = {block(Ref("entry")), ret(var(i32, Ref("param")))};

	Fn func = createTestFunction(i32, params, body);
	Fn transformed = eliminatePhiNodes(func);

	BOOST_CHECK_EQUAL(transformed.size(), func.size());
	for (size_t i = 0; i < func.size(); i++) {
		BOOST_CHECK(transformed[i] == func[i]);
	}
}

BOOST_AUTO_TEST_CASE(SimplePhiNode) {
	Type i32 = intTy(32);
	Type i1 = intTy(1);

	// Create variables and constants
	Term condVar = var(i1, Ref("cond"));
	Term paramA = var(i32, Ref("a"));
	Term paramB = var(i32, Ref("b"));
	Term resultVar = var(i32, Ref("result"));
	Term xVar = var(i32, Ref("x"));
	Term yVar = var(i32, Ref("y"));

	vector<Term> params = {condVar, paramA, paramB};
	vector<Inst> body = {// entry block
						 block(Ref("entry")), br(condVar, Ref("then"), Ref("else")),

						 // then block
						 block(Ref("then")), assign(xVar, Term(Add, i32, paramA, paramB)), jmp(Ref("merge")),

						 // else block
						 block(Ref("else")), assign(yVar, Term(Sub, i32, paramA, paramB)), jmp(Ref("merge")),

						 // merge block
						 block(Ref("merge")), Inst(Phi, {resultVar, xVar, label(Ref("then")), yVar, label(Ref("else"))}),
						 ret(resultVar)};

	Fn func = createTestFunction(i32, params, body);
	Fn transformed = eliminatePhiNodes(func);

	// Verify phi node was eliminated
	bool foundPhi = false;
	for (const auto& inst : transformed) {
		if (inst.opcode() == Phi) {
			foundPhi = true;
			break;
		}
	}
	BOOST_CHECK(!foundPhi);

	// Verify assignments were added before branches
	bool foundAssignmentBeforeBranch = false;
	for (size_t i = 0; i < transformed.size() - 1; i++) {
		if (transformed[i].opcode() == Assign && (transformed[i + 1].opcode() == Jmp || transformed[i + 1].opcode() == Br)) {
			foundAssignmentBeforeBranch = true;
			break;
		}
	}
	BOOST_CHECK(foundAssignmentBeforeBranch);
}

BOOST_AUTO_TEST_CASE(MultiplePhiNodes) {
	Type i32 = intTy(32);
	Type i1 = intTy(1);

	// Create variables
	Term condVar = var(i1, Ref("cond"));
	Term resultA = var(i32, Ref("resultA"));
	Term resultB = var(i32, Ref("resultB"));
	Term x1 = var(i32, Ref("x1"));
	Term x2 = var(i32, Ref("x2"));
	Term y1 = var(i32, Ref("y1"));
	Term y2 = var(i32, Ref("y2"));

	vector<Term> params = {condVar};
	vector<Inst> body = {block(Ref("entry")),
						 br(condVar, Ref("then"), Ref("else")),

						 block(Ref("then")),
						 assign(x1, intConst(i32, 1)),
						 assign(x2, intConst(i32, 2)),
						 jmp(Ref("merge")),

						 block(Ref("else")),
						 assign(y1, intConst(i32, 3)),
						 assign(y2, intConst(i32, 4)),
						 jmp(Ref("merge")),

						 block(Ref("merge")),
						 Inst(Phi, {resultA, x1, label(Ref("then")), y1, label(Ref("else"))}),
						 Inst(Phi, {resultB, x2, label(Ref("then")), y2, label(Ref("else"))}),
						 ret(Term(Add, i32, resultA, resultB))};

	Fn func = createTestFunction(i32, params, body);
	Fn transformed = eliminatePhiNodes(func);

	// Verify all phi nodes were eliminated
	bool foundPhi = false;
	for (const auto& inst : transformed) {
		if (inst.opcode() == Phi) {
			foundPhi = true;
			break;
		}
	}
	BOOST_CHECK(!foundPhi);

	// Verify both assignments were added for each branch
	int assignmentCount = 0;
	for (size_t i = 0; i < transformed.size() - 1; i++) {
		if (transformed[i].opcode() == Assign && (transformed[i + 1].opcode() == Jmp || transformed[i + 1].opcode() == Br)) {
			assignmentCount++;
		}
	}
	BOOST_CHECK_GT(assignmentCount, 2);
}

BOOST_AUTO_TEST_CASE(NestedBranches) {
	Type i32 = intTy(32);
	Type i1 = intTy(1);

	Term cond1 = var(i1, Ref("cond1"));
	Term cond2 = var(i1, Ref("cond2"));
	Term resultVar = var(i32, Ref("result"));

	vector<Term> params = {cond1, cond2};
	vector<Inst> body = {block(Ref("entry")),
						 br(cond1, Ref("then1"), Ref("else1")),

						 block(Ref("then1")),
						 br(cond2, Ref("then2"), Ref("else2")),

						 block(Ref("then2")),
						 jmp(Ref("merge")),

						 block(Ref("else2")),
						 jmp(Ref("merge")),

						 block(Ref("else1")),
						 jmp(Ref("merge")),

						 block(Ref("merge")),
						 Inst(Phi, {resultVar, intConst(i32, 1), label(Ref("then2")), intConst(i32, 2), label(Ref("else2")),
									intConst(i32, 3), label(Ref("else1"))}),
						 ret(resultVar)};

	Fn func = createTestFunction(i32, params, body);
	Fn transformed = eliminatePhiNodes(func);

	// Verify phi nodes were eliminated
	bool foundPhi = false;
	for (const auto& inst : transformed) {
		if (inst.opcode() == Phi) {
			foundPhi = true;
			break;
		}
	}
	BOOST_CHECK(!foundPhi);
}

BOOST_AUTO_TEST_CASE(SelfLoop) {
	Type i32 = intTy(32);
	Type i1 = intTy(1);

	Term cond = var(i1, Ref("cond"));
	Term n = var(i32, Ref("n"));
	Term i = var(i32, Ref("i"));

	vector<Term> params = {n};
	vector<Inst> body = {block(Ref("entry")),
						 assign(i, intConst(i32, 0)),
						 jmp(Ref("loop")),

						 block(Ref("loop")),
						 Inst(Phi, {i, i, label(Ref("entry")), Term(Add, i32, i, intConst(i32, 1)), label(Ref("loop"))}),
						 assign(cond, cmp(ULt, i, n)),
						 br(cond, Ref("loop"), Ref("exit")),

						 block(Ref("exit")),
						 ret(i)};

	Fn func = createTestFunction(i32, params, body);
	Fn transformed = eliminatePhiNodes(func);

	// Verify phi nodes were eliminated
	bool foundPhi = false;
	for (const auto& inst : transformed) {
		if (inst.opcode() == Phi) {
			foundPhi = true;
			break;
		}
	}
	BOOST_CHECK(!foundPhi);
}

BOOST_AUTO_TEST_SUITE_END()
--eliminate-phi.cpp
// Helper: Create a function that includes a phi node.
Fn createFunctionWithPhi() {
	// Create a function with one parameter (of type int32)
	vector<Term> params;
	params.push_back(var(intTy(32), "p"));

	vector<Inst> body;
	// Entry block for the function.
	body.push_back(block("entry"));

	// Construct the phi node.
	// The first operand is the variable that will receive the phi value.
	Term phiVar = var(intTy(32), "x");
	// Two possible incoming values.
	Term valL1 = intConst(intTy(32), 10);
	Term valL2 = intConst(intTy(32), 20);
	// Corresponding labels.
	Term lblL1 = label("L1");
	Term lblL2 = label("L2");
	vector<Term> phiOperands = {phiVar, valL1, lblL1, valL2, lblL2};
	Inst phiInst(Phi, phiOperands);
	body.push_back(phiInst);

	// Instead of trying to assign the phi instruction to phiVar (which is not allowed),
	// simply use phiVar directly. The phi elimination pass is expected to remove phi nodes.
	body.push_back(ret(phiVar));

	// Add additional basic blocks corresponding to the labels referenced.
	body.push_back(block("L1"));
	body.push_back(ret(intConst(intTy(32), 10)));
	body.push_back(block("L2"));
	body.push_back(ret(intConst(intTy(32), 20)));

	return Fn(intTy(32), "testPhi", params, body);
}

BOOST_AUTO_TEST_CASE(test_eliminatePhiNodes_removes_phi) {
	// Build a function that contains a phi node.
	Fn f = createFunctionWithPhi();

	// Verify that the original function indeed contains a phi instruction.
	bool containsPhi = false;
	for (const auto& inst : f) {
		if (inst.opcode() == Phi) {
			containsPhi = true;
			break;
		}
	}
	BOOST_CHECK_MESSAGE(containsPhi, "Original function should contain at least one phi node");

	// Run the phi elimination pass.
	Fn fNoPhi = eliminatePhiNodes(f);

	// Check that the resulting function has no phi nodes.
	for (const auto& inst : fNoPhi) {
		BOOST_CHECK_MESSAGE(inst.opcode() != Phi, "Phi node found after elimination");
	}
}

BOOST_AUTO_TEST_CASE(test_eliminatePhiNodes_no_change_without_phi) {
	// Build a function that does not contain any phi nodes.
	vector<Term> params;
	params.push_back(var(intTy(32), "p"));

	vector<Inst> body;
	body.push_back(block("entry"));
	body.push_back(ret(intConst(intTy(32), 42))); // Simply return a constant.

	Fn f = Fn(intTy(32), "noPhi", params, body);

	// Run the phi elimination pass.
	Fn fNoPhi = eliminatePhiNodes(f);

	// Ensure no phi instructions appear in the output.
	for (const auto& inst : fNoPhi) {
		BOOST_CHECK_MESSAGE(inst.opcode() != Phi, "Unexpected phi node found");
	}

	// Optionally, if your pass should leave non-phi functions unchanged,
	// check that the overall structure (number of instructions) remains the same.
	BOOST_CHECK_EQUAL(fNoPhi.size(), f.size());
}

BOOST_AUTO_TEST_CASE(test_convert_to_ssa) {
	// Create a simple function:
	//   int foo(int x, int y) {
	//       x = add(x, y);
	//       return x;
	//   }
	//
	// In our IR, function parameters are represented as Var terms.
	Type int32 = intTy(32);
	Ref x_ref("x");
	Ref y_ref("y");

	// Parameters (as Var terms).
	Term x = var(int32, x_ref);
	Term y = var(int32, y_ref);
	vector<Term> params = {x, y};

	// Build the function body:
	// 1. Assignment: x = add(x, y)
	// 2. Return: ret(x)
	vector<Inst> body;
	// Create an add term: add(x, y)
	Term addExpr = Term(Add, int32, x, y);
	// The assign instruction writes to x.
	body.push_back(assign(x, addExpr));
	// Return x.
	body.push_back(ret(x));

	// Create the function 'foo'
	Fn foo = Fn(int32, Ref("foo"), params, body);

	// Convert the function to SSA form (lowering mutable variables into allocas).
	Fn ssa = convertToSSA(foo);

	// Now verify the following expected properties:
	// - There is an alloca for each parameter (x and y) inserted at the beginning.
	// - The assign to x has been converted into a store (storing the new value into x’s alloca).
	// - The return instruction uses a load from x’s alloca instead of x directly.
	bool foundAllocaX = false;
	bool foundAllocaY = false;
	bool foundStoreForX = false;
	bool retUsesLoadForX = false;

	for (size_t i = 0; i < ssa.size(); ++i) {
		Inst inst = ssa[i];
		switch (inst.opcode()) {
		case Alloca: {
			// The first operand of an alloca is the pointer variable.
			Term ptr = inst[0];
			if (ptr.ref() == x_ref) {
				foundAllocaX = true;
			}
			if (ptr.ref() == y_ref) {
				foundAllocaY = true;
			}
			break;
		}
		case Store: {
			// In our convention, a store's second operand is the pointer.
			Term ptr = inst[1];
			if (ptr.ref() == x_ref) {
				foundStoreForX = true;
			}
			break;
		}
		case Ret: {
			// For a return, the operand should now be a load (if it was a variable usage).
			Term retOp = inst[0];
			if (retOp.tag() == Load) {
				retUsesLoadForX = true;
			}
			break;
		}
		default:
			break;
		}
	}

	BOOST_CHECK(foundAllocaX);
	BOOST_CHECK(foundAllocaY);
	BOOST_CHECK(foundStoreForX);
	BOOST_CHECK(retUsesLoadForX);
}
--eliminate-phi2.cpp
