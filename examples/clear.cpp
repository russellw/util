#define _CRT_SECURE_NO_WARNINGS

// C++ standard library
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
using std::cerr;
using std::cout;
using std::hash;
using std::ostream;
using std::pair;
using std::runtime_error;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;

struct Batch{
	vector<string>v;
};

Batch batch;

int main(){
	batch=Batch();
	return 0;
}
