#include "stdafx.h"
#include "etc.h"

static unordered_map<const char*, int> name_functions;
vec<function*> functions;

function::function(const char* name)
	: base(functions.size(), 0), name(name) {}

function::function(const char* name, int type, vec<int> params)
	: base(functions.size(), 0), name(name), type(type), params(params) {}

int get_function(const char* name) {
	auto& i = name_functions[name];
	if (!i) {
		i = functions.size();
		functions.push_back(new function(name));
	}
	return i;
}
