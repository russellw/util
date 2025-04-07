#include "stdafx.h"
#include "etc.h"

vec<const Type*> types;

namespace std {

template<> struct hash<Type> {
	size_t operator()(const Type& x) const {
		return Hash64WithSeed((char*)x.args.data(), x.args.size() * sizeof(Type*), (uint64_t)x.name);
	}
};

}

inline bool operator==(const Type& a, const Type& b) {
	return a.name == b.name && a.args == b.args;
}

static unordered_set<Type> set;

const Type* get_type(const char* name, vec<Type*>& args) {
	auto r = set.emplace(name, args, (int)types.size());
	if (r.second) {
		if (types.size() == 1 << type_bits) {
			fprintf(stderr, "too many types\n");
			exit(1);
		}
		types.push_back(&*r.first);
	}
	return &*r.first;
}
