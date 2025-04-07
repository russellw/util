#include "stdafx.h"
#include "etc.h"

#include <set>
using std::set;

namespace {

struct mpq_less {
	bool operator()(const __mpq_struct& x, const __mpq_struct& y) {
		return mpq_cmp(&x, &y) < 0;
	}
};

set<__mpq_struct, mpq_less> numbers;

}

const __mpq_struct* intern(mpq_t x) {
	mpq_canonicalize(x);

	auto r = numbers.insert(*x);
	if (!r.second)
		mpq_clear(x);
	return &*r.first;
}
